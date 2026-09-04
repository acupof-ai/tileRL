# Pod verification 2026-08-27 — where the 27B stands, and the #1 blocker

> **RESOLVED 2026-08-27 evening: check 2 is GREEN.** The wrong-logits root cause
> was zero-centered RMSNorm (`y = x_normed*(1+weight)`), not rope/mrope. Fixed
> (fold +1 at load, -1 at save). Greedy decode now correct: "France"→" Paris…",
> fib→correct code, "17+25"→"42". Full writeup:
> `docs/experience/wins/2026-08-27-zero-centered-rmsnorm.md`. Perf work is now
> unblocked and denominated on a correct model. History below kept for the
> method (external-bisection beats cosine probes and internal parity).

## Checkpoint config (from /data00/Qwen3.8-27B-NVFP4/config.json)

`Qwen3_5ForConditionalGeneration`, a VLM (visual tower ignored by the loader).

- hidden 5120, intermediate 17408, 64 layers, vocab 248320
- 24 attn heads / 4 KV heads / **head_dim 256** (not 128), partial_rotary 0.25 → rotary_dim 64
- layer_types: every 4th layer full_attention (idx 3,7,…,63 = 16 layers), rest GDN (48)
- GDN: 16 key heads × 128, 48 value heads × 128, conv kernel 4
- tie_word_embeddings False; rope_theta default; rope_scaling None
- **quant split** (`quantization_config`, mixed-precision):
  - fp8 e4m3 (per-channel W, per-token A): attn q/k/v/o, GDN in_proj_qkv/z/out, lm_head, **layers 56-63 mlp**
  - nvfp4 (group 16, e4m3 scales): **layers 0-55 mlp** gate/up/down
  - bf16 (unquantized): embed, GDN in_proj_a/in_proj_b, norms, conv1d

## Check results (commit 69c3914, grid fix applied)

| check | result | evidence |
|---|---|---|
| 1 memory | PASS | 27.50 GiB resident = 22.76 params + 4.74 M2 f32 cast; KV pool 1.00 GiB (M3 dense-pool fix confirmed on real HW); 305 quantized tensors, 0 masters |
| 2 logits | **FAIL** | all 3 prompts → id 158949 (~46/48). Degenerate on SHORT prompts too |
| 3 e4m3 | PASS | M=1 fro-relerr 0.0017 (weight path exact); M=8/512 ~0.035 (end-to-end, expected) |
| 4 throughput | FAIL | decode B=1 51.0 tok/s (0.97× of 52.6); B=8 141 tok/s; prefill 0.84× |

## The grid bug (FIXED, 69c3914)

`rmsnorm_partial/apply` put M on grid.y (cap 65535); a 27B prefill with
M>65535 overflowed → `CUDA_ERROR_INVALID_VALUE`, sticky, poisoned the forward.
The first run showed check 4 as ERROR (`grid=(1,73008,1)`). After swapping M to
grid.x, check 4 runs (`decode_graph_on: True`) and produces a number. **Fix
confirmed necessary and correct** — but it only un-masked check 2.

## The #1 blocker: check 2 (wrong logits) is a SEPARATE bug

Degeneration to id 158949 happens on short prompts (no overflow), so it is not
the grid. health_probe confirms the forward is **finite and non-degenerate**
(embedding norm 3.8, logits norm 1991, 3.8M distinct logit values) — so it is
**wrong math, not a NaN/blowup**: plausible logits that argmax to a junk token.
The 27B has **never had a reference-logits check** (audit C1).

## UPDATE 2026-08-27 (evening): rope fixed but NOT sufficient; bug is wiring/weights

The rotate_half fix (3c4d814) was a real bug but did NOT fix check 2 — pod
re-run still collapses all 3 prompts to id 158949. So there is a SECOND bug.
Systematic localization since:

**Every op passes parity at REAL 27B dims** (`scripts/op_parity.py`, layer 0/3):
- rmsnorm[hidden 5120] 5.4e-8, rmsnorm[head_dim 256] 1.2e-7
- linear_fp8 2.6e-2 (= the e4m3 activation-quant tolerance, same as check 3)
- silu_mul 4.4e-8, embedding 0.0
- (linear_fp4 already verified by check 3: M=1 fro-relerr 0.0017)

**Attention and GDN math match the HF source by inspection** (transformers
5.6.0 `models/qwen3_5/modeling_qwen3_5.py`, on the pod):
- attn q_proj gate split: HF `chunk(view(..., H, 2*head_dim), 2, -1)` = tileRL's
  `reshape(b,t,hq,2,d)` per-head [query|gate]. ✓ Order q_norm→rope→attn→
  ×sigmoid(gate)→o_proj matches (`attn_output_gate:True`, `output_gate_type:swish`
  = sigmoid gate). ✓
- GDN: conv→silu→split[k,k,v], q/k l2norm + scale `1/sqrt(key_dim)` on q only,
  GQA `repeat_interleave(nvh//nkh)`, gated RMSNorm `norm(core)*w*silu(z)` — all
  match tileRL's `gdn_forward`. ✓
- **H2 (mrope interleaved) RULED OUT**: HF `apply_interleaved_mrope` picks freqs
  per axis by `slice(offset,len,3)`, but for TEXT position_ids T=H=W are equal,
  so every axis reads the same freq → collapses to plain 1D RoPE. Not the bug.

The cross-prompt cosine probe (`health_probe.py`, rewritten) shows the residual
stream's cross-prompt cosine climbing embed 0.001 → L6 0.99 → plateau 0.998, and
per-sublayer added-delta cosine ~1.0 from L1 on. BUT this signal is **not proof**
of a bug — real transformers have massive-activation shared directions with
cos>0.99. It only says "prompts converge", consistent with either a bug or
normal massive activations.

**Conclusion: the bug is wiring or weight interpretation, not a kernel.** Next:
`scripts/hf_reference.py` runs the checkpoint through HF transformers for the
ground-truth first token + per-layer hidden norms, to bisect tileRL against a
known-correct forward. Prime remaining suspects (all wiring, invisible to
op-parity): fp8/fp4 layer-split routing, the fused-projection slice boundaries,
q/k/v/gate slice order in the real (unfused) load path.

### Two concrete, testable RoPE hypotheses (ranked) — 2026-08-27

The config carries `rope_parameters = {mrope_interleaved: True, mrope_section:
[11,11,10], partial_rotary_factor: 0.25, rope_theta: 1e7, rope_type: default}`.
Two independent ways tileRL's plain `make_rope` can be wrong for this:

**H1 — rotate_half vs adjacent-pair convention (most likely).** tileRL's
`make_rope` (kernels.py) rotates ADJACENT pairs `(2d, 2d+1)` — the GPT-J /
"interleaved" convention. Qwen/Llama use `rotate_half`: pair dim `d` with
`d + rot/2` (the GPT-NeoX / "half-split" convention). If the checkpoint expects
half-split and tileRL rotates adjacent pairs, EVERY attention layer rotates
wrong → exactly this plausible-but-wrong-logits signature. **Test:** diff
`backend.rope(q)` against a HF `apply_rotary_pos_emb` (rotate_half) on one
[1,1,1,64] vector; if they differ, this is it. Fix: change the pairing in
`make_rope` to `(d, d+rot/2)`, re-verify parity.

**H2 — M-RoPE interleaved frequency layout.** For TEXT-ONLY, M-RoPE with
T=H=W=arange is bit-identical to 1D RoPE ONLY if the sections slice the
frequency spectrum the same way tileRL pairs it. `mrope_interleaved: True`
means the 3 axes' frequencies are INTERLEAVED across the spectrum, which can
reorder which freq multiplies which dim even at equal positions. Confirmed real
trap: a sibling checkpoint ships `patch_mrope_text_fallback.py`
(github.com/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash). Less likely than H1
because equal positions neutralize the *axis* difference, but the *interleave*
may survive. **Test:** only after H1 is ruled out.

`_validate_hf_config` passes this checkpoint because it treats `rope_type
"default"` as fine and never inspects `mrope_interleaved` — add that check once
the convention is settled.

## OLD candidate list (superseded by H1/H2 above)

1. **head_dim 256 with partial_rotary 0.25 (rotary_dim 64).** The old assumed
   config was head_dim 128; the RoPE / attention path may mis-handle
   rotary_dim < head_dim at 256. First thing to probe.
2. **The fp8/nvfp4 layer split.** Layers 56-63 MLP are fp8, 0-55 are nvfp4 —
   if the loader routes a layer to the wrong kernel, output corrupts.
3. **GDN a/b are bf16, qkv/z/out are fp8** — a dtype-routing error in the GDN
   projection load.
4. `global_divide=True` NVFP4 dequant convention — never validated vs a
   reference framework.

## How to localize it

`scripts/health_probe.py` (added 7db31fa) runs one prefill and reports per-probe
norm/finite/distinct. Run on the pod:
`PYTHONPATH=src TILERL_TARGET=cuda python3 -u scripts/health_probe.py /data00/Qwen3.8-27B-NVFP4 --gpu 7`
It currently probes embedding→logits (coarse). To pin the exact layer, extend it
to hook each layer's residual output — the first layer whose norm explodes or
flatlines is the bug site. Then compare that layer's op against the torch-eager
reference at the real 27B dims.

## Levers, re-prioritized after the sweep

The micro_size_k load-width thesis is **dead** (measured: micro=16 −10%,
micro=32 −35%; see `docs/experience/errors/2026-08-27-micro-size-k-thesis-rejected.md`). New order:

0. **Fix check 2.** Nothing below matters until the model is correct.
1. **scale dtype f32→bf16** — the #1 GEMV lever now (¼ of the fp4 stream; Gate B's 6 µs gap).
2. **SR register-resident B kernel** (Marlin-shaped) — the only route to Marlin's 38.9 µs; CAN verdict stands.
3. **sm90 bf16-IO cell** (P1+P2+M2, ~2.3 ms + 4.7 GiB) — 5 kernels; blocked on tilelang needing dtype as a source literal (not a var), so it is 5 separate factories, not a parameterized one. Draft it cleanly, not from the workflow's duplicated version.
4. oscale fold into the GEMV epilogue (P3, 0.3 ms).

## What did NOT work / was reverted

The stopped workflow's bf16-cell draft (5 duplicated `make_*_bf16` factories +
backend/registry wiring) was reverted. Reason: tried to dedupe it via an `io`
dtype param, but tilelang 0.1.13's eager builder reads the `T.Tensor` dtype
from the function SOURCE TEXT — a closure var or default arg both raise
`NameError: name 'io' is not defined` at parse. Every `T.Tensor` dtype in the
codebase is a string literal for this reason. The bf16 cell is genuinely 5
factories; write them cleanly next pass and gate on pod parity.
