# Pod verification 2026-08-27 — where the 27B stands, and the #1 blocker

> Read this first next session. Two GPU runs done (before and after the
> rmsnorm grid fix). Verdict: **the model computes wrong logits — fix that
> before any perf work.** Every throughput number is denominated on a model
> that outputs garbage.

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
the grid. The 27B has **never had a reference-logits check** — this is the C1
class the audit named ("no numerical ground truth anywhere in this project").
Candidates, in order of suspicion, given the real config above:

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
micro=32 −35%; see `errors/2026-08-27-micro-size-k-thesis-rejected.md`). New order:

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
