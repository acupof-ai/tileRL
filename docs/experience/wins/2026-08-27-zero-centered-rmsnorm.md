# Zero-centered RMSNorm — the 27B wrong-logits root cause — cuda(H20), 2026-08-27

> Status: Shipped

## Context

The Qwen3.8-27B-NVFP4 produced garbage on the pod: greedy decode collapsed all
prompts to one junk token (id 158949), check 2 of `verify_h20_fp4.py` FAIL.
Logits were finite and plausibly-normed (~1991) — wrong math, not a NaN. The
27B had never had a reference-logits check (only tiny-model parity, which does
not exercise the real config).

## What Worked

**Qwen3.5 uses zero-centered RMSNorm: `y = x_normed * (1 + weight)`.** The HF
`Qwen3_5RMSNorm.forward` is `output * (1.0 + self.weight)`, and its weight is
initialized to ZEROS, not ones. tileRL applied plain `x_normed * weight` on all
five non-gated norm sites (input_norm, post_attn_norm, q_norm, k_norm,
final_norm), so every norm in the stack scaled the residual wrong → a ~2×
per-layer blowup. The GDN gated norm (`Qwen3_5RMSNormGated`) is NOT
zero-centered (weight init ones) and was already correct.

Fix: fold the `+1` into the five zero-centered norm weights at **load time**
(`model.py` load_hf), with the inverse `-1` in `save_hf` so the on-disk format
stays HF-canonical. Zero kernel change, zero hot-path change, tape/gradcheck
untouched. 3 lines of real logic.

After the fix, greedy decode is correct for the first time:
- "The capital of France is" → " Paris. The capital of Germany is Berlin. …"
- Fibonacci prompt → a correct Python function
- "17 plus 25?" → "42"

Checks 1/2/3 all PASS. Check 4 (throughput) unchanged at 0.97× — now
denominated on a CORRECT model.

## How it was found (the method that worked)

Cosine-collapse probes (`health_probe.py`) were **inconclusive** — real
transformers have massive-activation shared directions with cross-prompt
cos>0.99, so "prompts converge" does not prove a bug. Op-parity at real dims
(`op_parity.py`) showed every kernel matches its own reference (rmsnorm 5e-8,
linear_fp8 2.6e-2 = quant tol, silu/embed ~0) — the reference itself was wrong,
so kernel-vs-reference could never catch it.

**External ground truth was decisive.** `hf_reference.py` runs the checkpoint
through HF transformers 5.6.0 (" Paris", logits norm 1436), dumps per-layer
last-token hidden states; `hf_bisect.py` runs tileRL's per-layer forward on the
same real token ids and diffs element-wise. Pre-fix: L0 already diverged
(|tl|=24.9 vs |hf|=12.0, cos 0.86), error from layer 0. Reading HF's
`modeling_qwen3_5.py` at the first diverging op revealed the `(1+weight)`
convention. Post-fix: L0–L62 match (cos ≥0.994, norms track exactly).

## Rule

Norm conventions are per-architecture and INVISIBLE to internal parity (kernel
and reference share the bug). Qwen3.5/3.6 RMSNorm is zero-centered
`(1+weight)`; the GDN gated norm is plain. For any new checkpoint family,
bisect the first prefill against an external reference (HF transformers) at real
dims before trusting logits — tiny-model parity and cosine probes cannot see a
shared-convention bug.

## Results

| date | commit | machine | target | model | check 2 (logits) | decode ms/tick | throughput tok/s |
|---|---|---|---|---|---|---:|---:|
| 2026-08-27 | (this) | H20 gpu7 | cuda | Qwen3.8-27B-NVFP4 | PASS (" Paris"/fib/"42") | 19.53 | 51.2 (B=1), 141 (B=8) |

Raw artifacts: `/work/verify_fix.log`, `/work/bisect_fix.log`, `/work/hf_ref.pt`.
Instruments: `scripts/hf_reference.py`, `scripts/hf_bisect.py`, `scripts/op_parity.py`.
