# The 27B H20 kernel roofline, measured: every %bound in (0,100] — sm90 card 2, 2026-09-11

Runbook step 3 ([PENDING-REMOTE-CARDS](../PENDING-REMOTE-CARDS.md) §3): the
`bench --kernels` table with measured ms / roofline bound / %bound for the
Qwen3.8-27B on one H20, decode B=1/B=8 against 4096 pooled context and one
S=4096 prefill. Raw table:
[2026-09-11-h20-roofline-step3.log](2026-09-11-h20-roofline-step3.log).

```
tilerl bench --model qwen38-27b --kernels --checkpoint /work/Qwen3.8-27B-NVFP4 --batches 1,8
tilerl bench --model qwen38-27b --kernels --checkpoint /work/Qwen3.8-27B-NVFP4 --prefill 4096
```

Floors (same card, [card-2 rows PR] / measurements.jsonl, ids below):
hbm 3312.71 GB/s, bf16 136.45 TFLOP/s, **fp8 276.58 TFLOP/s** (2.03x bf16).

## Result: every timed row is physically in range

| tick | TIMED ms | Σ bound ms | TIMED %bound | TICK bytes | TICK flops |
|---|---:|---:|---:|---:|---:|
| decode B=1 s=4096 | 53.2 | 6.6 | 12.4% | 22,359,395,456 | 72,185,630,720 |
| decode B=8 s=4096 | 91.0 | 6.6 | 7.3% | 25,634,411,648 | 577,485,045,760 |
| prefill S=4096 | — | — | max row 77.7% | — | — |

Highest single-row %bound: prefill lm_head **77.7%**, nvfp4 down_proj
**76.6%**, gate_proj/up_proj 70.6/71.2%; fp8 GEMMs 42–67%; decode peaks at
lm_head 77.0% (B=1) / 43.3% (B=8). Nothing exceeds 100 — the table is a
physical roofline, not an over-claim. Fused attention/GDN/norm rows have no
GEMM timing fixture and render ms `pending` (their bounds still print).

## Three instrument errors the >100 gate and the run exposed

The first correct table did not appear until three bugs were fixed; each was
caught because the numbers contradicted the ledger, not by a hunch:

1. **Wrong-model timing (#499).** The original `bench --kernels` ran without
   `--model qwen38-27b`, timing tiny shapes against 27B rows and printing
   %bound in the thousands. Fix: the checkpoint-model guard (`#499`) refuses a
   checkpoint whose config.json disagrees with `--model`, on both
   `serve --dry-run` and `bench --kernels`; the 27B nests its scalars under
   `text_config`, which the guard resolves exactly as `load_hf` does.
2. **Decode timed as a fat prefill GEMM (#505).** The renderer passed the
   context s=4096 into the timer for decode rows, timing M=b·s GEMMs while
   pricing the M=b one-token GEMVs. It made the decode tick read 629 ms; the
   real GEMV tick is 53.2 ms. Fix: the timer takes the launch M (s on prefill,
   1 on decode), and a shape pin asserts the built GEMM's
   `flops == 2·M·N·K` against the priced row.
3. **An NVFP4 GEMM bounded by the bf16 ceiling (#505).** Against the bf16
   measured peak the prefill nvfp4 gate_proj read **133.8%** — impossible, so
   the ceiling was wrong, not the kernel. Reading the sm90 kernel: the prefill
   `linear_fp4` quantizes X per-token to e4m3 and issues **fp8 WGMMA (w4a8)**,
   while M≤8 decode dequants to bf16. The MMA dtype is declared once in the
   kernel registry (`LINEAR_MMA_BANDS`) by op and launch M, and the roofline
   ceiling follows that, not the weight face. Against the measured fp8 peak the
   same row reads 70.6%.

## Rule

A roofline ceiling belongs to the **MMA instruction's dtype**, read from the
kernel registry — never inferred from the weight storage face. And a timing
instrument must time the priced launch: same M, same kernel, same ceiling; a
shape pin on `2·M·N·K` and a hard `%bound > 100` abort are what turned the
three bugs into loud failures instead of a publishable table.

## Results

| date | commit | machine | target | result |
|---|---|---|---|---|
| 2026-09-11 | c1e84535 code; card-2 rows #508 | H20 physical card 2 (ordinal 0 masked) | sm90 | all %bound in (0,77.7]; decode 53.2/91.0 ms; floors 3312.71 GB/s, 136.45/276.58 TFLOP/s |

Floors ids: hbm `6c0659c91ab1`, bf16 `67f3ea83c9aa`, fp8 `e729ffedf310`,
clean rows at c1e84535. Pending-remote: timing fixtures for the fused
attention/GDN/norm kernels (their ms columns), and a B=8 prefill column.
