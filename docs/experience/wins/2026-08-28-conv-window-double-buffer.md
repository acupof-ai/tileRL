# Conv window double-buffered in the pool — cuda(H20), 2026-08-28

> Status: Shipped

## Context

The last per-layer glue in the decode tick: the fused GDN kernel returned a
new conv window and torch gathered/scattered it (14 index kernels per 8
layers, ~0.4 ms/tick). It could not be shifted in place: a window's q/k
columns are read by every block of the GQA group, so an in-place shift races.

## What Worked

`LinearStatePool.conv_windows` is `[S, L, 2, W, D]` with `win_parity[S]`:
the kernel reads plane `Par[slot]`, writes `1 - Par[slot]`, and the model
flips the slot's parity once per tick after the last GDN layer
(`Backend.flip_window_parity`). Every other path — chunk prefill, the
reference, prefix snapshots (`window_snapshot`) and restores
(`window_restore` → plane 0, parity 0) — indexes through the parity, which
stays 0 off the sm90 decode path. First cut declared `S, L` after their use
(UnboundLocalError); the second run was contended by another tenant on
GPU7 (rows at 0.39×, verify refused to start) — same rule as before, one
job per GPU.

Quiet host: d512 B=1 **83.9 → 87.2 tok/s** (11.5 ms/tick, past Arle's
84.5), d2048 76.7 → 79.8, d8192 56.5 → 58.7, B=8 flat, prefill −1.5%
(inside the gate), verify 1–3 PASS.

## Appendix: batched M≤8 GEMVs (same commit)

`make_linear_fp4_gemv_mx` / `make_linear_fp8_gemv_mx` (`tl_fp{4,8}_gemv_mx<G,MX,R>`):
one warp per R=4 weight rows, each lane keeps the MX=8 activation rows of
its 16-elem K-slice in registers and walks its rows, so X traffic is /R —
the 2026-08-26 small-M GEMV reloaded X per warp and was L1-bound. Routed for
2 ≤ M ≤ 8 instead of the padded WGMMA w4a8 / fp8 paths. B=8 aggregate
215 → 219 (d512), 204 → 216 (d2k), 164 → 171 (d8k); B=1 untouched; verify
PASS. Smaller than the 2× the per-byte argument promised: the B=8 tick is
now bound somewhere else (attention / GDN at M=8) — profile it at B=8 next.

## Rule

Block-shared state cannot be updated in place inside one launch; double
buffer with a parity the tick flips, and route every reader through the
parity so snapshots stay exact.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-08-28 | see bench-baseline.json | H20 gpu7 | cuda/sm90 | Qwen3.8-27B-NVFP4 | 0.55 | 11.5 (B=1, d512) | **87.2** B=1; 215.1 agg B=8 |
