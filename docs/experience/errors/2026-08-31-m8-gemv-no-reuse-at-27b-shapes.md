# The sm70 M-row fp4 GEMV only reuses its tile at one shape — 2026-08-31

> Status: Found, not fixed. This is the speculation blocker.

## Context

A speculative verify replay costs 271 ms where a plain decode replay costs
40.9 ms. Two earlier diagnoses (draft outside the graph; GDN state
gather/scatter) were both wrong. `scripts/profile_verify_replay.py` attributes
it per kernel and the answer is unambiguous:

| W | replay | GEMV kernel | µs/call | calls | GEMV total |
|---:|---:|---|---:|---:|---:|
| 1 | **40.9 ms** | `linear_fp4_gemv_sm70` | 64.5 | 497 | 32.1 ms |
| 2 | 271.0 ms | `linear_fp4_gemv_sm70_m` | 507.9 | 497 | 252.4 ms |
| 4 | 271.7 ms | `linear_fp4_gemv_sm70_m` | 507.5 | 497 | 252.2 ms |
| 8 | 271.6 ms | `linear_fp4_gemv_sm70_m` | 507.5 | 497 | 252.2 ms |

GDN was 1.9–2.9 ms and attention 1.7 ms across all widths. The entire 230 ms
step is the GEMV, and it is **flat in W** — W=2 pays the full M=8 tile.

## Root Cause

`linear_fp4_gemv_sm70_m` is supposed to load and decode each weight tile ONCE
and reuse it across M rows (that is its whole reason to exist). It does at one
shape and not at the shapes the 27B actually runs (`scripts/ab_m8_reuse.py`):

|  N | K | M=1 µs | M=8 µs | ratio | µs/row | reuse |
|---:|---:|---:|---:|---:|---:|:--|
| 4864 | 4864 | 274.3 | 452.4 | **1.65** | 56.5 | yes |
| 12288 | 5120 | 117.1 | 728.4 | 6.22 | 91.0 | NO |
| 17408 | 5120 | 135.9 | 1019.3 | 7.50 | 127.4 | NO |
| 5120 | 17408 | 154.4 | 968.2 | 6.27 | 121.0 | NO |

At the 27B shapes M=8 is **worse per row than M=1** (91–127 µs vs 117–154 µs for
a whole single-row pass), so the M-row path is a net loss there, not a win.

N=K=4864 is exactly the shape
`wins/2026-08-30-sm70-fp16-twiddle-gemv.md` benchmarked (it reports M=1 252.7 µs
→ M=8 453 µs, 1.79×, and this reproduces at 1.65×). The kernel was validated on
a square shape and never timed at the production projections.

Not yet explained. Ruled out by measurement or arithmetic:
- **Weight re-decode** — the `ld.global.nc.v2.u32` + `tl_fp4_decode8_f16` sit
  OUTSIDE the `for m` loop (kernels_linear.py:812-818), so W is decoded once.
- **Occupancy** — every shape grids >1200 blocks for 80 SMs.
- **f32→f16 converts** — 1280 per thread at both K=4864 and K=5120.
- **X re-reads across blocks** — 0.2 GiB at 4864 vs 0.7 GiB at 17408 = 0.8 ms of
  bandwidth, against a ~1.0 ms per-call gap. Same order, so a contributor, but it
  does not account for 6-7×.

Next step is ncu on the two shapes rather than more arithmetic.

## Why it matters

This one kernel is the entire speculation story on sm70:

- Fixed at its 1.65× scaling, a W=8 verify would be ~73 ms. At 62% acceptance,
  depth 3 gives 2.24 tokens → 30.7 tok/s, already past dense 25.8.
- It also sets the M=8 batch-decode path, so B=2..8 serving pays the same
  penalty. The B=8 aggregate number was measured with 8 separate sequences and
  so never isolated this.
- W=1 is untouched (different kernel), which is why today's dense gains are real.

## Corrections to earlier entries

- `errors/2026-08-31-draft-step-outside-graph.md` — the draft loop does run
  outside the graph, but that is not the dominant cost.
- `errors/2026-08-31-spec-blocked-on-gdn-state-path.md` — the GDN gather/scatter
  is real (4.73 ms/layer was my arithmetic from the flat W step) but the profiler
  shows GDN at 1.9–2.9 ms/tick total. That entry's root cause is **wrong**; the
  step is the GEMV kernel switch.

## Rule

Benchmark a kernel at the shapes it will actually be called with. A square
validation shape hid a 6-7× regression at every real projection of the model,
in a kernel whose entire purpose is the reuse that stops working there. When a
cost is flat in the parameter that should drive it (W=2 costs what W=8 costs),
the parameter is not what selects the work — find the switch.
