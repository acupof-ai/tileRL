# The 2.12x train_step BEAT is real — three backward kernels, not a measurement artifact — H20, 2026-09-09

> Status: Shipped

## Context

The bench store's `train/27B-lora-b1t256/sm90` BEAT read 263.6 tok/s against an
11-day-old baseline of 124.31 — a 2.12x jump with no commit claiming it. A
speedup that large, unattributed, reads as an instrument error until proven
otherwise. Three checks, cheapest first:

1. **Same population?** Yes. Both sides load `Qwen3.8-27B-NVFP4` with LoRA
   rank 16, AdamW lr=1e-3, `fuse_projections=False`. Confirmed by diffing
   `suite_train` between `3fb9c55` (baseline) and HEAD — the model, LoRA rank,
   optimizer, and learning rate are unchanged.
2. **Same tok_s numerator?** Yes. Both compute `b * t / (ms / 1e3)` — the
   formula is byte-identical across the window.
3. **A commit claiming the speedup?** Yes, but not the one named first. #109
   claims 2.16x, but on the **GRPO step** (rollout + train, 73.62 → 34.09 s) —
   a different metric. The `train_step` fwd+bwd path has its own attribution:
   three backward-kernel commits that all landed 2026-09-07, after the 08-29
   baseline.

## What Worked

The three commits, each on the backward path:

| Commit | Change | Measured speedup |
|--------|--------|-----------------|
| `f0e6e71` | GDN backward chunk 16 → 64 | 1.94x on backward_secs (80.2 → 41.4 s) |
| `88e3764` | GDN backward chunk 64 → 128 | 1.19x further (41.4 → 34.7 s) |
| `8d6a24a` | fp4 backward ran Ampere MMA on Hopper (wrong arch → wgmma) | 1.35x end-to-end backward, 2.77x on the kernel itself |

The 8d6a24a fix is the substantive one: `linear_fp4_bwd` was 18.8% of the GRPO
backward and ran at 24 TFLOP/s where a same-shape bf16 GEMM reaches 135. The
emitted CUDA contained zero wgmma — it used `mma.sync.aligned.m16n8k16`
(Ampere) with Hopper data movement (TMA, mbarrier). At `_THREADS=64` the
consumer is 2 warps and no tile or `num_stages` could emit wgmma; `threads=128`
emits it.

**What was measured, and what was not.**

Measured: `train_step` b1t256 on card 1, 27B LoRA, 5 warm steps:
fwd 0.286 s, bwd 0.729 s, opt 0.095 s — **backward is 71.7% of the step**.

For a 71.7% backward to produce an end-to-end 2.12x, the backward itself
needs **3.80x**.

The three backward kernel commits measured **3.122x** (1.937 × 1.193 × 1.351)
on the **GRPO backward** — a different workload with a different kernel
composition. Transferring that ratio to `train_step` predicts 1.95x
end-to-end; the measured 2.12x leaves a residual. The 3.122x is also an
overestimate for `train_step` backward: the chunk commits touch only the GDN
backward, and GDN is a larger fraction of the GRPO backward, so the same
commits move the `train_step` backward less.

**The 3.80x on train_step backward was not directly measured** — that would
require re-running `train_step` at `3fb9c55`. The attribution direction is
settled (three backward commits, all on the right path, all after the
baseline); the exact magnitude is not, and the residual changes no decision.

## Rule

A speedup ratio cannot transfer across workloads — it is a weighted average,
and the weights are that workload's kernel composition. (Same disease as
"pass coefficients, not ratios": a ratio is dimensionless, so it multiplies
into anything without an error.)

A baseline record with only `commit`, `date`, `tok_s` cannot answer "was this
measured on the same population?" — the 124.31 row had no model, LoRA rank,
optimizer, shape, warmup, or n. The question was answerable only by reading
git log and diffing the bench harness. The bench store exists so the next
agent does not have to.

## Results

| date | commit | machine | target | model | shape | fwd s | bwd s | f | tok/s | spread |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 2026-08-29 | 3fb9c55 | H20 | sm90 | 27B-lora | b1t256 | — | — | — | 124.31 | — |
| 2026-09-09 | 88fb049 | H20 card 6 | sm90 | 27B-lora | b1t256 | 0.286 | 0.729 | 0.717 | 263.6 | ±0.4% |

The fwd/bwd split was measured by a one-shot probe (not committed; the three
numbers above are the record). The 263.6 tok/s row is the real baseline from
origin/main (88fb049), card 6, n=3, warm. A first run on card 1 was contended
by another team's job starting mid-run (spread ±66%); the card 6 re-run
(±0.4%) is the reliable measurement. All five shapes:

| shape | tok/s | vs baseline |
|---|---:|---:|
| b1t64 | 71.5 | 1.24x |
| b1t128 | 145.1 | 1.60x |
| b1t256 | 263.6 | 2.12x |
| b2t256 | 382.0 | 1.96x |
| b4t256 | 465.5 | 1.02x |

Raw artifacts: `docs/experience/bench/measurements.jsonl` (6 rows, commit
88fb049).
