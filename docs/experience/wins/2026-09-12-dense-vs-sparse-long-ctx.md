# Dense vs sparse long-context on the 27B — H20 and V100, 2026-09-12

> Status: **two dense rows measured; three sparse rows pre-registered with their
> launch commands, blocked on #518 merge + sync.** The V100 256k sparse point is
> a pre-registered *prediction* (below) for the overnight run to hit or miss,
> not a measurement. Driver `scripts/bench_dense_point.py`, tick-boundary CUDA
> timing, held spans, `--eager`, 64 decode tokens (skip 16), sparse_k=128,
> bounds scorer.

## Context

One wall-clock question: at long context on a small card, does Quest sparse KV
(attend a fixed 128 selected pages + the causal own span, unit F #518) let a
context run that dense cannot fit, and what does it cost per token? Dense
attention grows with full history; sparse attention is bounded by the selected
set, so it should turn the dominant prefill term from quadratic to flat. The
two dense rows are the measured controls; the sparse rows fill in as the cards
return. Same held tokens, same harness, only `--sparse-k/--scorer/--cold-bytes`
differ.

KV: H20 holds bf16 KV (65536 B/tok); V100 sm70 runs f32 KV dense
(131072 B/tok) and f16 cold under sparse (the f16 cold path lands with #513).
Weights resident: H20; V100 28.33 GiB of 32 GiB (~4.9 GiB device free dense),
so V100 128k/256k are sparse-only regardless of speed.

## Results

Dense rows measured; sparse rows are blanks until the runs below return.

| machine | mode | ctx | prefill s | ms/tok | decode tok/s | device KV GiB | cold host GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| H20 | dense (graph) | 131072 | **88.6** | 0.676 | **58.28** | **8.02** | 0 |
| H20 | sparse k=128 (eager) | 131072 | **108.0** | 0.824 | **19.16** | 0.55 | 0 at finish |
| V100 | dense f32 (eager) | 32768 | **594.6** | 18.15 | **8.28** | **4.04** | 0 |
| V100 | sparse k=128 (eager) | 32768 | **341.3** | 10.42 | **1.252** | 1.16 | 0 at finish |
| V100 | sparse k=128 (eager) | 131072 | **2271.1** | 17.33 | **0.556** | 1.16 | 0 at finish |

Dense controls: H20 128k prefill 88.621 s / decode 58.279 tok/s (17.2 ms/tok) /
KV 8612478976 B; V100 32k prefill 594.578 s / decode 8.278 tok/s
(120.8 ms/tok) / KV 4339007488 B.

**V100 32k sparse — prefill wins, decode loses** (head 95f69fe7, eager,
64 prefill ticks, 64 decode tokens skip 16): prefill 341.335 s =
10.417 ms/tok (**1.74x faster than dense**); decode 1.252 tok/s =
**798.4 ms/tok, 6.6x SLOWER than dense** (both arms `--eager`, so unlike the
H20 rows this is an apples-to-apples eager-vs-eager gap — real, fetch-bound on
PCIe, not a graph artifact); device KV 1.16 GiB (1107-block hot
pool = 4 Quest groups × 128 + window + chunk). Cold host reads 0 at finish
because `_release` forgets the request's cold blobs; the tier cycles pages
every tick while running. The asymmetry is unit F's merge state:
`_sparse_finalize` demotes every resident page after each tick ("the
cross-tick hot set is the later perf PR"), so prefill amortizes each fetched
page across a 512-query chunk while decode re-fetches, per token, the union of
all four source groups — up to 512 pages, each `promote_keyed` ending in a
full `cuda.synchronize`, i.e. up to 512 serial ~1 MiB pinned H2D copies per
token. The 0.37–0.43 ms sparse-attention decode figure the prediction below
used was a hot-resident measure, not this demote-all state, so the decode
prediction missed by ~10x. The remedy is a pinned cross-tick hot set (the hot-pin
PR 52 is implementing), not the selector; this V100 point re-runs on that head
when it lands.

**H20 128k sparse (EAGER) is slower on prefill and decode** (head f0a45485,
card 6, 256 prefill ticks, KV_DEVICE 0.55 GiB = the 1107-block hot pool in
bf16): prefill 107.988 s = 0.824 ms/tok (**1.22x slower** than dense 0.676),
decode 19.160 tok/s = 52.2 ms/tok vs dense 17.2. **The decode rows are not
apples-to-apples: this sparse tick is the "first cut eager" path (no CUDA
graph); the dense 17.2 ms row is graph-on. Dense eager on H20 is ~48 ms
(#523, 11.68 graph vs 47.99 eager), so 52.2 ms sparse-eager ≈ dense-eager —
the decode gap is the missing graph, not sparsity or page fetches.** Mark the
sparse row eager pending a graph-captured sparse decode. On prefill, dense
bf16 WGMMA attention is cheap, so selection plus fetch overhead exceeds the
attention it removes; sparse at 128k on an H20 buys capacity past the 8.6 GiB
bf16 KV fit, not prefill latency, and at 128k it fits so dense is the choice.

**V100 128k sparse (eager)** (head f0a45485, 256 prefill ticks): completes where
dense cannot fit (dense 128k needs 17.2 GiB f32 KV against ~4.9 GiB free) — the
capacity win is real. Prefill 2271.1 s = 17.33 ms/tok (37.9 min), close to the
18.15 ms/tok dense pays at 32k: sparse holds the per-token prefill roughly flat
with context because it caps attended keys, while dense's attention grows
quadratically. Decode 0.556 tok/s = 1797 ms/tok, worse than the 32k sparse's
798 ms — under demote-all the per-token fetch set grows with selected pages
across a deeper cold tier, so sparse decode on a slow interconnect degrades
with context, the opposite of dense. This is the row the hot-pin PR must move.

**Hot-pin 32k on H20** (head 5b8df74d, card 6, eager): cross-tick residency
works — per steady decode tick **promotions = 0** (sum 0), demotions 0.1/tick,
so a stable selection moves no pages; decode 21.783 tok/s = 45.9 ms/tok. With
zero fetches the tick is still ~46 ms, which matches dense-eager ~48 ms, not
dense-graph 11.7 ms: this confirms the decode lever for sm90 is **graph-
capturing the sparse tick** (fixed-width packed table, selection as device
ops with no host sync, contiguous bounds), not batching more H2D and not
incremental page scoring (the Quest bound is ~0.07 GFLOP/tick). The 1.22x
prefill net is a separate, eager-independent question.


## Sparse launch commands

Run only after #518 merges and the pod/V100 trees sync to it.

**H20 card 7, 128k sparse** (`scripts/_staged_sparse_h20.sh`):

```bash
bash scripts/pod_run.sh --wait --lend-ref "$LEND" sparse128k 7 -- \
  TILERL_TARGET=cuda /usr/bin/python3 scripts/bench_dense_point.py \
  --source /work/tilerl-ckpt/Qwen3.8-27B-NVFP4 \
  --span /work/tilerl-ckpt/spans/held-128k.json --ctx 131072 \
  --new-tokens 64 --skip 16 --sparse-k 128 --scorer bounds \
  --cold-bytes 12884901888 --max-wait-s 2400
```

**V100 32k sparse** (pairs with the 32k dense row; `scripts/_staged_sparse_v100.sh`):

```bash
bash scripts/v100.sh run sparse32k \
  "python3 scripts/bench_dense_point.py --source $CK \
   --span \$HOME/held_32768_row0.jsonl --ctx 32768 \
   --new-tokens 64 --skip 16 --eager --sparse-k 128 --scorer bounds \
   --cold-bytes 17179869184 --max-wait-s 3000"
```

**V100 128k sparse** (no dense pair; needs the pushed 131072-token span):

```bash
bash scripts/v100.sh run sparse128k \
  "python3 scripts/bench_dense_point.py --source $CK \
   --span \$HOME/held-128k.json --ctx 131072 \
   --new-tokens 64 --skip 16 --eager --sparse-k 128 --scorer bounds \
   --cold-bytes 21474836480 --max-wait-s 7200"
```

## Pre-registered prediction — V100 256k sparse (overnight, after #513 f16 cold)

This is a prediction built from measured parts, recorded before the run so the
run can hit or miss it. Do not read it as a result. Sparse prefill attention is
bounded by the 128-page selected set (2048 keys) plus the own causal span, not
by the 262144-token history, so it is summed as a flat per-token term like the
linears rather than extrapolated quadratically.

- **Non-attention tick: 4.36 ms/tok** after #524, flat and context-independent
  (whole-tick 8k A/B: 11.64 → 4.36 ms/tok; the pre-#524 "other/linear" bucket in
  the 64k split was a flat 4.3). This line already includes the GDN chunk, norms,
  writes and lm_head, so GDN is not added again. ×262144 = 1143 s = **19.1 min**.
- **Sparse attention: ~2.0 ms/tok → ~8.7 min.** Derived from the measured dense
  per-key rate, not the decode figure. Dense causal prefill at 64k costs
  1842 s over 16 layers × 65536²/2 query·key pairs = 5.36e-8 s per
  layer·query·key (a wall rate of 8.58e-7 s/token², matching 441 s @32k and
  1842 s @64k). A sparse query touches 2048 selected keys (128 pages×16) plus an
  own causal span averaging ~256 keys within a 512 chunk:
  16 × 2304 × 5.36e-8 ≈ 1.98 ms/tok → ×262144 ≈ 519 s = **8.7 min**.
  The 64k *decode* measure (0.37–0.43 ms sparse vs 3.46 ms dense over the 16
  full-attn layers) is a single query whose K load is not amortized and is
  launch/K-load bound (3.7x slower per gathered key); a 512-query prefill chunk
  reuses each selected key across its query tile and is MAC-bound, so it lands
  near the dense per-key rate. ~1.0 s of attention per 512 chunk, ~512 chunks.
  This term would be ~8 h extrapolated dense at 256k — the whole point.

**Predicted prefill ≈ 28 min** (19.1 non-attention + 8.7 attention; hit band
25–32 min).

**Predicted decode ≈ 8.5 tok/s.** #524 keeps the M≤8 decode ladder, so the
weight stream is unchanged. Dense 32k decode is 120.8 ms/tok; sparse attention
swaps 3.46 ms → 0.40 ms (−3.1 ms), giving ~117.7 ms/tok = 8.50 tok/s. Under
sparsity this stays flat as context grows (the dense number degrades), which is
the decode claim to check at 32k/128k/256k.

**Cold capacity: fits.** f16 KV at 256k is 65536 B/tok ×262144 = 17.18 GiB
total; the 128 hot pages (~0.13 GiB) stay device, so ~16 GiB demotes to host
against 26.6 GiB host free (~10.6 GiB slack). Depends on the #513 f16 cold
path; without it V100 cannot take f16 cold and this row does not run.

Miss conditions worth naming up front: prefill far over 28 min means the
selected-K load is not amortized across the prefill query tile (attention is
per-call-launch bound, not MAC bound); decode far under 8.5 tok/s means the cold
page fetch stalls the decode tick; an OOM host kill means the cold budget
under-counts the non-hot pages.

### Measured vs the pre-registration (overnight V100)

| run (head) | prefill s / ms·tok | decode tok/s / ms/tok | promotions per dec tick |
|---|---|---|---|
| **ladder 256k** (#533 f0a45485, pre-#524) | 4806.9 / 18.34 (80.1 min, 512 ticks) | 0.957 / 1045 | 274.4 (+274.5 demote) |
| **hot-pin 32k** (#534 dc19c3d1) | 208.8 / 6.37 | **5.653 / 176.9** | **0.0** (0.1 demote) |
| **M-tile 256k** (main fadb2726, +#524) | 2841.8 / 10.84 (47.4 min, 512 ticks) | 0.968 / 1034 (demote-all, no pin) | 263.9 (+264 demote) |

The 28-min prefill prediction is tested by the M-tile row only; the ladder row
is its pre-#524 control (18.34 ms/tok carries the slow M=32 GEMM ladder), so
80 min there is expected, not a miss. **The M-tile result is 47.4 min /
10.84 ms/tok — the 28-min point prediction missed high (1.7×), and even the
25–32 min band missed.** Decomposing: 10.84 ms/tok is still above the
predicted 4.36 ms/tok flat non-attention term, so ~6.5 ms/tok is attention —
higher than the 2.0 ms/tok estimated from the dense per-key rate, because at
256k the selector scores and promotes from a cold tier (the first pass pays
the D2H demote + H2D promote for pages not yet resident), not the steady
amortized rate the estimate assumed. The M-tile still gives 1.69× over the
ladder (80.1→47.4 min), so #524's linear win is real; the sparse attention
term at full cold context was simply under-priced. The decode 8.5 tok/s
prediction was made
for the **hot-pin** engine and 32k is its first check: 5.653 tok/s. Hot-pin
eliminates the fetch exactly (0 promotions, 0.1 demotions per steady tick) and
recovers 4.5x over demote-all (1.252 tok/s) but is still 0.68x the dense-eager
8.28 tok/s — the residual is sm70 per-tick Quest scoring over all candidate
pages (compute, not memory), distinct from the sm90 graph gap. The ladder 256k
decode shows why pin matters: 274 promote + 274 demote per tick, each a
synchronous ~1 MiB PCIe copy → 1045 ms/tok.

## Rule

Pre-register the long-context sparse prediction as a sum of a measured flat
linear term and an attention term capped at the *selected* key count, then run
it: sparse KV is only interesting if that attention term actually stops scaling
with history, and a number written before the run is the one that can be missed.

Operational: a chained remote run must pin the git ref it syncs — a chain whose
`v100.sh` syncs "the worktree" shipped the docs branch (no scorer/group fixes)
after the worktree switched mid-session, and the 256k run refused at submit
(pool=339) instead of running. Checkout the pinned sha and verify the fix
markers are present before the sync, or the chain runs the wrong tree silently.

