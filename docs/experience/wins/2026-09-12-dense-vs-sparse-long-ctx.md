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
| H20 | dense | 131072 | **88.6** | 0.676 | **58.28** | **8.02** | 0 |
| H20 | sparse k=128 | 131072 | **108.0** | 0.824 | **19.16** | 0.55 | 0 at finish |
| V100 | dense (f32) | 32768 | **594.6** | 18.15 | **8.28** | **4.04** | 0 |
| V100 | sparse k=128 | 32768 | **341.3** | 10.42 | **1.252** | 1.16 | 0 at finish |
| V100 | sparse k=128 | 131072 | _pending_ (no dense pair — dense cannot fit) | | _pending_ | | 20.0 |

Dense controls: H20 128k prefill 88.621 s / decode 58.279 tok/s (17.2 ms/tok) /
KV 8612478976 B; V100 32k prefill 594.578 s / decode 8.278 tok/s
(120.8 ms/tok) / KV 4339007488 B.

**V100 32k sparse — prefill wins, decode loses** (head 95f69fe7, eager,
64 prefill ticks, 64 decode tokens skip 16): prefill 341.335 s =
10.417 ms/tok (**1.74x faster than dense**); decode 1.252 tok/s =
**798.4 ms/tok, 6.6x SLOWER than dense**; device KV 1.16 GiB (1107-block hot
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

**H20 128k sparse loses both phases** (head f0a45485, card 6, 256 prefill ticks,
KV_DEVICE 0.55 GiB = the 1107-block hot pool in bf16): prefill 107.988 s =
0.824 ms/tok (**1.22x slower** than dense 0.676), decode 19.160 tok/s =
52.2 ms/tok (**3.0x slower** than dense 17.2). Dense attention on the H20 runs
bf16 WGMMA and is not the bottleneck it is on the V100, so selection scoring
plus the demote-all page fetches cost more than the attention they remove —
sparse does not pay for itself on a fast interconnect when dense already fits.
The decode penalty is smaller in relative terms than the V100 (3.0x vs 6.6x),
consistent with H20's faster host path. The sparse value at 128k on an H20 is
capacity for contexts past what 8.6 GiB of bf16 KV allows, not latency; at
128k it fits, so dense is the right choice. This is the same demote-all merge
state, so the hot-pin PR should recover most of the decode loss but the prefill
net is likely still negative on sm90 where dense attention is cheap.

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

## Rule

Pre-register the long-context sparse prediction as a sum of a measured flat
linear term and an attention term capped at the *selected* key count, then run
it: sparse KV is only interesting if that attention term actually stops scaling
with history, and a number written before the run is the one that can be missed.
