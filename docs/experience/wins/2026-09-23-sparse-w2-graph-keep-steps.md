# Captured sparse verify keeps every step — V100 sm70, 2026-09-23

> Status: landed behind the #807 guard (`_sparse_capture_allowed`, unchanged by
> this commit), which keeps the CUDA sparse capture **off** at
> `spec_depth>=1`. So the merged tree's device behavior does not change here —
> the fix removes the divergence the guard was fencing, and lifting the guard
> is a separate call on this evidence.

## Context

On CUDA the sparse captured decode graph arms a width-2 verify, and its drafts
stopped accepting after the first one or two ticks. The wall-clock symptom is a
decode rate **7–13 tok/s at W=2 against 11.6–14.9 tok/s at W=1** — the
speculative arm was slower than not speculating. Underneath it is a correctness
fault: the graph verdict and the eager verdict disagree on the same tick.

Both arms build a `BatchKv` per tick, but with different `keep_steps`:
`SparseDecodeGraph` passed `keep_steps=int(W > 1)` (i.e. 1 at W=2), while the
eager verify passes `keep_steps=width` (2). `keep_steps=1` fails the
`gdn_decode_fused` eligibility gate (`t > 1 and keep_steps != t`), so the
captured arm fell back to the GDN chunk kernel, which casts q/k/v/z through bf16
on the way in. The eager arm kept the fused f32 path. One layer of lost
precision at the first GDN layer propagates to every full-attention layer.

## What Worked

`decode_graph.py` now builds both the CUDA and CPU sparse graphs with
`keep_steps=W if W > 1 else 0`, matching the eager verify's `keep_steps=width`.
W=1 keeps 0, since no step pool exists without a draft.

Measured with the `--parity` per-tick token probe on V100 sm70, two trees
differing **only** in those two lines (`git diff --stat` asserted inside the
window: `decode_graph.py | 4 ++--`), one probe revision, four arms — patch and
control, each run in both orders so a position effect would show as an
order-to-order split.

- **Parity**: patch 6/6 cells MATCH; control 3/3 W=2 cells
  `ALIGNMENT_UNMATCHED` (W=1 cells match on both).
- **Accept rate**: patch `tok/fwd = strict accept = 1.958`, identical to eager,
  on all three buckets. Control 1.382 / 1.880 / 1.679 — 1.382 is the old
  1 + 0.382 shape.
- **Order control**: graph agrees to within 12% between orders on every cell
  (headline b1024 W=2: 154.3 / 153.6 ms/tick), and the control reproduces its
  rate in both orders. No position effect.

## Rule

**A captured sparse verify tick must build `BatchKv` with the same
`keep_steps` the eager verify uses.** The CPU token gate cannot see this — the
CPU cell registers no `gdn_decode_fused`, so both arms fall back to the same
reference kernel and stay token-exact — which is why a structural gate, not the
CPU oracle, is what catches it.

This is a **correctness** fix, not a speed-up: at W=2 the graph runs **7–13
tok/s**, below its own W=1 **11.6–14.9 tok/s**. Restoring parity removed a
divergence; the speculative arm still costs throughput here.

The parity window ran the two trees directly, so it measured the graph with the
guard bypassed. On the merged tree `_sparse_capture_allowed` still returns
`False` for CUDA at `spec_depth>=1`, so this fix is not yet in any production
path — it makes the guarded configuration correct, it does not re-enable it.

## Results

Same-day A/B on one V100 (sm70), bucket 512 / 1024 / 2048 × W=1 / W=2, patch
`ad2d0495` vs control `95408efd` (patch with the fix reverted), probe `1a55ef94`.
ms/tick and tok/s are decode-loop means; `tok/fwd` is tokens per forward and
`accept` the strict per-width rate. Each arm was run twice, in both orders.

| arm | order | bucket | W | verdict | tok/fwd | accept | graph ms/tick | graph tok/s | eager ms/tick | eager tok/s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| patch | A | 512 | 1 | MATCH | 1.000 | – | 86.4 | 11.6 | 168.8 | 5.9 |
| patch | A | 1024 | 1 | MATCH | 1.000 | – | 67.3 | 14.9 | 159.8 | 6.3 |
| patch | A | 2048 | 1 | MATCH | 1.000 | – | 86.4 | 11.6 | 190.7 | 5.2 |
| patch | A | 512 | 2 | MATCH | 1.958 | 1.958 | 195.6 | 10.0 | 243.8 | 8.0 |
| patch | A | 1024 | 2 | MATCH | 1.958 | 1.958 | 154.3 | 12.7 | 237.7 | 8.2 |
| patch | A | 2048 | 2 | MATCH | 1.958 | 1.958 | 268.1 | 7.3 | 354.8 | 5.5 |
| patch | B | 512 | 1 | MATCH | 1.000 | – | 91.1 | 11.0 | 167.4 | 6.0 |
| patch | B | 1024 | 1 | MATCH | 1.000 | – | 66.9 | 14.9 | 162.9 | 6.1 |
| patch | B | 2048 | 1 | MATCH | 1.000 | – | 85.3 | 11.7 | 196.2 | 5.1 |
| patch | B | 512 | 2 | MATCH | 1.958 | 1.958 | 174.8 | 11.2 | 237.1 | 8.3 |
| patch | B | 1024 | 2 | MATCH | 1.958 | 1.958 | 153.6 | 12.8 | 974.7 ☨ | 2.0 |
| patch | B | 2048 | 2 | MATCH | 1.958 | 1.958 | 282.5 | 6.9 | 344.3 | 5.7 |
| control | A | 512 | 1 | MATCH | 1.000 | – | 97.4 | 10.3 | 261.0 | 3.8 |
| control | A | 1024 | 1 | MATCH | 1.000 | – | 67.2 | 14.9 | 168.0 | 6.0 |
| control | A | 2048 | 1 | MATCH | 1.000 | – | 86.7 | 11.5 | 186.1 | 5.4 |
| control | A | 512 | 2 | ALIGNMENT_UNMATCHED | 1.382 | 1.382 | 146.4 | 9.4 | 231.0 | 8.5 |
| control | A | 1024 | 2 | ALIGNMENT_UNMATCHED | 1.880 | 1.880 | 266.2 | 7.1 | 878.7 ☨ | 2.2 |
| control | A | 2048 | 2 | ALIGNMENT_UNMATCHED | 1.679 | 1.679 | 358.4 | 4.7 | 522.1 | 3.8 |
| control | B | 512 | 1 | MATCH | 1.000 | – | 86.8 | 11.5 | 169.1 | 5.9 |
| control | B | 1024 | 1 | MATCH | 1.000 | – | 250.8 ☨ | 4.0 | 165.6 | 6.0 |
| control | B | 2048 | 1 | MATCH | 1.000 | – | 85.1 | 11.8 | 188.2 | 5.3 |
| control | B | 512 | 2 | ALIGNMENT_UNMATCHED | 1.382 | 1.382 | 149.8 | 9.2 | 233.3 | 8.4 |
| control | B | 1024 | 2 | ALIGNMENT_UNMATCHED | 1.880 | 1.880 | 154.6 | 12.2 | 238.5 | 8.2 |
| control | B | 2048 | 2 | ALIGNMENT_UNMATCHED | 1.679 | 1.679 | 240.3 | 7.0 | 463.9 | 4.2 |

Control `tok/fwd` is the same measurement as `accept` on a W=2 cell (every tick
has width>1), so the two columns carry identical values there; at W=1 the strict
rate is undefined by construction and is marked `–`, not zero.

☨ **Two cells, at the b1024 / ORDER B combination — a reproducible pairing, not
machine jitter (corrected 2026-09-23, see below).** control-B b1024 W=1 graph
**250.8** ms/tick where the same arm's other W=1 cells read 86.8 / 85.1 and
ORDER A's same cell read 67.2; and patch-B b1024 W=2 eager **974.7** ms/tick
against ORDER A's 237.7. Neither moves a verdict, but b1024 should not be quoted
as a stable latency.

**Correction.** This paragraph originally read "recorded as machine jitter, not
position effect." That is wrong. The phase window of the same day
(`2026-09-23-sparse-w2-phase-timing.md`) hit the same pairing again: two cells,
both b1024, both ORDER B, in **two different arms**, with `step − graph` of
**41.909** and **41.911 ms** — agreeing to three decimals across arms, which is a
deterministic signature rather than jitter. That makes three windows at which the
b1024/ORDER-B pairing misbehaves, across arms and across measurands (this window
saw it inside the graph, the phase window outside it). Mechanism undetermined;
tracked as an open defect in
[errors/2026-09-23-b1024-order-b-outside-graph.md](../errors/2026-09-23-b1024-order-b-outside-graph.md)
and in [OPEN.md](../OPEN.md).

Raw artifacts: `sparse-w2-graph-keep-steps-2026-09-23/parity-patch-A-20260923-171959.json`,
`…/parity-patch-B-20260923-171959.json`,
`…/parity-ctrl-A-20260923-171959.json`,
`…/parity-ctrl-B-20260923-171959.json` (probe revision `1a55ef94`).
