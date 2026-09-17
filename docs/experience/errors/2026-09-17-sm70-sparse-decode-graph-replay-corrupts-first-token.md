# sm70 sparse decode graph first replay corrupts the first token — 2026-09-17

> Status: open, measured verdict (H2_BAD). The cmax-bucket contamination
> hypothesis (H2) is now tested on device and CONFIRMED: a sparse decode graph
> lazily captured at a cmax bucket returns a wrong first token on its first
> replay, at every sparse bucket and both decode widths. The sm70 sparse decode
> graph stays eager/disabled — the current hybrid gate behavior; no `src`
> change is required, only a reclassification of "lever B untested" to
> "tested BAD". Supersedes the H2-untested line of
> [2026-09-14-sm70-sparse-graph-mtp-corrupts-output.md](2026-09-14-sm70-sparse-graph-mtp-corrupts-output.md).

## Context

V100 sm70, Qwen3.8-27B-NVFP4, sparse k=128, cold tier mirroring the hybrid
serve (1 GiB pinned / 8 GiB SSD, f16), decode graph on, MTP depth 1. Probe
`scripts/probe_sparse_graph_cmax_bucket.py` (#700), main 220b3c95, file md5
`eda0a9a1f71b797ec689e139fb0e4865`. One engine per subprocess (fresh CUDA
context; two concurrent 27B engines OOM sm70), graph arm and eager arm built
identically with `sparse_min_tokens=0` and `sparse_device_select=True` so the
two differ ONLY in `decode_graph`. Each row is primed so its FIRST decode
tick lands exactly on the target cmax bucket; the first decode-produced token
(temperature 0, deterministic) is compared graph vs eager.

| cmax bucket | W | graph token | eager token | captured | observed cmax |
|---|---:|---:|---:|---|---:|
| 512  | 1 | **8317** | 317 | true | 512 |
| 1024 | 1 | **16509** | 44  | true | 1024 |
| 2048 | 1 | **1894** | 1895 | true | 2048 |
| 512  | 2 | **8317** | 317 | true | 512 |
| 1024 | 2 | **16509** | 44  | true | 1024 |
| 2048 | 2 | **1894** | 1895 | true | 2048 |

6/6 bucket×W CONTAMINATED, each producing a token in a single decode step,
target-bucket assert passing (no silent next-bucket) and the graph present in
the capture dict. Peak reserved 29716 MiB. The graph/eager token equality is
exact-compare; there is no tolerance story here, the ids differ outright.

## Root cause

Mechanism unproven (hypothesis only — do not cite as established). Measured
pattern:

- At buckets 512 and 1024 the graph token equals prompt length + 6
  (8311 → 8317; 16503 → 16509). A vocab id that tracks the input position
  points at the replay reading a position/index-correlated field as logits,
  not at a near-tie in a real distribution.
- At bucket 2048 the two ids differ by one (1894 vs 1895), consistent with a
  near-tie argmax flipped by a small perturbation.
- W=1 and W=2 are identical to the token, so the draft head is not the
  corrupting component.

This is the first replay of a graph freshly captured at a new cmax bucket on
a live-traffic shape — the H2 mechanism. The 2026-09-14 H1 (warmup scribbling
live block/slot 0, fixed against the reserved pad frame in #585) is
orthogonal and stays disproved; pad-frame capture did not remove this.

## Fix

None. Operational conclusion: keep the sparse decode graph eager/disabled on
sm70 — exactly what the hybrid engine already does (`_sparse_graph_on` forced
false under hybrid since #586). A real fix needs the captured tick's first
replay to match eager on live shapes (capture staging / width selection),
which is open. The dense decode graph is unaffected and stays on; eager
sparse and dense+graph remain token-exact.

## B=4 illegal-access arm: not exercised

The `--only-b4` child exited rc=12, but this was the harness's missing-B=4
guard, NOT OOM, NOT a poisoned allocator, and NOT an illegal memory access.
Only `(B,W,cmax,own_w) = (1,2,1024,9)` and `(2,2,1024,9)` graphs were
captured: under `max_num_batched_tokens=512` the four 8311-token prefills
stagger into decode, so a B=4 wave never formed and the live bucket had
already drifted 512 → 1024 by B=2. (B=1 and B=2 captures succeeding is itself
evidence the context was not poisoned.) The original B=4 illegal-access
question therefore remains untested by this run, but is moot while the graph
stays disabled for H2_BAD. Re-testing it needs a b4 harness change — larger
batched-prefill tokens or holding rows at the decode boundary — and a re-review.

## Rule

A sparse decode graph must not be enabled on a device until a REAL-DEVICE gate
shows per-token graph=eager parity on the first replay at each cmax bucket and
both widths. A CPU tiny run cannot provide this: the dry seam matched
6/6 because it does not capture the device graph. The gate belongs in the
device down-window, not CI.
