# expandable_segments on the V100 hybrid launcher — pending-remote, 2026-09-20

> Status: pending-remote. The env reaches the child (CPU-verified); the served
> effect on this card is not measured and is what the next V100 window is for.

## Context

`scripts/serve_hybrid_v100.sh` — the live V100 sm70 hybrid supervisor — served
without `PYTORCH_CUDA_ALLOC_CONF`. On this card and torch build the flag is
**load-bearing**: the same build OOMs without it and runs with it, and it took
free memory **396 MiB → 1.11 GiB**
([errors/2026-09-03-expandable-segments-is-load-bearing](../errors/2026-09-03-expandable-segments-is-load-bearing.md)).

That entry left shipping it explicitly **undecided** — "`expandable_segments`
changes allocator behaviour globally and its cost on this workload is unmeasured
here ... One env default is a small diff and a large blast radius; it needs its
own A/B." This entry is that A/B, scheduled rather than assumed.

## What worked

`export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}`
in the launcher's env block, so the served child inherits it. An
operator-supplied value wins, so a run that opts out can.

**Scope is the sm70 launcher only.** It is not a cross-backend default:
`serve_h20.sh` (sm90), `serve_v100_dense.sh` and `serve_v100.sh` are untouched,
and a gate asserts that. The flag's blast radius is why.

**Coexistence with `--decode-graph` is prior evidence, not inference.** The
2026-09-03 B=1 baseline ran this exact flag with **8 precaptured dense decode
graphs** on this card and stack
([wins/2026-09-03-single-stream-b1-baseline](2026-09-03-single-stream-b1-baseline.md)),
which is the configuration the launcher serves.

## Rule

A published number that needed an env var the tree does not set is a scoped
number. Shipping the variable is a separate decision from publishing the number,
and it is the decision that needs the device A/B.

## Gates (CPU)

`tests/test_serve_hybrid_fuse.py`, driven with a stub python that dumps its
environment:

- the child process really receives `expandable_segments:True`;
- an operator's `PYTORCH_CUDA_ALLOC_CONF` survives verbatim;
- `serve_h20.sh`, `serve_v100_dense.sh`, `serve_v100.sh` do not set it.

With the export deleted the first goes red.

Not covered, deliberately: the flag's **throughput** cost, which is what the
2026-09-03 entry named as unmeasured and what this window is for.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| | | V100 sm70 | 27B-nvfp4 | | | | |

Pending the next V100 window: one full warm serve, confirming no OOM, steady
decode held against the current baseline, and that a real OOM still raises
`FatalDeviceError` — the exit path is unchanged, this flag catches nothing.
