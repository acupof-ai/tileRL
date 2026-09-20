# H20 graph-off arm silently ran graph-on: omitting --decode-graph is AUTO, not off — 2026-09-20

> Status: root cause found and fixed (hotfix to `scripts/serve_h20.sh`). A
> measurement-integrity incident caught before the graph-vs-eager contrast was
> published: the first A2 boot advertised `decode_graph=off` while the engine was
> still capturing graphs, so it was a repeat of A1 and would have produced a
> graph net-contribution of zero (a false "graph does not help on H20").

## Context

The H20 (sm90) sweep compares depth/sparse-k arms with decode graph on and off
(A1 d1/k128/graph-on, A2 d1/k128/graph-off, …). `scripts/serve_h20.sh` gained
`SERVE_DECODE_GRAPH` in #758 so the same supervisor could launch both. The
measurement contract is that a control arm must be the real eager build, and
perf2's permanent arm check is the runtime field `/health decode_graph`, not the
launcher banner.

## Symptom

First A2 boot with `SERVE_DECODE_GRAPH=0`:

- boot line printed `… decode_graph=off …` (the launcher echoed env INTENT);
- the resolved argv contained neither `--decode-graph` nor any off flag;
- `/health` reported **`"decode_graph": true`**;
- startup logged **`8 decode graphs in 3s`** — the capture action actually ran.

The last two are the evidence: a mode flag reading true and a non-zero graph
capture count say capture was live.

`blocks_total` is NOT a mode discriminant and must not be read as one. It reports
`engine.usable_blocks = kv.num_blocks − (pad_block present)`: the build allocates
the KV pool as `num_blocks + pad` (one replay row when graph-on) and the capacity
answer subtracts that same pad row, so the net figure is identical on/off (4425
on both arms here). The raw pool differs by one block but is not in /health.

The warmup `4x2k` phase name and its timing are NOT evidence either:
`serve_warmup_hybrid.py`'s `4x2k` is an unconditional four-request warmup, not a
capture step, and the timing spread across graph-on boots (311.9 s first, 67.4 s
later) is cold-vs-warm TileLang JIT cache, not capture on vs off. The capture
COUNT line (`N decode graphs in Ns`) is the authoritative startup signal — 8 vs 0.

So the "graph-off" arm was graph-on. Its ticks and acceptance counters would have
been identical to A1, manufacturing a zero graph delta.

## Root cause

A three-state CLI flag, with an off switch the supervisor never sent:

- `tilerl.cli serve`: `--decode-graph` is `argparse store_const const=True
  default=None`; the only force-off is the separate **`--no-decode-graph`
  (`const=False`)**.
- `engine._graph_on(None)`: a `None` arg means AUTO — on CUDA, AUTO returns
  **True on sm90** (sm70 is the one arch where AUTO disables capture).
- #758's `SERVE_DECODE_GRAPH=0` branch built `GRAPH_ARGS=()` — it merely
  **omitted** `--decode-graph`, leaving the arg at its default `None`, which on
  this H20 resolved to graph ON.

The intent string (`decode_graph=off`) was logged from the env var, decoupled
from the argv actually exec'd, so the self-certification line asserted a
configuration the process did not have.

## Fix and rule

`serve_h20.sh` graph-off branch now passes the explicit force-off:

```sh
GRAPH_ARGS=(--no-decode-graph)
[ "$DECODE_GRAPH" != 0 ] && GRAPH_ARGS=(--decode-graph)
```

Rules this pins:

1. For an on/off/AUTO `store_const default=None` flag, "the launcher did not pass
   it" must never be read as "off". A control arm sends the explicit force-off
   constant; omission selects AUTO.
2. A launcher's self-certification line is allowed to describe the ACTUAL argv,
   not the operator's intent — and even then, an arm's authoritative check is the
   runtime capture state: the startup **capture count (`N decode graphs in Ns`,
   expect 0 off / 8 on)** first, then `/health decode_graph`; the resolved argv
   carrying `--no-decode-graph` is supporting evidence. A banner is only a log
   index. `blocks_total` is deliberately NOT used: it is `usable_blocks`
   (pool `N+pad` minus the pad row), net-identical on/off. Startup/warmup wall
   time is never mode evidence (TileLang JIT cache temperature), and warmup phase
   names like `4x2k` are unconditional requests, not capture markers.

## Verification

`tests/test_serve_h20_sh.py` dry-run gates assert the off argv contains
`--no-decode-graph` and not `--decode-graph`, and the on argv the reverse;
`--dry-run` on the host confirms both. Device acceptance for the real A2 is the
capture evidence — startup `0 decode graphs in 0s` plus `/health
"decode_graph": false` (argv `--no-decode-graph` supporting) — together with a
zero-traffic `/health` counter freeze proving `LIVENESS_POLL_S=999999` stopped
the guard's idle chat injection. `blocks_total` is not consulted (net-identical
4425 on/off), and warmup phase names/wall time are not used.
