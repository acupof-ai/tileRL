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
- `/health` reported **`"decode_graph": true`**, `blocks_total` = 4425 (the graph
  build's pool sizing, identical to A1);
- warmup still ran the `4x2k` decode-graph capture in **5.3 s** (the real graph-on
  A1 capture was 311.9 s; both prove capture happened).

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
   runtime state (`/health decode_graph: false`), the changed pool build
   (`blocks_total` differs from the graph build), and the absence of graph-capture
   warmup lines. A banner is only a log index.

## Verification

`tests/test_serve_h20_sh.py` dry-run gates assert the off argv contains
`--no-decode-graph` and not `--decode-graph`, and the on argv the reverse;
`--dry-run` on the host confirms both. Device acceptance for the real A2 is
perf2's three runtime checks (health false, blocks differ from 4425, no `4x2k`
capture), plus a zero-traffic `/health` counter freeze proving
`LIVENESS_POLL_S=999999` stopped the guard's idle chat injection.
