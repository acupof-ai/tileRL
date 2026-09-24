# serve ignores SIGINT and SIGTERM, during load and when healthy — V100, 2026-09-24

> Status: open

## Context

During the #805 production cutover window the service was started by a script
whose correctness gate had already gone red (the driver predated its own gate
fix). It came up on the wrong configuration and had to be stopped. Stopping it
took three signals, and two of them did nothing. The restart path assumes a
graceful stop works, so this matters beyond the one incident.

Measurand is the process table: `ps -o stat= -p PID` after each signal, plus
`/proc/PID` existence, on the exact pid (never `pgrep | head -1`).

## What happened

pid 691782, launcher 691779, started by `run_serve_prod.new.sh` on the $M tree
(`82e3dfe3`), `--sparse-min-tokens` absent (min0), 4 slots, cold tier enabled.

| signal | when | observed |
|---|---|---|
| `SIGINT` | ~1 min in, state `RNl` (still loading) | ignored for 60 s; process kept loading |
| `SIGINT` (implicit) | after load finished | n/a — it reached healthy with `running:1` |
| `SIGTERM` | immediately after `/health` returned `{"status":"ok"}`, state `SNl` | ignored for 45 s; still present |
| `SIGKILL` | after the above | exited within 1 s, GPU memory released |

The service log's last line records the escalation: `run_serve_prod.new.sh: line
17: 691782 Killed …`.

## Why, as far as the code shows

`src/tilerl/cli.py` `cmd_serve` calls `uvicorn.run(app, …)` at its **last** line
(:257), inside a `try/finally` whose `finally` calls `engine.shutdown()`. Model
build, cold-tier setup and engine construction all happen before that call. So
during load no signal handler for INT/TERM is installed by this program at all —
the default disposition for SIGINT in a foreground process is to terminate, but
this process was started with `nohup … &`, and nothing in `src/` installs a
handler (`git grep -n "signal\." 82e3dfe3 -- src/` returns only two comment
hits; there is no `KeyboardInterrupt` anywhere in `src/`).

The healthy-phase ignore is **not explained by that** and is the part that
needs reading: by then `uvicorn.run` is on the stack and uvicorn normally
installs SIGINT/SIGTERM handlers. Either the handler is installed on a loop that
is not the one draining, or the signal is delivered while the engine holds the
GIL inside a long tick. The one measurement that bears on it: the serve log's
last ticks before the kill are `dec=0 pre=1` with `total` 6228/5188/3910/2626 ms
and `model` 4578/3534/2895/1660 ms — i.e. the process was inside multi-second
prefill ticks, which is where a Python-level handler would be deferred. That is
consistent with GIL-starvation rather than a missing handler, but it is a
plausible reading, not a measured one.

## What worked

Nothing yet. Workaround for now: `kill -KILL` by exact pid, then wait on
`nvidia-smi --query-compute-apps` going empty to confirm the memory came back.

## Notes for whoever picks this up

- The two phases are different bugs at most, and the same bug at least — do not
  write them up as one finding without checking which.
- A restart/rollback path must not assume INT or TERM stops the serve. Any SOP
  that says "stop the service and wait" needs a bounded wait and an explicit
  escalation to KILL, or it hangs.
- `nohup` masks the foreground-SIGINT default, so "it should have died on
  Ctrl-C" is not available as an explanation here.
- The fastest discriminator for the healthy-phase case: send SIGTERM while the
  service is **idle** (no request in flight). If it dies while idle but not while
  prefilling, it is deferral-during-tick; if it survives idle too, the handler is
  genuinely absent.
