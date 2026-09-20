# A fixed sleep is not a readiness check — 1-in-5 flake under xdist, 2026-09-20

**Date:** 2026-09-20
**Sessions:** worktrees-53 (inspector), fixmisc-72, tilerl-b1
**Task:** a macOS launcher-lock test reported as flaky 1-in-5 on CI

## Context

`test_serve_v100_sh.py::test_a_second_supervisor_is_refused_and_the_lock_is_why`
asserts that a second `serve_v100.sh` is refused because a holder already has the
lock. It spawned the holder, slept a fixed `0.5 s`, then ran the second copy and
asserted `returncode == 1 and "already running" in r.stderr`.

On a serial single-file run it passed 5/5. The report of a 1-in-5 flake was
initially dismissed on that basis — **the shape was wrong**: CI runs
`pytest -v -n auto`, and the flake only exists in that shape. Reproduced under the
CI invocation: **4 failures in 30 runs** (runs 18, 20, 25, 27).

## Root cause

The failure output, which had not been captured before:

```
assert r.returncode == 1 and "already running" in r.stderr, r.stderr[:200]
AssertionError: date: invalid argument 's' for -I
assert (0 == 1)
```

`0 == 1` is the whole answer: **the second supervisor was not refused, it ran**.
The `date: invalid argument 's' for -I` string is the launcher's own BSD-`date`
noise printed by a process that should never have started — a symptom that reads
like a `date` bug and is not one.

Under xdist the holder shell (`bash -c 'exec 9>…; flock -n 9; sleep 10'`) has not
yet reached `flock` when the second copy starts 0.5 s later. The lock is
legitimately free at that moment, so the second copy passes `flock -n 9` and
runs. Nothing is broken; the test asked its question too early.

Discriminating experiment — one line changed, `sleep(0.5)` → `sleep(3.0)`:

| holder warmup | failures |
|---|---|
| 0.5 s (fixed sleep) | **4 / 30** |
| 3.0 s (fixed sleep) | 0 / 15 |

That separates the three candidates. It is not the `flock` shim's lifetime (the
two-process control returns REFUSED under **both** warmups), and not a shared
lock path between parallel tests (both files use unique `mkdtemp` prefixes, so no
test can collide with another's lock file). It is the fixed interval.

## Fix

Poll for an observable readiness signal instead of sleeping. The holder writes a
marker only after `flock` returns, and the test waits for that marker with a
deadline, asserting the holder neither exited nor timed out:

```
exec 9>LOCK; flock -n 9 || exit 9; : > holder.ready; sleep 10
```

then `while not ready.exists(): assert holder.poll() is None; assert within deadline; sleep 0.05`.

The marker is the point: "the holder is holding the lock" becomes an observed
fact rather than an elapsed-time assumption, and a holder that fails to take the
lock now fails loudly instead of silently freeing it.

Verified: **30/30 green under `-n auto`**, the shape that failed 4/30.

## Rule

A cross-process readiness state must be **observed**, never waited out with a
fixed sleep. `sleep N` encodes a guess about how fast another process reaches a
state; under parallel test execution that guess is a coin flip, and it fails as a
wrong-result assertion (here `rc == 0`) rather than as a timeout — so the failure
names whatever the wrongly-started process printed, not the race. When a test
starts a helper and then depends on that helper's state, have the helper publish
the state and poll for it.

Corollary for reading flakes: a green serial run does **not** rebut a parallel-run
flake. Reproduce in the shape CI uses before dismissing the report.
