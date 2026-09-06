# A guard that stopped guarding: #214 broke pod_run's direct-python claim — 2026-09-07

> Status: fixed. `pod_run_claim` polls both claim shapes; the selftest covers both.

## Context

#214 (`2f01965`) made `pod_run.sh` claim its job's card and die if it cannot, because a launcher
that continued past a refused claim left card 6 holding 69583 MiB with no claim for 5 minutes.
It claimed through `card_claim.py acquire --pid $JOB --wait-for-device`. That PR shipped the two
scripts and no entry, which is part of why the gap below went unexamined.

Today two 27B profile runs were killed by that guard, 90 s and then 300 s into their model
load, with the message `no descendant of pid N opened a GPU device`.

## Root cause

**`--wait-for-device` strictly requires a DESCENDANT.** `pod_run.sh` runs `setsid $CMD &` and
hands `$JOB` to the claim; for `-- python3 scripts/...` that pid **is** the python that opens the
card, so the poll waits for a descendant that never appears, reports a timeout, and `pod_run`
kills the pid it could not claim.

Measured against the real `/work/aupai/scripts/card_claim.py`, not inferred:

| pid state | `--require-device` | `--wait-for-device N` |
|---|---|---|
| python holding no device | `holds no GPU device fd` | `no descendant ... in Ns` |
| python holding a device, no children | **`claimed`, 2 s after the fd opens** | **times out at N** (checked to N=30) |

The flag was wrong, not the timeout. I raised it 90 → 300 first and that killed the second run.

**It is `--wait-for-device` specifically, not `pod_run.sh` since forever.** 25 reported three
direct-python launches (`kgA`, `kgA2`, `kgB`) that printed `claimed 6 for tilerl-kgB` and the
like. Those ran under the **pre-#214** claim code, which passed no device flag at all and
retried on `no device fd yet` — a bare `acquire --pid` claims a self-holding python without
complaint. #214 replaced that with `--wait-for-device`, which is correct only for a wrapper and
broke exactly the direct-python shape that had been working. So the regression window is #214
(`2f01965`) to this fix, not the whole life of the script.

A `setsid` fork was the obvious candidate for why it worked for 25 and not for me, and it is
wrong: measured on this pod, `setsid cmd &` leaves `$!` as the command itself with no children
**whether or not** the calling shell is a process-group leader. Nothing about the launching
environment changes which pid `$JOB` names here.

**Does a claim survive the `setsid` parent exiting?** Yes — `card_claim` keys a claim's liveness
to the claimed pid itself (`_alive` → `os.kill(pid, 0)`, `:274-280`), never to its parent, so a
claim printed at launch holds until that python dies.

**Two things hid it.**

1. Every post-#214 run that looked claimed was claimed **by hand**, including that morning's
   `bwdops` run — I read `ORPHAN`, verified the pid by fd, and claimed it myself, then read the
   resulting good state as evidence the launcher worked.
2. #214's selftest launches `-- bash wrapper.sh`. A wrapper *does* have a python descendant, so
   the arm passed on the one shape where the descendant flag is correct, and the direct-python
   shape was never exercised.

## The second defect in the same guard

Arm 2 — the control asserting an unclaimable job exits 4 — was passing on **rc 0**. Its job
slept 1 s while the poll ran up to 300 s, so the job exited first and the code took the benign
`job exited before it could claim` path. The kill it exists to test never ran. It now uses an
8 s job under a 2 s poll and asserts the `killing` line.

Both `DEVICE_WAIT` and `JOB_SECS` are threaded into the runner's **execution**, not just the
emit — the same mistake this file already carried a comment about, from the run where
`CLAIM_MODE` reached only the emit and arm 2 passed against the default mode.

## Fix

`pod_run_claim` polls both shapes once a second, up to `DEVICE_WAIT` (300):

- `--wait-for-device 1` — a wrapper whose python is a descendant
- `--require-device` — a pid that *is* the python

Neither alone covers both. It also prints `claim pending for <pid>, polling up to Ns` before
blocking, so a startup window reads differently from a hang.

Selftest arms: wrapper claims (1), multi-arm wrapper re-claims per arm (3), **direct-python
claims as itself (4)**, unclaimable job is killed and released (2).

## Residual hazard

`card_claim.py status` prints `ORPHAN card N holds M MiB with no claim` for the whole startup
window of a legitimate job — up to 300 s now — and that reading is indistinguishable from a
real orphan unless you check for a live `card_claim` process. Anyone acting on a bare ORPHAN
line during a startup window will kill a healthy job; I nearly did. The fix belongs in
`card_claim.py status` (print `PENDING` when a wait for **that card** is alive), which lives in
aupai, not here.

## Rule

A guard's test must exercise the shape the guard runs against in production. #214's arm proved
the wrapper path; the direct-python path — the one every profile run takes — was never executed,
so a change that fixed the wrapper broke direct-python and looked green for a day.

When a guard times out, read the flag's semantics before raising its deadline. A timeout says
"the thing I poll for did not appear"; it does not say the thing is slow. Raising 90 to 300
cost a second 27B run and produced the identical message.

A good state you produced by hand is not evidence the automation works. The claim I placed
myself made the launcher look correct in exactly the runs that would have exposed it.

Scope a regression to the change that caused it before writing "never". I wrote that pod_run had
never claimed a direct-python job; 25's three successful launches were the counterexample, and
reading the pre-#214 code showed a bare `acquire --pid` had always worked for that shape. The
window is #214 to this fix. A peer's contradicting evidence was what forced the check — and the
first mechanism we both reached for (a setsid fork) was also wrong, measured.
