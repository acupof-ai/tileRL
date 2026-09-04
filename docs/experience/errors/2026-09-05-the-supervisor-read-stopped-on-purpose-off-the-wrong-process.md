# The supervisor read "stopped on purpose" off the wrong process — 2026-09-05

**Date:** 2026-09-05
**Task:** restart the V100 server to pick up new SSE code, keep it available for ckl
**Symptom:** `kill -TERM <server pid>` left nothing listening on 8000 and the supervisor
gone too — the one failure the supervisor exists to prevent.

## Context

`scripts/serve_v100.sh` restarts the server when it dies, up to 10 times. Its stop
condition was the child's exit code:

```bash
case $rc in 0 | 130 | 143) exit "$rc" ;;   # stopped on purpose
```

143 is `128 + SIGTERM`. That reads correctly for the case it was written for — TERM to
the *supervisor*, whose trap forwards to the child and then exits 143 itself — and
`tests/test_serve_v100_sh.py::test_a_deliberate_stop_is_not_restarted` asserted exactly
that behaviour, so the gate agreed with the defect.

## Root cause

The child exits 143 whenever *anything* TERMs it, and the supervisor cannot tell the
sources apart from the exit code alone:

| what happened | child rc | should restart? | old behaviour |
|---|---|---|---|
| TERM to the supervisor | 143 | no | exit ✓ |
| TERM to the server only | 143 | **yes** | exit ✗ |
| OOM killer takes the server | 143 | **yes** | exit ✗ |
| server crashes | 7 | yes | restart ✓ |

Two of the four rows are the reason a supervisor exists, and both were read as deliberate.
On the pod that meant `--depth 1` at 22 GiB vanished with no listener and no log line
saying anything was wrong: `ss -ltnp` showed nothing on 8000 and `/proc/1987997` was gone.

## Fix

Intent is knowable at the source. The trap sets a flag before it forwards the signal, and
the loop reads that instead of guessing from the child:

```bash
stopping=
trap 'stopping=1; ... exit 143' TERM INT
...
if [ -n "$stopping" ] || [ "$rc" = 0 ]; then exit "$rc"; fi
```

`test_a_deliberate_stop_is_not_restarted` encoded the defect, so it was replaced by
`test_a_clean_exit_is_not_restarted` (rc 0 still stops) plus
`test_killing_only_the_server_restarts_it`, which signals the stub server's own pid and
asserts a second boot **and** that the supervisor is still alive. The stub now writes
`$$` so a test can address the server rather than the supervisor.

Negative control on the pod, the old `case` line restored:

```
FAILED test_killing_only_the_server_restarts_it
AssertionError: the server was killed and never came back (1 boot(s)):
nothing is listening and the supervisor has walked away
1 failed, 5 passed
```

Fixed script: 6 passed.

## Rule

**A signal's exit code says what happened, not who asked for it.** Anything that needs to
distinguish "I did this" from "this happened to me" has to record intent where the intent
exists — in the handler — because `128 + n` is identical either way.

**A test can be the defect's own alibi.** `test_a_deliberate_stop_is_not_restarted` passed
throughout and was the reason the behaviour looked settled. When a test asserts the thing
you are about to call a bug, the test is part of the change, and deleting it is the fix,
not a loss of coverage. Same shape as
[a-golden-test-proves-only-what-it-exercises]: the assertion covered the case its author
had in mind and named itself as though it covered the category.

**A change to a wire format breaks the tools that read it, including your own.** Adding
cumulative usage to content frames broke `probe_page_rate.py`, whose
`if obj.get("usage"): continue` then skipped every content frame and asserted "no content
frames arrived" against a server that had just sent 109 — the same defect the change was
fixing in the page. Grep for every reader of a field before changing what carries it.
