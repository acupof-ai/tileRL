# The boot line's sha said the new flags were live; the process had the old ones — V100 sm70, 2026-09-05

## Context

Landing `--max-ctx 4096 → 32768` in `scripts/serve_v100.sh`. Deployment followed
what `docs/serve-v100.md:37` says: TERM the **child**, leave the supervisor, and it
reboots at the new code. The log agreed:

```
=== boot 3 at 2026-09-05T09:36:51+08:00  sha 6a4e737 dirty 0 ===
```

`6a4e737` is the commit carrying 32768, `dirty 0`, and the checkout on disk had
`--max-ctx 32768` at line 54. Every signal said the change was live.

`/health` returned **`blocks_total: 256`** — the same number as at 4096.

## Root Cause

**Bash parses the whole `for` loop before executing it, so the supervisor relaunches
the child from the flag string it read at its own start.** That supervisor started
`01:28:58`; `serve_v100.sh` was rewritten by the checkout at `09:36:42`, eight hours
later. The `git rev-parse --short HEAD` inside the loop body *does* run fresh each
boot — so the sha stamped into the log was current while the command line beside it
was eight hours stale.

Read directly, the child's own `/proc/2151849/cmdline`:

```
… --host 0.0.0.0 --port 8000 --max-batch 1 --max-ctx 4096
```

Both facts in one log line, disagreeing: `sha 6a4e737` (fresh) and, implicitly, the
4096 flags (stale). The line is not wrong about the sha. It is silent about the flags,
and the sha invites you to conclude the flags came with it.

**Why it survived the first check.** `blocks_total: 256` is exactly what
`--max-ctx 4096 --max-batch 1` produces, and I had *just* written an entry explaining
that 256 is the cap rather than the fit. So the number I saw was the number I had
spent the previous hour explaining — the failure mode where a stale reading matches a
freshly-built expectation. Had 32768 been live, `blocks_total` would have been 2048.

## Fix

For a change to the supervisor script itself, restart the supervisor:

```bash
kill -TERM <supervisor-pid>              # both processes go; card free in ~10 s
cd ~/tilerl-git && setsid nohup scripts/serve_v100.sh >/dev/null 2>&1 &
```

Verified after: child `2152549`, `/proc/…/cmdline` ends `--max-ctx 32768`.

`docs/serve-v100.md` now carries this as a second case beside the TERM-the-child
rule, since the two are opposite and the log cannot distinguish them.

## Rule

**A restart's boot line proves which commit is checked out, never which flags the
running process received.** Verify a deployed flag by reading
`/proc/<pid>/cmdline`, not by the sha, not by `git status`, and not by the flag's
value on disk — a long-lived bash supervisor holds the loop body it parsed at its own
start. And when the confirming number equals the number the old configuration
produces, it confirms nothing.
