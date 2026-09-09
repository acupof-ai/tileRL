# A trap string is parsed twice

2026-09-10. `scripts/pod_sync.sh` run mode, `scripts/pod_fan.sh` — the `.pod_running` trap.

## Context

The `.pod_running` marker (this PR) makes a pod sync refuse to wipe a tree a detached
job still runs in. The job writes its `pid start-time` line on start and removes it on
exit via a trap:

```bash
trap "sed -i.bak /^$$[ ]/d $REMOTE_DIR/.pod_running 2>/dev/null; ..." EXIT
```

The pod run finished green — `DONE_podrune2e` printed, the trap fired (visible in the
`set -x` trace) — and the marker line survived.

## Root cause

The trap string is parsed twice: once when `trap` is called, and again when the trap
**fires**. At assembly the pattern sits inside double quotes, so `/^$$[ ]/d` is one
word. On EXIT the string is re-parsed as a bare command, and the space inside `[ ]`
word-splits it:

```
+ sed -i.bak '/^2287233[' ']/d' /work/.../.pod_running
```

sed received the script `/^2287233[` (unbalanced bracket) and `]/d` as a *filename*;
both errors went to `/dev/null`, the line stayed. A trap that fails silently is the
worst kind — the trace showed it running.

## Fix

Use a pattern with no space: `/^$$[[:space:]]/d`. `pod_run.sh`'s `release()` needed no
change — its pattern is double-quoted *in the runner*, so the quotes survive re-parsing.

## Rule

A trap command string is code that gets parsed a second time, in a context where the
quotes that held it together at assembly are gone. Put no unquoted whitespace in any
word of it — or quote inside the string so the second parse sees the quotes. A `2>/dev/null`
on a cleanup trap hides exactly the failure that leaves the garbage behind; verify the
cleanup happened, not that the trap ran.
