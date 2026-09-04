# A full /tmp reads as a run that produced nothing, and I diagnosed it four ways first

**Date:** 2026-09-04
**Arch:** sm70 (Tesla V100-SXM2-32GB), pod-side tooling
**Task:** #71

## Context

Launched the depth x context sweep, checked the log, got **0 bytes**. Checked again
later: still 0 bytes, and the pid had changed. Checked a third time: 0 bytes, no
process at all, GPU empty.

Each of those readings got an explanation, and each explanation was wrong:

1. "Python is block-buffering to a pipe" — plausible, and the reason I added `-u`.
2. "Run 1 finished and run 2 truncated it" — I read `O_TRUNC` in
   `/proc/PID/fdinfo/1` and the log's mtime matching run 2's start. Both facts were
   real; the conclusion did not follow. `O_TRUNC` in `fdinfo` records the flags the
   file was *opened* with, which every `>` redirect sets, so it says nothing about a
   second truncation.
3. "The `&` backgrounded the whole `&&` chain, so the script never started" — this
   one was true for one launch attempt, and fixing it did not fix the symptom.
4. "`setsid` did not detach and the shell died with the ssh session."

The actual cause, found by running a three-line script that only echoes:

```
$ printf ... > /tmp/probe.sh
zsh:printf:1: write error: no space left on device
```

**`/` is 100% full with 0 bytes available, and `/tmp` is on `/`.** Every log write
in the whole sequence failed silently. There was never any output to buffer, truncate
or lose.

```
/dev/vda1   99G  95G     0 100%  /        <- /tmp lives here
/dev/vdb   492G 421G   46G  91%  /data00
```

Relaunched with the log on `/data00`: **1840 bytes within 25 s**, `set -x` trace
visible, run healthy.

## Root cause

A shell redirect to a full filesystem does not fail the command. The process starts,
runs, and its writes are discarded, so the observable is an empty file next to a
working process — indistinguishable from buffering, from a truncating second writer,
and from a launch that never happened. All three of those are real failure modes I
have hit on this box, which is exactly why each had a ready explanation.

`/tmp/hf_cache` is **37 GB of it** — a `Qwen3.5-9B` snapshot under this account, not
another user's as an earlier note in this repo claimed. Not deleted here: it is not
in the way once logs go to `/data00`, and deleting 37 GB of someone's weights is not
a side effect a bench run gets to have.

## Fix

Logs go to `/data00`, never `/tmp`, on this pod. The runner also gained two things
worth keeping for their own sake:

- `python -u`, so a run that dies mid-way still shows where. Without it the rows only
  exist if the process exits cleanly, which is backwards for a script whose job is to
  survive long enough to be interrupted.
- `>>` instead of `>`, so a second arm cannot overwrite the first's rows.
- `set -x`, which is what would have shown the empty log was empty *from line 1* —
  before any Python ran, before buffering could be the explanation.

## Why the four wrong diagnoses all survived

Each was checked against evidence, and the evidence was real. What was missing is
that none of them was checked against a **negative control**: a run that produces
output for certain. The three-line echo script is that control, and it took one
command. Four rounds of `/proc` forensics on the hypothesis I already had cost more
than that and could not have reached the answer, because a full disk looks the same
as every hypothesis I was testing.

Worse, one of the four was promoted to a report: I wrote "run 1 finished and run 2
truncated it" with the fd flags quoted as proof. The flags were quoted correctly and
the inference from them was invented — the same defect as tonight's `NB` mechanism
and the stale-branch byte count.

## Rule

When an instrument reads empty, prove the instrument can read non-empty before
explaining the emptiness. One echo through the same path settles it.

And check the boring resource first. Disk, memory, permissions, quota — a full
filesystem produces a silent, universal, mechanism-shaped symptom that will fit
whatever hypothesis you brought, and `df` costs nothing. I had already seen this
disk at 100% earlier the same day and used it to explain a *different* failure
(`nvcc fatal: Failed to preprocess host compiler properties`), then did not check it
when the next silent failure appeared.
