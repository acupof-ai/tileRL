# Four launches of one cell, four silent failures — H20, 2026-09-08

**Date:** 2026-09-08
**Machine:** H20 pod, card 0, `/work/tilerl-s-v100-sm70-fp4` at a43a379
**Status:** open — two of the four causes are unfixed bugs in `scripts/pod_run.sh`

## Context

The two-arm DRAM cell for #271 (fixed publisher × tier {off, on}) took four launches to start. No
launch printed a traceback the caller could see, and two of them would have produced a complete,
internally consistent table rather than an error. The cell itself is not the subject here; the
launcher is.

## The four

**1. `pod_run.sh` does not wait, and its header says the opposite loudly enough to be believed.**
Its comment block opens with "a bash parent that WAITS, so the job is reaped" — true of the
*pod-side* bash it installs, and irrelevant to the caller. `pod_exec` ends with
`setsid bash /work/pod_run_$NAME.sh ... & sleep 2; echo started`, so the script returns 0 about two
seconds after launch. A `for ARM in off on` loop therefore puts both arms on the card at once: two
27B servers, one card, one port 8000. Caught by reading `pgrep` output rather than the exit code,
before arm `on` went out.

**2. The command after `--` is re-split on whitespace.** `CMD="$*"` at line ~49, then
`setsid $CMD` unquoted inside the runner. A quoted `bash -c '<multi-line script>'` arrives as
`bash -c` with no argument: `bash: -c: option requires an argument`, exit 2, card claimed, nothing
running. The failure reached `/work/pod_run_<name>.out`; the caller saw `started` and exit 0.

**3. `uv run` in the arm resolved a torch newer than the driver.** `pod_sync.sh` wipes the tree, so
`uv run` rebuilds the venv on the pod and picks a wheel by its own resolution, not the box's:
`RuntimeError: The NVIDIA driver on your system is too old (found version 12090)`, raised inside
`get_backend()` before `/health` ever answered. The pod's actual environment is `/work/tl013/bin` +
`PYTHONPATH`, which `pod_run.sh` already exports; `/work/tl013/bin/python3` has torch 2.11.0+cu129
with `cuda.is_available() True`. Calling `python3` directly is both correct and skips a venv build
per arm.

**4. `serve --model` defaults to `tiny`.** `cli.py:1001` —
`choices=["tiny","tiny-agent","qwen38-27b"], default="tiny"`. Both arms would have served a
random-weight toy, completed, agreed with each other, and reported wall-clock and reuse counts for
a model with no 157 MiB snapshots and therefore no budget pressure at 1 GiB. Nothing in the bench
output names the model. This is the one that would have shipped a number.

## The fifth is a real defect, and calling it cosmetic was wrong

Once arm off was actually running, its wrapper log read:

```
pod_run: job pid 2772963, log /work/pu/work/pod_run_pub271off.sh: line 73: 0: command not found
 to 300s for a device fd
```

First reading, recorded here because it was wrong: `nvidia-smi` output interleaving with an
unflushed `echo` in the same redirect — cosmetic. The process table did show one clean tree
(wrapper → arm → one 27B serve → one bench, no duplicate, claim held), which is what made
"cosmetic" feel settled.

A peer refused it on a mechanism argument: bash prefixes `line N:` only when it looks up a real
command, so interleaved stdout cannot produce that string. Correct. The check it proposed then
ruled out its own hypothesis too:

- `sed -n '73p'` of the pod-side runner → `wait $JOB; rc=$?`, clean under `cat -A`.
- `grep -n '^[[:space:]]*0[[:space:]]*$'` over all 76 lines → no match. No bare `0` anywhere,
  so argv interpolation did not emit a stray word.

So bash executed a word `0` at line 73 of a file that contains no such word — only possible if the
file changed underneath it. Bash reads a script by byte offset, and `pod_run.sh` writes
`/work/pod_run_$NAME.sh` at a **fixed path**: an earlier failed launch's wrapper was still parked in
`pod_run_claim`'s 300 s poll when the next launch's `base64 -d >` truncated and rewrote that path
under it. It resumed at a stale offset and ran a fragment. The truncated `log /work/pu` is the same
collision on `/work/pod_run_$NAME.out`, which both wrappers had open.

Reproduced standalone rather than left as a story — `/tmp/probe_byte_offset.sh`, a script that
rewrites itself mid-`sleep`:

```
bo/s.sh: line 62: 0: command not found
replaced-tail
REPRODUCED: a mid-execution rewrite yields 'line N: <word>: command not found'
```

So the log under-reported four real failures as clean tables **and** over-reported one real defect
in a form I misread as noise. The process table was the right authority for "is one arm running";
it could not see a corrupted script, because a wrapper reading a rewritten file looks exactly like
a healthy one.

## Root cause

The three fixable-by-me causes share a shape: **the launcher's success signal is decoupled from the
job's.** `pod_run.sh` returns 0 for "I handed it off", and every real failure lands in a file on the
pod that nothing reads. Cause 4 is a different shape and the more dangerous one — a default that is
correct for a smoke test and silently wrong for a measurement, on a flag whose value never appears
in the output.

## Fix

Cell-side (done, `/tmp/cell_271.sh`): arms are files under `/work` passed as single words; each arm
polls its own `POD_RUN_DONE_<name>` line with a 240-tick cap before the next launches; arms call
`python3` and pass `--model qwen38-27b` explicitly; the serve wait loop checks `kill -0 $SRV` each
tick so a dead server exits 5 instead of running the 240-tick timeout to completion.

Launcher-side (not done, needs its own PR since every session uses the file): quote the command
(`"$@"` through the runner instead of `$CMD`), and either block until `POD_RUN_DONE` or say in the
usage line that it does not.

## Rule

A launcher's exit code attests to the handoff, not to the job. Poll for the job's own completion
line, and read the log the launcher wrote rather than the status it returned.

A measurement's model, precision and budget are arguments, never defaults. A default that is
sensible for a smoke test produces a clean comparable table for the wrong system, and no field of
the output disagrees.

And the fifth one's rule, which is about diagnosis rather than launching: **a symptom's mechanism
has to be named before it can be dismissed.** "Cosmetic interleaving" was a category, not a
mechanism, and it happened to cover the evidence. `line N: <word>: command not found` is emitted by
exactly one thing — a command lookup — and once that is stated, interleaving is excluded without any
further data. The peer who refused it had less information than I did and was right anyway, because
it reasoned from what produces the string instead of from what the string resembles.
