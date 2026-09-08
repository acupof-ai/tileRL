# A blind spot one flag wide — and the flag was not the cause — 2026-09-08

**Status:** the launcher fix is in `scripts/pod_run.sh`, gated by selftest arm 7 with a negative
control. **The diagnosis that motivated it was wrong**, and the real cause is recorded below.

## Context

A 500-question level-5 MATH eval ran on card 0. Expected duration, computed from the measured
29.67 ms decode tick at B=8 and the measured 1029-token base mean: **31.8 min** at that mean,
**63.3 min** if every completion saturates the 2048 cap. At 43 minutes it was inside that window
and its log held **zero bytes**. Two sessions independently began treating the healthy run as
dead.

## What I claimed, and why it was wrong

I attributed the empty log to `python3` block-buffering stdout to a file, launched a second eval
with `-u`, saw **2720 bytes in 30 s** against the first run's 0 bytes at 43 min, and called that a
controlled comparison — same binary, same command, one flag apart.

**It was not a controlled comparison.** Two things differed, and neither was the flag:

1. **All 20 lines of those 2720 bytes were TileLang's own logger** (`[TileLang:tilelang.jit.kernel:INFO] … begins to compile kernel`). That logger writes through Python's
   `logging` to **stderr**, which is unbuffered-ish by line and, more to the point, is not what
   `PYTHONUNBUFFERED` was being credited for.
2. **The first `-u` run compiled kernels; the second did not.** The TileLang cache held 18 entries
   by then, so the relaunched job emitted nothing at all — **0 bytes at 8:13 with `-u` set**,
   which is the same reading I had called proof of buffering.

So the flag's effect and the cold-cache effect were confounded, and the run I used as the
"unbuffered" arm was really the "cold cache" arm.

## The real cause

`cli.py:559`:

```python
log = (lambda *a, **k: None) if args.json else print
```

**With `--json`, every progress line is discarded before it reaches stdout.** The manifest is
printed once, at the end (`cli.py:1059`). So an eval-only run launched with `--json` has *no
progress output to buffer or unbuffer* — the log is empty because nothing is written, not because
something is held.

That also explains the shape cleanly: the eval arm writes `eval-before.jsonl` only on completion,
and the run directory holds just the manifest written before the arms. **A 60-minute eval and a
hung eval are indistinguishable for 60 minutes, and `-u` does not change that by one byte.**

## What the fix is and is not

`setsid env PYTHONUNBUFFERED=1 $CMD` is still right, and it is kept: any pod job that *does*
print progress and is not run with `--json` would otherwise have it held in a 4–8 KB block, and
relying on each caller to remember `-u` is how this class of gap persists. `PYTHONUNBUFFERED`
rather than a `-u` in argv because `$CMD` is usually `bash -c '… python3 …'`, which has no argv
position for a flag on the inner interpreter.

**It does not fix the observability of a `--json` run.** That needs one of:

- progress that does not route through `log` — `gsm8k_accuracy` writing a row per N problems, or
- `--json` writing progress to **stderr** while keeping stdout a single parseable object.

The second one ships here: `_progress(as_json)` returns `print` normally and a stderr-and-flush
printer under `--json`, replacing the no-op at both call sites. stderr rather than stdout because
`--json` exists so a caller can parse the manifest, and a caller finds it by first brace
(`tests/test_recipes.py:47`) — stdout already carries TileLang's C++ cache warnings, so it was
never a lone object, and keeping *our* lines off it is what leaves that parse workable. The test
asserts both halves, because either alone passes with the defect facing the other way: silence
satisfies "stdout parses", and printing to stdout satisfies "progress exists".

The second is smaller and preserves the reason `--json` silences `log` at all: stdout must stay
one JSON document. Filed rather than done here, because the run this was diagnosed from is still
on the card and the change touches the eval arm.

## Verification of what did land

Selftest **arm 7** asserts the launcher hands python an unbuffered stdout **through** a `bash -c`
wrapper — the case `-u` cannot reach — and reads what python sees, `sys.stdout.write_through`,
rather than grepping the runner text. `write_through` is the effect; the environment variable is
the mechanism, and a test of the mechanism would pass on a runner that set it where it had no
consequence.

| mutant | test |
|---|---|
| drop the `env PYTHONUNBUFFERED=1` prefix | **FAILS** (`UNBUF None WT False`) |
| control | passes |

## Two of my own defects, both caught by gates that already existed

**The comment warning about backticks contained backticks.** The runner heredoc is
`<<RUNNER_EOF`, unquoted, so a backtick inside it is command substitution **on the caller** — my
comment wrote the flag name in backticks and the assembly ran it, printing `-u: command not found`
to stderr. Selftest arm 0 exists to catch exactly that and did. I first misread the reported line
number as pre-existing code rather than my own addition.

**A hardcoded `CUDA_VISIBLE_DEVICES=0` in a job launched against card 6.** The claim check refused
and killed it, with the diagnosis already written: *"the claim would protect cards the job cannot
touch while it runs on cards nobody claimed — an orphan behind a healthy claim."* Without that
guard the hedge run would have contended on card 0 with the very run it was launched to hedge.
Relaunched and verified **from the device**: card 6 at 28801 MiB, card 0 unchanged.

## Also settled

`/proc/<pid>/fd` cannot answer which card a job uses — every process in the container has all
eight `nvidia*` nodes open from the device mount, so 48 fds proves the process *can see* the
cards, not which one it computes on. `CUDA_VISIBLE_DEVICES` read at exec time is the authority;
`nvidia-smi`'s per-card memory row is the corroboration. And `nvidia-smi --query-compute-apps`
reports **host-namespace** pids, which need not match pids inside the container
(`/proc/<pid>/status`'s `NSpid` says whether they agree).

## Rule

**A two-arm comparison needs the arms to differ in one thing, and a warm cache is a second
thing.** My arms differed in the flag *and* in whether TileLang had to compile — and the output I
read as the flag's effect was entirely the compiler's. The tell was available immediately: the
2720 bytes were all one logger's lines, and reading three of them would have shown they had
nothing to do with the program's own progress output.

**Before fixing a silence, find the writer.** I went from "the log is empty" to "stdout must be
buffered" without checking whether anything writes to it. One grep — `log = (lambda …) if
args.json` — was the whole answer, and it was upstream of every measurement I then took.

## A second guard that tested the build, not the device

Found while running the suite for this branch, unrelated to it and older than it:
`test_the_fp8_kv_pool_generates_what_the_bf16_pool_does` has always failed on `metal` with
`RuntimeError: Undefined type Float8_e4m3fn` at `kv_cache.py:97`. Its skip read

```python
if not hasattr(torch, "float8_e4m3fn"):
```

**`hasattr` is a property of the torch BUILD; allocation is a property of the DEVICE.** The
build has the dtype on every target we run, so the guard passed everywhere, and on mps the
allocation one line later raises. Confirmed pre-existing: the same test fails on `metal` and
passes on `cpu` at `ac41db4`, which carries none of this branch's changes.

It was invisible because CI runs `TILERL_TARGET=cpu` only, and the local metal run is the only
place the difference appears. Now guarded by attempting the allocation on
`get_backend().device`, which skips on metal and still runs on cpu — checked both ways, because
a skip guard that is too broad passes by testing nothing.

**Same shape as the entry above.** Both are a check that reads an adjacent property: `hasattr`
for "can this device hold it", and a log's byte count for "is this process alive". The adjacent
property was available and cheap, and answering it felt like answering the question.
