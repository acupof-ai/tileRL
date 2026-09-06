# H20 snapshot at current main — sm90, 2026-09-06

> Status: Shipped. Newest snapshot on file was 2026-09-03; every row moved.

## Context

The newest bench snapshot on file was 2026-09-03 (`9e3836b`). This is a dated
snapshot of the three serving/training shapes at current main on an idle H20
card 6, real 27B NVFP4 — not a treatment/control comparison, so no row here
claims a cause.

Pod state, because the pod checkout is **not a git repo** and its version cannot
be read with `git log`: `pod_sync.sh` wiped and repushed it from a detached
`faae3c8`, `.synced_commit` reads `faae3c8`, `_scanned` is present 5x (#160 in)
and `ssd_save_ms` absent (#158 not in) — so the pod matched main rather than any
of my branches. Card re-checked immediately before each launch: all eight cards
0 MiB, no `card_claims` dir, no compute apps.

## What Worked

Three runs, `scripts/pod_run.sh <name> 6 -- …`, model `/work/Qwen3.8-27B-NVFP4`.

**Decode and prefill** (`bench_qwen38_baseline.py`, in-process, decode graph on,
warmup 25.2 s then 0.2 s, peak 30.0 GiB):

| shape | this run | newest on file | moved |
|---|---:|---:|---:|
| decode B=1 | **94.3 tok/s** (10.61 ms/tick) | 92.4 (`unknown`, 08-28) | +2.1% |
| prefill 512 | **3020.1 tok/s** (0.3311 ms/tok) | 2689.8 (`9e3836b`, 09-03) | **+12.3%** |
| prefill 2048 | **2870.8** (0.3483) | 2671.7 (`9e3836b`, 09-03) | +7.5% |
| prefill 8192 | **2719.6** (0.3677) | 2558.6 (`9e3836b`, 09-03) | +6.3% |

**Batch decode** (`bench_decode_b8.py`, 8 concurrent, 61 steady ticks):

| shape | this run | newest on file | moved |
|---|---:|---:|---:|
| B=8 aggregate | **354.8 tok/s** (23.7 ms/tick) | 328.4 (`6e260b8`, 08-29) | +8.0% |

All eight replies were semantically correct (Paris, Jupiter, Shakespeare, Au,
Everest, yen, Portuguese) — worth stating because a degenerate model produces a
plausible tok/s on garbage, and this run's tokens are real.

**GRPO** (`train --model qwen38-27b --rl --max-new-tokens 2048 --data
/work/gsm8k_train.jsonl`, group 8, adapter 124.8M params). Two runs, and the
first one's number is **not** a step cost:

| run | tilelang cache | step 1 total | rollout | backward | optimizer |
|---|---|---:|---:|---:|---:|
| A, `--steps 1` | cold | 118.1 s | 73.183 | 44.697 | 0.165 |
| B, `--steps 3` | warm from A | **25.3 s** | **16.959** | **8.070** | 0.241 |

**4.67x of run A is one-time cost** — 4.32x on rollout, 5.54x on backward. Same
config, same data, same cap, same card; the only difference is that A compiled the
kernels and captured the graphs inside its own timing. The B=1 round in this same
snapshot showed the same shape openly (warmup 25.2 s, then 0.2 s) and I still
quoted A as a step cost.

The per-token check is what makes B credible: **0.097 s/token** of rollout (16.959
over 174 tokens) against the 0.077 s/token from run 2's MATH rows — **1.3x**, where
run A read 0.420 s/token, **5.5x**. A 5.5x gap against a known number is an
instrument problem, and it was.

**Run B died at step 2 with CUDA OOM**, so no steady-state median exists at this
cap: `Tried to allocate 296.00 MiB. GPU 0 has 95.22 GiB of which 221.56 MiB is
free. Process 750071 has 95.00 GiB in use` — one process, not contention. Peak in
run A was 88.21 GiB.

**No capacity fact is claimed, and the cause is now measured.** MATH run 2
(`0f7006c74ea0`) ran **45 steps** at group 8 and cap 2048 on one H20 without OOM
([the rollouts grew into the cap](../errors/2026-09-06-the-rollouts-grew-into-the-cap.md)),
so "cap 2048 does not fit" was wrong. The configurations differed on a variable
neither side had listed: `recipes.py:36` gives `grpo-math-27b` **`micro=1`** and my
runs took the CLI default **`micro=0`**, which `train.py:137` turns into one
backward over all 8 group rows instead of one row at a time. Rerun at `--micro 1`:
**3/3 steps, peak 44.55 GiB against 88.21, median 67.3 s/step**
([the OOM was micro=0](../errors/2026-09-06-the-oom-was-micro-zero.md)). So the
snapshot's GRPO row is a `micro=0` measurement, and `micro=0` at this group and cap
is what does not fit.

That probe also supplies the valid comparison this entry declined above: **67.3
s/step median against P1's 56.88 — 1.18x**, same task, same group, same `micro=1`,
differing only in cap (2048 vs 256). The 25.3 s warm first step stands as a warm
first step and not as a median.

Run 2's own peak, which would have answered this in one line, was never written:
per tilerl-25, who ran it, run 2 died on SIGTERM at step 45 and `write_manifest`
lived only inside `_finish` until `8388cbf`. **I wrote here that my sync destroyed
it; that was wrong** — corrected in
[the OOM was micro=0](../errors/2026-09-06-the-oom-was-micro-zero.md), which also
lands the `runs/` exemption `pod_sync.sh` needed anyway.

**No comparison to run 2's 229.2 s is made.** That row is MATH level 5 at a
1434-token mean; this is gsm8k at 174 tokens, and #140's buckets put the backward
at width 256 against run 2's 2048. The ratio would measure the task and the bucket.
The one same-task reference is P1's **56.88 s/step median** (`wins/2026-09-05-p1-grpo-27b-run.md:22`),
also gsm8k, also group 8 — but at cap **256** against this cap **2048**, and 25.3 s
is a warm *first* step rather than a median over 100. Both differences push the
same way, so the honest statement is that no valid s/step comparison exists yet at
cap 2048; what run B establishes is that a warm step is ~4.7x cheaper than the
cold one, and that step 2 does not fit.

## Three readings that would mislead the next agent

**The run exits 1 and the ledger says FAIL, and the phase split is still valid.**
The failing gate is `groups_untied`: value 1.0 against threshold 0.5. All eight
rollouts in the group tied, so the advantage is zero and the step carried **no
learning signal** — but rollout, backward and optimizer all executed, so the split
is real work. P1's own run FAILS the same gate at 0.81
(`wins/2026-09-05-p1-grpo-27b-run.md:33`), so a tie is the common case on gsm8k, not
a broken run. `reward 1.0000`, `ce 2.5102`, `tok 174` is consistent with it.

**`0.0 step/s` in the bench summary is display precision, not zero.** `1.0/118.1`
is 0.00847, and the summary formats to one decimal. Reading that line as a failed
timing is the trap.

**No `baseline-candidate.json` was written, and that is correct.** The row is
`SEED train-run/qwen38-27b-grpo-g8-t2048/cuda` — the shape key encodes
`t2048` and no such key existed, so there is nothing to beat and
`Gate.finish` writes candidates only when `self.candidates` is non-empty
(`bench_harness.py:117`). A missing candidate file here is the seed path, not a
suppressed promotion.

## What is not claimed

**No row is attributed to a PR.** Every prefill row moved between `9e3836b`
(09-03) and `faae3c8`, and the candidate PRs in that window are nameable, but
picking one without a bisect is a guess wearing provenance. Agreed with 27 that a
27B bisect costs more card time than the answer is worth today, so the rows read
*moved, unattributed, between 9e3836b and faae3c8*.

**The decode delta is against an unprovenanced row.** `decode-kv/d512-b1/sm90`
carries `commit: "unknown"`, dated 08-28. +2.1% against an unknown commit is not
a delta worth quoting; what this run supplies is the first fully-provenanced
value for that shape (`faae3c8`, card 6, `bench_qwen38_baseline.py`,
`/work/Qwen3.8-27B-NVFP4`), and the old row should be treated as superseded
rather than compared against.

**`bench-baseline.json` is untouched.** Per #112 a beat is a candidate first and a
promotion is deliberate; nothing here is promoted.

## Rule

**A single-step run times its own warmup.** Run A's 118.1 s is 4.67x run B's warm
first step at the same config, and I published it as a step cost while this same
snapshot's decode round printed `warmup pass 1: 25.2s / pass 2: 0.2s` three
paragraphs earlier. One step is never a step: the first one carries JIT and graph
capture, so a cost claim needs at least a second step on a warm cache — or, when
step 2 does not fit in memory, an explicit statement that no steady-state number
exists.

Second: a per-token cross-check catches this without a rerun. 0.420 s/token against
a known 0.077 is 5.5x, which is the "suspect the instrument" threshold; the warm
step reads 0.097, 1.3x.

Third: a ledger FAIL and an invalid measurement are different things. Read which
gate failed — `groups_untied` asks whether the step *learned*, and P1's shipped run
fails it too at 0.81.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | faae3c8 | H20 card 6 | cuda sm90 | Qwen3.8-27B NVFP4 | 0.3311 (512) | 10.61 (B=1) | 94.3 (B=1) |
| 2026-09-06 | faae3c8 | H20 card 6 | cuda sm90 | Qwen3.8-27B NVFP4 | 0.3483 (2048) | 23.7 (B=8 tick) | 354.8 (B=8 agg) |
| 2026-09-06 | faae3c8 | H20 card 6 | cuda sm90 | Qwen3.8-27B NVFP4 | 0.3677 (8192) | — | 2719.6 (prefill 8192) |
| 2026-09-06 | faae3c8 | H20 card 6 | cuda sm90 | Qwen3.8-27B NVFP4 + LoRA 124.8M | — | — | **25.3 s** warm step 1 (16.96 rollout / 8.07 backward); cold 118.1 s; **step 2 OOM at cap 2048, no median** |

Raw artifacts: `/work/snap1.log`, `/work/snapb8.log`, `/work/grpo1.log`,
`/work/grpo3.log`, `runs/8d4f82034be3/manifest.json` and its `rollouts.jsonl` (all
on the pod).
