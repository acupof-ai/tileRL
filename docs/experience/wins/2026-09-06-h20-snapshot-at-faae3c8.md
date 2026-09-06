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

**One GRPO step** (`train --model qwen38-27b --rl --steps 1 --max-new-tokens 2048
--data /work/gsm8k_train.jsonl`, group 8, adapter 124.8M params, peak 88.21 GiB):

| phase | s |
|---|---:|
| rollout | **73.183** |
| backward | **44.697** |
| optimizer | 0.165 |
| **total** | **118.1** |

Against the 229.2 s median cited for run 2, that is **1.94x faster**. The split
is 62% rollout / 38% backward.

## Three readings that would mislead the next agent

**The run exits 1 and the ledger says FAIL, and the timings are still valid.**
The failing gate is `groups_untied`: value 1.0 against threshold 0.5. All eight
rollouts in the group tied, so the advantage is zero and this step carried **no
learning signal** — but rollout, backward and optimizer all executed, so 118.1 s
is a real measurement of the step's cost. What it is not is a measurement of a
step that learns. `reward 1.0000`, `ce 2.5102`, `tok 174` on one gsm8k group is
consistent with a tie, not with a broken run.

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

A ledger FAIL and an invalid measurement are different things: read which gate
failed before discarding the numbers. Here the gate is about whether the step
*learned*, and the question asked was what the step *costs*.

## Results

| date | commit | machine | target | model | prefill ms/tok | decode ms/tok | throughput tok/s |
|---|---|---|---|---|---:|---:|---:|
| 2026-09-06 | faae3c8 | H20 card 6 | cuda sm90 | Qwen3.8-27B NVFP4 | 0.3311 (512) | 10.61 (B=1) | 94.3 (B=1) |
| 2026-09-06 | faae3c8 | H20 card 6 | cuda sm90 | Qwen3.8-27B NVFP4 | 0.3483 (2048) | 23.7 (B=8 tick) | 354.8 (B=8 agg) |
| 2026-09-06 | faae3c8 | H20 card 6 | cuda sm90 | Qwen3.8-27B NVFP4 | 0.3677 (8192) | — | 2719.6 (prefill 8192) |
| 2026-09-06 | faae3c8 | H20 card 6 | cuda sm90 | Qwen3.8-27B NVFP4 + LoRA 124.8M | — | — | 118.1 s/GRPO step (73.18 rollout / 44.70 backward) |

Raw artifacts: `/work/snap1.log`, `/work/snapb8.log`, `/work/grpo1.log`,
`runs/8d4f82034be3/manifest.json`, `runs/8d4f82034be3/rollouts.jsonl` (all on the
pod).
