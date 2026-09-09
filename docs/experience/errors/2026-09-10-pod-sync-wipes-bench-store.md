# pod_sync.sh wipes the bench store on every sync

## Context

An eval-only run (`--steps 0`, recipe `grpo-gsm8k-27b`, card 6, run
`5b50c39cd5b2`) produced gsm8k_pct and rollout_tokens for both before and
after arms. `_emit_eval_records` appended four rows to the bench store on
the pod. A background polling loop that checked the process state every
20 seconds via `pod_sync.sh` overwrote the store with the local (stale)
copy each time, wiping the rows minutes after they were written. The
process completed cleanly; the loss was invisible until the store was
checked and found to still have 44 rows.

The eval numbers themselves are recovered and in the store:
gsm8k_pct 91.4% (before) / 91.6% (after), rollout_tokens 346.8 / 345.0.
This is the third independent reading of the base (previous two: 91.0%),
confirming the verdict that P1's +5 pt gate is unsatisfiable — the gate
demands 96.4% against a 91.4% base. See
[errors/2026-09-08-difficulty-is-the-task-hesitation-is-the-model.md](2026-09-08-difficulty-is-the-task-hesitation-is-the-model.md).

## Root Cause

`docs/experience/bench/measurements.jsonl` lives inside the synced tree.
`pod_sync.sh` tars the entire worktree (excluding only `.venv`,
`__pycache__`, `.git`, `*.pyc`, `*.egg-info`) and extracts it over the
remote tree. Every sync is a full overwrite. The store is both a tracked
file (synced) and runtime data (appended by `benchrec.append()`), and
these two roles conflict: any sync after a job writes rows silently
reverts the store to the local snapshot.

The sync mechanism has the power to delete runtime data, and it does not
know what it is deleting.

## Fix

`pod_run.sh` now exports `TILERL_BENCH_STORE=/work/tilerl-bench/measurements.jsonl`,
outside the synced tree. On every launch, tracked rows the pod store
lacks are merged in by id (idempotent, deterministic) — seed-once would
drift as main lands new rows, and two consumers (`measured_best_floor`,
`known_devices()`) read the pod store. The `benchrec.py`
`TILERL_BENCH_STORE` env var already existed; the fix wires it into the
launcher.

Also fixed: `_emit_eval_records` hardcoded `"H20"` as the device name,
splitting card 6's population from the 37 rows that record
`"NVIDIA H20"`. Now reads `torch.cuda.get_device_name(backend.device)`.
The `--device-name` known-set gate (#418) does not cover this path —
`_emit_eval_records` bypasses `record_common`, so the store has two
write entry points and the gate guards only one.

## Rule

Never use `pod_sync.sh` to poll a running job's state — every call
overwrites the tree, including the bench store. Use `tn exec` + `crictl
exec` for read-only checks. The store is runtime data, not code; it
lives outside the tree on the pod.
