#!/bin/bash
# Gate: training start is deterministic across before-arm cache states.
#
# Same sha, same seed, two 3-step GRPO runs: arm A skips the before arm (clean
# engine), arm B runs it (--eval-n 100, pre-warmed engine). Step 1-3 rollouts
# must be identical row for row. Without the pool reset in cli.py the two paths
# diverge from step 1 (measured 17/24 and 23/24 rows differing, 2026-09-09);
# with it, both converge to the same canonical pool state.
#
# TILERL_POOL_RESET=zero|sort|both attributes the fix (the 2026-09-09 A/B):
#   sort  -> green means free-list order was the mechanism (pure numerics)
#   zero  -> green means freed blocks' stale K/V was read (serving correctness bug)
#   both  -> the shipped default
# TILERL_GATE_REPRO=1 runs arm A twice and diffs A1-vs-A2: the gate is only
# interpretable if the clean path is reproducible against itself.
#
# Pod-only: needs the 27B and a GPU. Exit 0 green, 1 red.
set -uo pipefail
TREE=${TREE:-$(cd "$(dirname "$0")/.." && pwd)}
DATA=${DATA:-/work/p1_gsm8k_train.jsonl}
EVAL=${EVAL:-/work/p1_gsm8k_test.jsonl}
cd "$TREE"
export TILERL_POOL_RESET=${TILERL_POOL_RESET:-both}
# Per-arm runs dirs: A1 and A2 share the clean config, hence the same run id, so one
# TILERL_RUNS would make A2 refuse to rerun a finished run. A fresh dir per arm also
# guarantees B's before-arm cache misses (a hit would skip the eval and make B clean).
RUNS=$(mktemp -d /tmp/gate_ds_runs.XXXXXX)
COMMON="--recipe grpo-gsm8k-27b --data $DATA --steps 3 --eval-mmlu 0 --eval-every 0 \
  --length-penalty 0.0 --allow-short-rollouts"

# A 3-step run exits non-zero (a gate at _finish fails with no curve points); the
# rollouts file is the success criterion, not the exit code.
TILERL_RUNS=$RUNS/a1 python -m tilerl.cli train $COMMON > /tmp/gate_ds_a1.log 2>&1
A1_DIR=$(ls -td "$RUNS"/a1/*/ | head -1)
if [ ! -f "$A1_DIR/rollouts.jsonl" ] || [ "$(wc -l < "$A1_DIR/rollouts.jsonl")" -ne 24 ]; then
  echo "arm A1 (clean) failed"; tail -5 /tmp/gate_ds_a1.log; exit 1; fi

A2_DIR=""
if [ "${TILERL_GATE_REPRO:-0}" = "1" ]; then
  TILERL_RUNS=$RUNS/a2 python -m tilerl.cli train $COMMON > /tmp/gate_ds_a2.log 2>&1
  A2_DIR=$(ls -td "$RUNS"/a2/*/ | head -1)
  if [ ! -f "$A2_DIR/rollouts.jsonl" ] || [ "$(wc -l < "$A2_DIR/rollouts.jsonl")" -ne 24 ]; then
    echo "arm A2 (clean repro) failed"; tail -5 /tmp/gate_ds_a2.log; exit 1; fi
fi

TILERL_RUNS=$RUNS/b python -m tilerl.cli train $COMMON --eval-gsm8k "$EVAL" --eval-n 100 \
  > /tmp/gate_ds_b.log 2>&1
B_DIR=$(ls -td "$RUNS"/b/*/ | head -1)
if [ ! -f "$B_DIR/rollouts.jsonl" ] || [ "$(wc -l < "$B_DIR/rollouts.jsonl")" -ne 24 ]; then
  echo "arm B (prewarmed) failed"; tail -5 /tmp/gate_ds_b.log; exit 1; fi

echo "mode=$TILERL_POOL_RESET  A1=$A1_DIR  A2=$A2_DIR  B=$B_DIR"
python3 - "$A1_DIR" "${A2_DIR:-$A1_DIR}" "$B_DIR" <<'PY'
import json, sys
def load(p): return [json.loads(l) for l in open(p + "rollouts.jsonl")]
a1, a2, b = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
assert len(a1) == len(b) == 24, (len(a1), len(b))
if sys.argv[1] != sys.argv[2]:
    repro = sum(1 for x, y in zip(a1, a2) if x != y)
    print(f"repro A1-vs-A2 diff rows: {repro}/24")
gate = sum(1 for x, y in zip(a1, b) if x != y)
print(f"gate A1-vs-B diff rows: {gate}/24")
sys.exit(1 if gate else 0)
PY
rc=$?
if [ $rc -eq 0 ]; then echo "GATE GREEN (mode=$TILERL_POOL_RESET)"; else
  echo "GATE RED (mode=$TILERL_POOL_RESET)"; fi
exit $rc
