#!/bin/bash
# Gate: training start is deterministic across before-arm cache states.
#
# Same sha, same seed, two 3-step GRPO runs: arm A skips the before arm (clean
# engine), arm B runs it (--eval-n 100, pre-warmed engine). Step 1-3 rollouts
# must be identical row for row. Without the pool reset in cli.py the two paths
# diverge from step 1 (measured 17/24 and 23/24 rows differing, 2026-09-09);
# with it, both converge to the same canonical pool state.
#
# Pod-only: needs the 27B and a GPU. Exit 0 green, 1 red.
set -uo pipefail
TREE=${TREE:-$(cd "$(dirname "$0")/.." && pwd)}
DATA=${DATA:-/work/p1_gsm8k_train.jsonl}
EVAL=${EVAL:-/work/p1_gsm8k_test.jsonl}
cd "$TREE"
# Fresh runs dir per gate invocation: the run id is a hash of the config, so without
# this the arm-A run id collides with any same-config run already in ./runs and the
# code refuses to rerun a finished run. A fresh dir also guarantees arm B's before-arm
# cache misses (a hit would skip the eval and make B the clean path again).
export TILERL_RUNS=$(mktemp -d /tmp/gate_ds_runs.XXXXXX)
COMMON="--recipe grpo-gsm8k-27b --data $DATA --steps 3 --eval-mmlu 0 --eval-every 0 \
  --length-penalty 0.0 --allow-short-rollouts"

python -m tilerl.cli train $COMMON > /tmp/gate_ds_a.log 2>&1 || {
  echo "arm A (clean) failed"; tail -5 /tmp/gate_ds_a.log; exit 1; }
A_DIR=$(ls -td "$TILERL_RUNS"/*/ | head -1)

python -m tilerl.cli train $COMMON --eval-gsm8k "$EVAL" --eval-n 100 > /tmp/gate_ds_b.log 2>&1 || {
  echo "arm B (prewarmed) failed"; tail -5 /tmp/gate_ds_b.log; exit 1; }
B_DIR=$(ls -td "$TILERL_RUNS"/*/ | head -1)

echo "A_DIR=$A_DIR"
echo "B_DIR=$B_DIR"
python3 - "$A_DIR" "$B_DIR" <<'PY'
import json, sys
a = [json.loads(l) for l in open(sys.argv[1] + "rollouts.jsonl")]
b = [json.loads(l) for l in open(sys.argv[2] + "rollouts.jsonl")]
assert len(a) == len(b) == 24, (len(a), len(b))
diff = sum(1 for ra, rb in zip(a, b) if ra != rb)
print(f"rollout diff rows: {diff}/{len(a)}")
sys.exit(1 if diff else 0)
PY
rc=$?
if [ $rc -eq 0 ]; then echo "GATE GREEN: clean and prewarmed start converge"; else
  echo "GATE RED: training start depends on before-arm cache state"; fi
exit $rc
