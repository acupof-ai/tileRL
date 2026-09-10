#!/usr/bin/env bash
# Same pod, same model, same task: seconds per RL step (rollout + update) and
# MMLU before/after, tileRL vs verl+sglang. Run from the synced checkout on the
# pod (scripts/pod_sync.sh run rl_compare 'bash scripts/rl_compare.sh').
#
# Arm A: tileRL, one process, LoRA on the frozen fp4 base. Arm B: verl GRPO, pending-remote —
# verl is not installed on the pod, and sglang cannot load NVFP4 on Hopper, so B runs the
# bf16 conversion (docs/experience/errors/2026-08-28-sglang-bf16-checkpoint-garbage.md).
set -euo pipefail
SRC=${TILERL_QWEN38_SOURCE:-/work/Qwen3.8-27B-NVFP4}
# Card 0 is tileRL's grant; the previous default 7 belongs to aupai. Override with GPU=.
GPU=${GPU:-0} STEPS=${STEPS:-20} GROUP=${GROUP:-8} LEN=${LEN:-256} MMLU=${MMLU:-200}
DATA=${DATA:-/work/gsm8k_train.jsonl}
RUNS=${TILERL_RUNS:-runs}
[ -f "$DATA" ] || HF_ENDPOINT=https://hf-mirror.com python3 scripts/gsm8k_jsonl.py train "$DATA" --n 512

# A: tileRL
CUDA_VISIBLE_DEVICES=$GPU TILERL_QWEN38_SOURCE=$SRC python3 -m tilerl.cli train \
  --model qwen38-27b --rl --data "$DATA" --steps "$STEPS" --group "$GROUP" \
  --max-new-tokens "$LEN" --eval-mmlu "$MMLU" | tee /work/rl_compare_tilerl.log
# Read the manifest, not the log: the step line gained phase timings and no longer
# ends in "<n>s", so the old trailing-number regex matched nothing and reported a
# crash-length median of zero. Bind THIS run by its id (the last tee'd line is the
# ledger summary, whose first field is the run id), not "newest manifest" -- the
# script runs on a shared card where another run may finish in the same window.
python3 - "$RUNS" <<'PY'
import json, pathlib, sys
runs_dir, log_path = sys.argv[1], "/work/rl_compare_tilerl.log"
rid = pathlib.Path(log_path).read_text().strip().splitlines()[-1].split()[0]
m = json.loads((pathlib.Path(runs_dir) / rid / "manifest.json").read_text())
g = m["metrics"]
print(f"tilerl run {m['id']}: {g.get('steps_completed')} steps, "
      f"median {g['secs_per_step_median']:.1f}s/step, total {g['secs_total']:.0f}s")
PY

# B: verl GRPO, FSDP actor + sglang rollout. Same group/length/steps, one card.
# pending-remote — check the keys against the installed verl before trusting a number:
#   pip install verl && python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo \
#     data.train_files=/work/gsm8k_train.parquet data.train_batch_size=1 \
#     data.max_response_length=$LEN actor_rollout_ref.model.path=/work/Qwen3.8-27B-bf16 \
#     actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.n=$GROUP \
#     actor_rollout_ref.model.lora_rank=16 trainer.n_gpus_per_node=1 \
#     trainer.total_training_steps=$STEPS 2>&1 | tee /work/rl_compare_verl.log
