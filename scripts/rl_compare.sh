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
GPU=${GPU:-7} STEPS=${STEPS:-20} GROUP=${GROUP:-8} LEN=${LEN:-256} MMLU=${MMLU:-200}
DATA=${DATA:-/work/gsm8k_train.jsonl}
[ -f "$DATA" ] || HF_ENDPOINT=https://hf-mirror.com python3 scripts/gsm8k_jsonl.py train "$DATA" --n 512

# A: tileRL
CUDA_VISIBLE_DEVICES=$GPU TILERL_QWEN38_SOURCE=$SRC python3 -m tilerl.cli train \
  --model qwen38-27b --rl --data "$DATA" --steps "$STEPS" --group "$GROUP" \
  --max-new-tokens "$LEN" --eval-mmlu "$MMLU" | tee /work/rl_compare_tilerl.log
python3 - <<'PY'
import re
secs = [float(m) for m in re.findall(r"  ([\d.]+)s$", open("/work/rl_compare_tilerl.log").read(), re.M)]
print(f"tilerl: {len(secs)} steps, median {sorted(secs)[len(secs)//2]:.1f}s/step, total {sum(secs):.0f}s")
PY

# B: verl GRPO, FSDP actor + sglang rollout. Same group/length/steps, one card.
# pending-remote — check the keys against the installed verl before trusting a number:
#   pip install verl && python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo \
#     data.train_files=/work/gsm8k_train.parquet data.train_batch_size=1 \
#     data.max_response_length=$LEN actor_rollout_ref.model.path=/work/Qwen3.8-27B-bf16 \
#     actor_rollout_ref.rollout.name=sglang actor_rollout_ref.rollout.n=$GROUP \
#     actor_rollout_ref.model.lora_rank=16 trainer.n_gpus_per_node=1 \
#     trainer.total_training_steps=$STEPS 2>&1 | tee /work/rl_compare_verl.log
