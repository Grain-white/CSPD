#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
MODEL=${ROOT}/output/CSPD-ppo-qwen3-1.7b-seed42-rewardfix-mean12-v2/checkpoints/global_step_300/actor/huggingface-merged
EXP=CSPD-eval-ppo-step300-fixedverify-mean12-v4-merged

if [[ ! -f "${MODEL}/model.safetensors" ]]; then
  echo "Merged PPO model is missing: ${MODEL}/model.safetensors" >&2
  exit 1
fi

mkdir -p "${ROOT}/output/logs"
sbatch \
  --job-name=eval-ppo-v4 \
  --partition=a01 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --cpus-per-task=32 \
  --mem=460000 \
  --time=02:00:00 \
  --output="${ROOT}/output/logs/%j-eval-ppo-v4.log" \
  --error="${ROOT}/output/logs/%j-eval-ppo-v4.err" \
  --wrap="srun bash -lc 'MODEL_PATH=${MODEL} EXP_NAME=${EXP} SAVE_FREQ=-1 TEST_FREQ=-1 ${ROOT}/run_math_ppo_cspd.sh ppo trainer.resume_mode=disable trainer.val_only=True algorithm.adv_estimator=grpo critic.enable=False'"
