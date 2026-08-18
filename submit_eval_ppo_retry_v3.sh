#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
CKPT=${ROOT}/output/CSPD-ppo-qwen3-1.7b-seed42-rewardfix-mean12-v2/checkpoints/global_step_300/actor/huggingface
EXP=CSPD-eval-ppo-step300-fixedverify-mean12-v3-retry2

if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "Checkpoint is missing or incomplete: ${CKPT}" >&2
  exit 1
fi

sbatch \
  --job-name=eval-ppo-v3r \
  --partition=a01 \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:4 \
  --cpus-per-task=32 \
  --time=02:00:00 \
  --output=${ROOT}/output/logs/%j-eval-ppo-v3-retry.log \
  --wrap="srun bash -lc 'DISABLE_SWANLAB=1 TRAIN_LOG=/tmp/eval-ppo-v3-retry.log MODEL_PATH=${CKPT} EXP_NAME=${EXP} SAVE_FREQ=-1 TEST_FREQ=-1 ${ROOT}/run_math_ppo_cspd.sh ppo trainer.resume_mode=disable trainer.val_only=True'"
