#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
COMMON=(
  TRAIN_FILE=${ROOT}/data/gsm8k/train.parquet
  VAL_FILES=[${ROOT}/data/gsm8k/test.parquet]
  TRAIN_MAX_SAMPLES=8
  GPUS_PER_NODE=1
  TOTAL_TRAINING_STEPS=1
  SAVE_FREQ=-1
  TEST_FREQ=-1
  VAL_BEFORE_TRAIN=False
  MAX_PROMPT_LENGTH=512
  MAX_RESPONSE_LENGTH=256
  TRAIN_BATCH_SIZE=8
  ROLLOUT_N=2
  MAX_NUM_SEQS=16
  GPU_MEM_UTIL=0.35
  CRITIC_LOSS_TYPE=bce
  CSPD_REWARD_RANGE=01
  CSPD_TAIL_MODE=${CSPD_TAIL_MODE:-residual}
  DISABLE_SWANLAB=1
)

cd "${ROOT}"
env "${COMMON[@]}" CSPD_TOPK=8 CSPD_PROPOSAL_TOPK=16 CSPD_PREFIXES=2 \
  EXP_NAME=CSPD-cspd-qwen3-1.7b-seed42-debug-bce-twostage \
  "${ROOT}/scripts/training/run_math_ppo_cspd.sh" cspd

env "${COMMON[@]}" \
  EXP_NAME=CSPD-ppo-qwen3-1.7b-seed42-debug-bce \
  "${ROOT}/scripts/training/run_math_ppo_cspd.sh" ppo
