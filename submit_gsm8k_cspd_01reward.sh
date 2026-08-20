#!/usr/bin/env bash
# GSM8K CSPD critic-check with cspd_reward_range=01 (verifier is 0/1, not ±1).
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
PARTITION=${PARTITION:-a01}
TIME_LIMIT=${TIME_LIMIT:-4:00:00}
TAG=${TAG:-gsm8k-01reward-criticcheck-50}
GPUS=${GPUS:-2}

mkdir -p "${ROOT}/output/logs"

EXP_NAME="CSPD-cspd-qwen3-1.7b-seed42-${TAG}"
sbatch \
  --job-name="cspd-cspd-gsm8k01" \
  --nodes=1 \
  --partition="${PARTITION}" \
  --time="${TIME_LIMIT}" \
  --ntasks-per-node=1 \
  --gpus-per-node="${GPUS}" \
  --mem="${MEM:-230000}" \
  --cpus-per-task="${CPUS_PER_TASK:-16}" \
  --output="${ROOT}/output/logs/%j-cspd-gsm8k01.log" \
  --error="${ROOT}/output/logs/%j-cspd-gsm8k01.err" \
  --wrap="srun bash -lc '\
TRAIN_FILE=${ROOT}/data/gsm8k/train.parquet \
VAL_FILES=[${ROOT}/data/gsm8k/test.parquet] \
TRAIN_MAX_SAMPLES=7473 \
GPUS_PER_NODE=${GPUS} \
EXP_NAME=${EXP_NAME} \
TOTAL_TRAINING_STEPS=50 \
SAVE_FREQ=-1 \
TEST_FREQ=10 \
VAL_N=4 \
VAL_BEFORE_TRAIN=True \
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=1024 \
TRAIN_BATCH_SIZE=32 \
ROLLOUT_N=8 \
GPU_MEM_UTIL=0.55 \
CSPD_REWARD_RANGE=01 \
${ROOT}/run_math_ppo_cspd.sh cspd'"
