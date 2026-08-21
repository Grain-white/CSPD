#!/usr/bin/env bash
# Quick matched GSM8K run: BCE success critic for PPO/CSPD; CSPD uses K0 -> K reranking.
set -euo pipefail

METHOD=${1:-}
if [[ "${METHOD}" != "cspd" && "${METHOD}" != "ppo" ]]; then
  echo "Usage: $0 {cspd|ppo}" >&2
  exit 2
fi

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
SBATCH=/rmprog/slurm/v24.05.1/bin/sbatch
SRUN=/rmprog/slurm/v24.05.1/bin/srun
PARTITION=${PARTITION:-a01}
GPUS=${GPUS:-2}
STEPS=${TOTAL_TRAINING_STEPS:-20}
K=${CSPD_TOPK:-8}
K0=${CSPD_PROPOSAL_TOPK:-16}

if [[ "${METHOD}" == "cspd" ]]; then
  TAG=${TAG:-gsm8k-bce-twostage-k${K0}-quick${STEPS}-v1}
  JOB_NAME=cspd-gsm-bce-k${K0}
else
  TAG=${TAG:-gsm8k-bce-quick${STEPS}-v1}
  JOB_NAME=ppo-gsm-bce
fi

EXP_NAME="CSPD-${METHOD}-qwen3-1.7b-seed42-${TAG}"
mkdir -p "${ROOT}/output/logs"

"${SBATCH}" \
  --job-name="${JOB_NAME}" \
  --nodes=1 \
  --partition="${PARTITION}" \
  --exclude="${EXCLUDE_NODES:-g55}" \
  --time="${TIME_LIMIT:-6:00:00}" \
  --ntasks-per-node=1 \
  --gpus-per-node="${GPUS}" \
  --mem="${MEM:-230000}" \
  --cpus-per-task="${CPUS_PER_TASK:-16}" \
  --output="${ROOT}/output/logs/%j-${JOB_NAME}.log" \
  --error="${ROOT}/output/logs/%j-${JOB_NAME}.err" \
  --wrap="${SRUN} bash -lc '\
TRAIN_FILE=${ROOT}/data/gsm8k/train.parquet \
VAL_FILES=[${ROOT}/data/gsm8k/test.parquet] \
TRAIN_MAX_SAMPLES=7473 \
GPUS_PER_NODE=${GPUS} \
EXP_NAME=${EXP_NAME} \
TOTAL_TRAINING_STEPS=${STEPS} \
SAVE_FREQ=${SAVE_FREQ:-20} \
TEST_FREQ=${TEST_FREQ:-5} \
VAL_N=${VAL_N:-4} \
VAL_BEFORE_TRAIN=True \
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=1024 \
TRAIN_BATCH_SIZE=32 \
ROLLOUT_N=8 \
GPU_MEM_UTIL=0.55 \
CRITIC_LOSS_TYPE=bce \
CSPD_REWARD_RANGE=01 \
CSPD_TAIL_MODE=${CSPD_TAIL_MODE:-residual} \
CSPD_TOPK=${K} \
CSPD_PROPOSAL_TOPK=${K0} \
${ROOT}/scripts/training/run_math_ppo_cspd.sh ${METHOD}'"
