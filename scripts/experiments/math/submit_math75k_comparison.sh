#!/usr/bin/env bash
# MATH-train-7.5k paired PPO vs CSPD, hypers aligned with valueprobfix-diag-v5.
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
TAG=${TAG:-math75k-valueprobfix-v5}
PARTITION=${PARTITION:-a01}
TIME_LIMIT=${TIME_LIMIT:-2-00:00:00}
TRAIN_FILE=${TRAIN_FILE:-${ROOT}/data/math-train-7.5k.parquet}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-7500}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-5}
VAL_N=${VAL_N:-12}

mkdir -p "${ROOT}/output/logs"

for method in ppo cspd; do
  exp="CSPD-${method}-qwen3-1.7b-seed42-${TAG}"
  sbatch \
    --job-name="cspd-${method}-m75k" \
    --nodes=1 \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --ntasks-per-node=1 \
    --gpus-per-node=4 \
    --mem="${MEM:-460000}" \
    --cpus-per-task="${CPUS_PER_TASK:-32}" \
    --output="${ROOT}/output/logs/%j-${method}-math75k.log" \
    --error="${ROOT}/output/logs/%j-${method}-math75k.err" \
    --wrap="srun bash -lc '\
TRAIN_FILE=${TRAIN_FILE} \
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES} \
EXP_NAME=${exp} \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} \
SAVE_FREQ=${SAVE_FREQ} \
TEST_FREQ=${TEST_FREQ} \
VAL_N=${VAL_N} \
${ROOT}/scripts/training/run_math_ppo_cspd.sh ${method}'"
done
