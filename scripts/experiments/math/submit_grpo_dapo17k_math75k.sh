#!/usr/bin/env bash
# Comparable GRPO baselines for DAPO-Math-17k and MATH-train-7.5k.
# Hypers aligned with valueprobfix-diag-v5 / math75k PPO-CSPD comparison.
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
PARTITION=${PARTITION:-a01}
TIME_LIMIT=${TIME_LIMIT:-2-00:00:00}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-5}
VAL_N=${VAL_N:-12}

mkdir -p "${ROOT}/output/logs"

submit_one() {
  local tag="$1"
  local train_file="$2"
  local train_max="$3"
  local job_suffix="$4"
  local exp="CSPD-grpo-qwen3-1.7b-seed42-${tag}"

  sbatch \
    --job-name="cspd-grpo-${job_suffix}" \
    --nodes=1 \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --ntasks-per-node=1 \
    --gpus-per-node=4 \
    --mem="${MEM:-460000}" \
    --cpus-per-task="${CPUS_PER_TASK:-32}" \
    --output="${ROOT}/output/logs/%j-grpo-${job_suffix}.log" \
    --error="${ROOT}/output/logs/%j-grpo-${job_suffix}.err" \
    --wrap="srun bash -lc '\
TRAIN_FILE=${train_file} \
TRAIN_MAX_SAMPLES=${train_max} \
EXP_NAME=${exp} \
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} \
SAVE_FREQ=${SAVE_FREQ} \
TEST_FREQ=${TEST_FREQ} \
VAL_N=${VAL_N} \
${ROOT}/scripts/training/run_math_ppo_cspd.sh grpo'"
}

# Interpret "dapo-math1.7k" as the existing DAPO-Math-17k budget.
submit_one \
  "dapo17k-grpo-v5" \
  "${ROOT}/data/dapo-math-17k-seed42.parquet" \
  17000 \
  "dapo17k"

submit_one \
  "math75k-grpo-v5" \
  "${ROOT}/data/math-train-7.5k.parquet" \
  7500 \
  "m75k"
