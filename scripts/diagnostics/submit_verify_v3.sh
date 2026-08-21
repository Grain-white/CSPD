#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
mkdir -p "${ROOT}/output/logs"
sbatch --job-name=cspd-verify-v3 --nodes=1 --partition=a01 \
  --time=02:00:00 --ntasks-per-node=1 --gpus-per-node=4 \
  --mem=460000 --cpus-per-task=32 \
  --output="${ROOT}/output/logs/%j-verify-v3.log" \
  --error="${ROOT}/output/logs/%j-verify-v3.err" \
  --wrap="srun bash -lc 'EXP_NAME=CSPD-verify-qwen3-1.7b-mean12-v3 TOTAL_TRAINING_STEPS=1 SAVE_FREQ=-1 TEST_FREQ=-1 ${ROOT}/scripts/training/run_math_ppo_cspd.sh ppo'"
