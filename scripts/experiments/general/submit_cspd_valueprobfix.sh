#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
EXP_NAME=CSPD-cspd-qwen3-1.7b-seed42-valueprobfix-diag-v5
mkdir -p "${ROOT}/output/logs"

sbatch \
  --job-name=cspd-vdiag \
  --nodes=1 \
  --partition=a01 \
  --time=2-00:00:00 \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --mem=460000 \
  --cpus-per-task=32 \
  --output="${ROOT}/output/logs/%j-cspd-valueprobfix-diag.log" \
  --error="${ROOT}/output/logs/%j-cspd-valueprobfix-diag.err" \
  --wrap="srun bash -lc 'EXP_NAME=${EXP_NAME} TOTAL_TRAINING_STEPS=300 SAVE_FREQ=50 TEST_FREQ=5 VAL_N=12 ${ROOT}/scripts/training/run_math_ppo_cspd.sh cspd'"
