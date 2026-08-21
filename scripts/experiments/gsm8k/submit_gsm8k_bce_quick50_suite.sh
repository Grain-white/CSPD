#!/usr/bin/env bash
# Extend the matched GSM8K BCE runs from step 20 to 50 and launch no-tail to 50.
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
SUBMIT=${ROOT}/scripts/experiments/gsm8k/submit_gsm8k_bce_twostage_quick.sh
STEPS=50

# Keep the original experiment names so resume=auto loads global_step_20.
env \
  TOTAL_TRAINING_STEPS=${STEPS} SAVE_FREQ=50 TEST_FREQ=5 \
  TAG=gsm8k-bce-quick20-v1 JOB_NAME=ppo-gsm-bce-50 \
  SWANLAB_RUN_ID=c4o11mvg SWANLAB_RESUME=must \
  "${SUBMIT}" ppo

env \
  TOTAL_TRAINING_STEPS=${STEPS} SAVE_FREQ=50 TEST_FREQ=5 \
  TAG=gsm8k-bce-twostage-k16-quick20-v1 JOB_NAME=cspd-gsm-resid-50 \
  CSPD_TAIL_MODE=residual \
  SWANLAB_RUN_ID=lz9pgfqe SWANLAB_RESUME=must \
  "${SUBMIT}" cspd

env \
  TOTAL_TRAINING_STEPS=${STEPS} SAVE_FREQ=50 TEST_FREQ=5 \
  TAG=gsm8k-bce-baseline-tail-k16-quick20-v1 JOB_NAME=cspd-gsm-base-50 \
  CSPD_TAIL_MODE=baseline \
  SWANLAB_RUN_ID=4txpznt2 SWANLAB_RESUME=must \
  "${SUBMIT}" cspd

# No checkpoint exists for the full-size no-tail run, so train it from step 0.
env \
  TOTAL_TRAINING_STEPS=${STEPS} SAVE_FREQ=50 TEST_FREQ=5 \
  TAG=gsm8k-bce-no-tail-k16-quick50-v1 JOB_NAME=cspd-gsm-notail-50 \
  CSPD_TAIL_MODE=no_tail \
  "${SUBMIT}" cspd
