#!/usr/bin/env bash
# Extend the matched GSM8K BCE runs from step 20 to 150 and launch no-tail and
# a standard-MSE-critic PPO control to step 150.
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
SUBMIT=${ROOT}/scripts/experiments/gsm8k/submit_gsm8k_bce_twostage_quick.sh
STEPS=150

# Keep the original experiment names so resume=auto loads global_step_20.
env \
  TOTAL_TRAINING_STEPS=${STEPS} SAVE_FREQ=50 TEST_FREQ=5 \
  TAG=gsm8k-bce-quick20-v1 JOB_NAME=ppo-gsm-bce-150 \
  SWANLAB_RUN_ID=c4o11mvg SWANLAB_RESUME=must \
  "${SUBMIT}" ppo

env \
  TOTAL_TRAINING_STEPS=${STEPS} SAVE_FREQ=50 TEST_FREQ=5 \
  TAG=gsm8k-bce-twostage-k16-quick20-v1 JOB_NAME=cspd-gsm-resid-150 \
  CSPD_TAIL_MODE=residual \
  SWANLAB_RUN_ID=lz9pgfqe SWANLAB_RESUME=must \
  "${SUBMIT}" cspd

env \
  TOTAL_TRAINING_STEPS=${STEPS} SAVE_FREQ=50 TEST_FREQ=5 \
  TAG=gsm8k-bce-baseline-tail-k16-quick20-v1 JOB_NAME=cspd-gsm-base-150 \
  CSPD_TAIL_MODE=baseline \
  SWANLAB_RUN_ID=4txpznt2 SWANLAB_RESUME=must \
  "${SUBMIT}" cspd

# No checkpoint exists for the full-size no-tail run, so train it from step 0.
env \
  TOTAL_TRAINING_STEPS=${STEPS} SAVE_FREQ=50 TEST_FREQ=5 \
  TAG=gsm8k-bce-no-tail-k16-quick150-v1 JOB_NAME=cspd-gsm-notail-150 \
  CSPD_TAIL_MODE=no_tail \
  "${SUBMIT}" cspd

# Standard PPO control: same data/rollout/update budget, original MSE critic.
env \
  TOTAL_TRAINING_STEPS=${STEPS} SAVE_FREQ=50 TEST_FREQ=5 \
  TAG=gsm8k-ppo-original-mse-150-v1 JOB_NAME=ppo-gsm-mse-150 \
  CRITIC_LOSS_TYPE=mse \
  "${SUBMIT}" ppo
