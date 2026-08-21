#!/usr/bin/env bash
# GSM8K CSPD tail ablation: condition the success posterior on retained top-K.
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
STEPS=${TOTAL_TRAINING_STEPS:-20}
K0=${CSPD_PROPOSAL_TOPK:-16}

env \
  CSPD_TAIL_MODE=no_tail \
  TAG=${TAG:-gsm8k-bce-no-tail-k${K0}-quick${STEPS}-v1} \
  JOB_NAME=${JOB_NAME:-cspd-gsm-notail-k${K0}} \
  TOTAL_TRAINING_STEPS=${STEPS} \
  "${ROOT}/scripts/experiments/gsm8k/submit_gsm8k_bce_twostage_quick.sh" cspd
