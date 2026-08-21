#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
ENV_PATH=/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full
MODEL=/WORK/PUBLIC/alex_work/Meiqi.Gu/models/Qwen3-1.7B

export PATH=${ENV_PATH}/bin:${PATH}
export PYTHONNOUSERSITE=1
export PYTHONPATH=${ROOT}/sdpo_verl:${ROOT}:${PYTHONPATH:-}
export HF_HOME=${ROOT}/output/hf_home
export HF_DATASETS_CACHE=${ROOT}/output/hf_datasets
export HUGGINGFACE_HUB_CACHE=${ROOT}/output/hf_hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
cd "${ROOT}"

MATH_OUT=${ROOT}/output/gt-posterior-math75k-pilot
MATH_STEP_LIST=${MATH_STEPS:-"50 100 150 200 250 300"}
for step in ${MATH_STEP_LIST}; do
  python analysis/gt_posterior_probe.py \
    --root "${ROOT}" --step "${step}" --actor-method cspd --critic-methods cspd ppo \
    --tag math75k-valueprobfix-v5 --dataset-label math75k \
    --panel "${MATH_OUT}/fixed_base_entropy_panel.json" --output-dir "${MATH_OUT}" \
    --model "${MODEL}" --data "${ROOT}/data/math-train-7.5k.parquet" --reuse-actor-records
done
python analysis/aggregate_gt_posterior_probe.py "${MATH_OUT}"

DAPO_OUT=${ROOT}/output/gt-posterior-dapo17k-pilot
DAPO_STEP_LIST=${DAPO_STEPS:-"50 100 150 200 250"}
for step in ${DAPO_STEP_LIST}; do
  python analysis/gt_posterior_probe.py \
    --root "${ROOT}" --step "${step}" --actor-method cspd --critic-methods cspd \
    --tag valueprobfix-diag-v5 --dataset-label dapo17k \
    --panel "${DAPO_OUT}/fixed_base_entropy_panel.json" --output-dir "${DAPO_OUT}" \
    --model "${MODEL}" --data "${ROOT}/data/dapo-math-17k-seed42.parquet" --reuse-actor-records
done
python analysis/aggregate_gt_posterior_probe.py "${DAPO_OUT}"

DAPO_SUCCESS_OUT=${ROOT}/output/gt-posterior-dapo17k-success
for step in ${DAPO_STEP_LIST}; do
  python analysis/gt_posterior_probe.py \
    --root "${ROOT}" --step "${step}" --actor-method cspd --critic-methods cspd \
    --tag valueprobfix-diag-v5 --dataset-label dapo17k_success \
    --panel "${DAPO_SUCCESS_OUT}/panel.json" --output-dir "${DAPO_SUCCESS_OUT}" \
    --model "${MODEL}" --data "${ROOT}/data/dapo-math-17k-seed42.parquet" --reuse-actor-records
done
python analysis/aggregate_gt_posterior_probe.py "${DAPO_SUCCESS_OUT}"
