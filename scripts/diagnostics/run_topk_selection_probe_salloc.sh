#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
ENV_PATH=/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full
MODEL=/WORK/PUBLIC/alex_work/Meiqi.Gu/models/Qwen3-1.7B
OUT=${ROOT}/output/topk-selection-comparison

export PATH=${ENV_PATH}/bin:${PATH}
export PYTHONNOUSERSITE=1
export PYTHONPATH=${ROOT}/sdpo_verl:${ROOT}/analysis:${ROOT}:${PYTHONPATH:-}
export HF_HOME=${ROOT}/output/hf_home
export HF_DATASETS_CACHE=${ROOT}/output/hf_datasets
export HUGGINGFACE_HUB_CACHE=${ROOT}/output/hf_hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
mkdir -p "${OUT}" "${ROOT}/output/logs"

cd "${ROOT}"

MATH_STEP_LIST=${MATH_STEPS:-"50 100 150 200 250 300"}
DAPO_STEP_LIST=${DAPO_STEPS:-"50 100 150 200 250"}

for step in ${MATH_STEP_LIST}; do
  python analysis/topk_selection_probe.py \
    --root "${ROOT}" \
    --step "${step}" \
    --actor-method cspd \
    --critic-method cspd \
    --tag math75k-valueprobfix-v5 \
    --dataset-label math75k \
    --panel "${ROOT}/output/gt-posterior-math75k-pilot/fixed_base_entropy_panel.json" \
    --output-dir "${OUT}" \
    --model "${MODEL}" \
    --retain-k 8 \
    --proposal-k0 16 32 64
  python analysis/aggregate_topk_selection_probe.py "${OUT}"
done

for step in ${DAPO_STEP_LIST}; do
  python analysis/topk_selection_probe.py \
    --root "${ROOT}" \
    --step "${step}" \
    --actor-method cspd \
    --critic-method cspd \
    --tag valueprobfix-diag-v5 \
    --dataset-label dapo17k \
    --panel "${ROOT}/output/gt-posterior-dapo17k-pilot/fixed_base_entropy_panel.json" \
    --output-dir "${OUT}" \
    --model "${MODEL}" \
    --retain-k 8 \
    --proposal-k0 16 32 64
  python analysis/aggregate_topk_selection_probe.py "${OUT}"
done

for step in ${DAPO_STEP_LIST}; do
  python analysis/topk_selection_probe.py \
    --root "${ROOT}" \
    --step "${step}" \
    --actor-method cspd \
    --critic-method cspd \
    --tag valueprobfix-diag-v5 \
    --dataset-label dapo17k_success \
    --panel "${ROOT}/output/gt-posterior-dapo17k-success/panel.json" \
    --output-dir "${OUT}" \
    --model "${MODEL}" \
    --retain-k 8 \
    --proposal-k0 16 32 64
  python analysis/aggregate_topk_selection_probe.py "${OUT}"
done

python analysis/aggregate_topk_selection_probe.py "${OUT}"
