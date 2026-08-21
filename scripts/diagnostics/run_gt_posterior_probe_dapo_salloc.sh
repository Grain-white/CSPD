#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
ENV_PATH=/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full
MODEL=/WORK/PUBLIC/alex_work/Meiqi.Gu/models/Qwen3-1.7B
DATA=${ROOT}/data/dapo-math-17k-seed42.parquet
OUT=${ROOT}/output/gt-posterior-dapo17k-pilot
PANEL=${OUT}/fixed_base_entropy_panel.json
STEPS=${STEPS:-"50 100 150 200 250"}

export PATH=${ENV_PATH}/bin:${PATH}
export PYTHONNOUSERSITE=1
export PYTHONPATH=${ROOT}/sdpo_verl:${ROOT}:${PYTHONPATH:-}
export HF_HOME=${ROOT}/output/hf_home
export HF_DATASETS_CACHE=${ROOT}/output/hf_datasets
export HUGGINGFACE_HUB_CACHE=${ROOT}/output/hf_hub
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
mkdir -p "${OUT}" "${ROOT}/output/logs"

cd "${ROOT}"
for step in ${STEPS}; do
  python analysis/gt_posterior_probe.py \
    --root "${ROOT}" \
    --step "${step}" \
    --actor-method cspd \
    --critic-methods cspd \
    --tag valueprobfix-diag-v5 \
    --dataset-label dapo17k \
    --panel "${PANEL}" \
    --output-dir "${OUT}" \
    --model "${MODEL}" \
    --data "${DATA}" \
    --topk "${TOPK:-8}" \
    --mc-samples "${MC_SAMPLES:-32}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
    --num-prompts "${NUM_PROMPTS:-3}" \
    --prefixes-per-prompt "${PREFIXES_PER_PROMPT:-1}"
  python analysis/aggregate_gt_posterior_probe.py "${OUT}"
done

python analysis/aggregate_gt_posterior_probe.py "${OUT}"
