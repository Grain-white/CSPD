#!/usr/bin/env bash
# Quick 50-step GSM8K PPO/CSPD critic check with fixed verifier
# (#### OR \\boxed{} fallback) + strong #### instruction in parquet.
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
PARTITION=${PARTITION:-a01}
TIME_LIMIT=${TIME_LIMIT:-4:00:00}
TAG=${TAG:-gsm8k-verfix-criticcheck-50}
GPUS=${GPUS:-2}

mkdir -p "${ROOT}/output/logs"

submit_one() {
  local method="$1"
  local exp="CSPD-${method}-qwen3-1.7b-seed42-${TAG}"
  sbatch \
    --job-name="cspd-${method}-gsm8k50" \
    --nodes=1 \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS}" \
    --mem="${MEM:-230000}" \
    --cpus-per-task="${CPUS_PER_TASK:-16}" \
    --output="${ROOT}/output/logs/%j-${method}-gsm8k50.log" \
    --error="${ROOT}/output/logs/%j-${method}-gsm8k50.err" \
    --wrap="srun bash -lc '\
TRAIN_FILE=${ROOT}/data/gsm8k/train.parquet \
VAL_FILES=[${ROOT}/data/gsm8k/test.parquet] \
TRAIN_MAX_SAMPLES=7473 \
GPUS_PER_NODE=${GPUS} \
EXP_NAME=${exp} \
TOTAL_TRAINING_STEPS=50 \
SAVE_FREQ=-1 \
TEST_FREQ=10 \
VAL_N=4 \
VAL_BEFORE_TRAIN=True \
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=1024 \
TRAIN_BATCH_SIZE=32 \
ROLLOUT_N=8 \
GPU_MEM_UTIL=0.55 \
${ROOT}/scripts/training/run_math_ppo_cspd.sh ${method}'"
}

# sanity: verifier accepts boxed
PYTHONPATH=${ROOT}/sdpo_verl:${PYTHONPATH:-} \
  /WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full/bin/python - <<'PY'
from verl.utils.reward_score import gsm8k as g
assert g.compute_score("blah\n\\boxed{72}", "72") == 1.0
assert g.compute_score("....\n#### 72", "72") == 1.0
assert g.compute_score("\\boxed{1}", "72") == 0.0
print("verifier ok: #### and boxed")
PY

submit_one ppo
submit_one cspd
