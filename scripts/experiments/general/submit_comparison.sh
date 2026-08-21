#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
TAG=${TAG:-rewardfix-mean12-v2}
mkdir -p "${ROOT}/output/logs"
for method in ppo cspd; do
  sbatch --job-name="cspd-${method}" --nodes=1 --partition="${PARTITION:-a01}" \
    --time="${TIME_LIMIT:-16-00:00:00}" --ntasks-per-node=1 --gpus-per-node=4 \
    --mem="${MEM:-460000}" --cpus-per-task="${CPUS_PER_TASK:-32}" \
    --output="${ROOT}/output/logs/%j-${method}.log" --error="${ROOT}/output/logs/%j-${method}.err" \
    --wrap="srun bash -lc 'EXP_NAME=CSPD-${method}-qwen3-1.7b-seed42-${TAG} ${ROOT}/scripts/training/run_math_ppo_cspd.sh ${method}'"
done
