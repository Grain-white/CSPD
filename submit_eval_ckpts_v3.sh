#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
mkdir -p "${ROOT}/output/logs"

for method in ppo cspd; do
  ckpt="${ROOT}/output/CSPD-${method}-qwen3-1.7b-seed42-rewardfix-mean12-v2/checkpoints/global_step_300"
  [[ -d "${ckpt}/actor" ]] || { echo "Missing actor checkpoint: ${ckpt}" >&2; exit 1; }
  exp="CSPD-eval-${method}-step300-fixedverify-mean12-v3"
  sbatch --job-name="eval-${method}-v3" --nodes=1 --partition=a01 \
    --time=02:00:00 --ntasks-per-node=1 --gpus-per-node=4 \
    --mem=460000 --cpus-per-task=32 \
    --output="${ROOT}/output/logs/%j-eval-${method}-v3.log" \
    --error="${ROOT}/output/logs/%j-eval-${method}-v3.err" \
    --wrap="srun bash -lc 'EXP_NAME=${exp} SAVE_FREQ=-1 TEST_FREQ=-1 ${ROOT}/run_math_ppo_cspd.sh ${method} trainer.resume_mode=resume_path trainer.resume_from_path=${ckpt} trainer.val_only=True'"
done
