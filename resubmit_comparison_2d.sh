#!/usr/bin/env bash
# LEGACY: this script contains hard-coded job IDs and may cancel those jobs.
# Prefer submit_comparison.sh. Review OLD_JOBS and all overrides before reuse.
set -euo pipefail

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
OLD_JOBS=(461081 461082)

for job in "${OLD_JOBS[@]}"; do
  state=$(squeue -h -j "${job}" -o '%T' || true)
  if [[ "${state}" == "PENDING" ]]; then
    scancel "${job}"
    echo "Cancelled pending job ${job}"
  elif [[ -n "${state}" ]]; then
    echo "Refusing to cancel job ${job}: current state is ${state}" >&2
    exit 1
  else
    echo "Job ${job} is no longer in the queue; not cancelled"
  fi
done

export TAG=fixedverify-retrain-v3
export TIME_LIMIT=2-00:00:00
export TOTAL_TRAINING_STEPS=300
export SAVE_FREQ=300
export TEST_FREQ=50
export VAL_N=12
bash "${ROOT}/submit_comparison.sh"
