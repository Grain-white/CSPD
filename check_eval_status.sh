#!/usr/bin/env bash
set -u

JOB_ID="${1:-458712}"
ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
LOG="${ROOT}/output/logs/${JOB_ID}-eval-ppo-v3-retry.log"

echo '=== QUEUE ==='
squeue -j "${JOB_ID}" -o '%.18i %.12P %.24j %.2t %.10M %.4D %R' 2>&1 || true
echo '=== ACCOUNTING ==='
sacct -j "${JOB_ID}" --format=JobID,JobName%24,State,ExitCode,Elapsed,Start,End,NodeList -n -P 2>&1 || true
echo '=== LOG ==='
if [[ -f "${LOG}" ]]; then
  tail -n 240 "${LOG}"
else
  echo "LOG_NOT_CREATED: ${LOG}"
fi
