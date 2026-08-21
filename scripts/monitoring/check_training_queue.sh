#!/usr/bin/env bash
set -u

JOBS=("${@:-461081 461082}")

echo '=== USER QUEUE ==='
squeue -u alex1 -o '%.18i %.12P %.24j %.2t %.10M %.4D %R'
echo '=== TARGET START ESTIMATES ==='
for job in ${JOBS[*]}; do
  echo "--- ${job} ---"
  squeue --start -j "${job}" -o '%.18i %.2t %.19S %R' 2>&1 || true
  sprio -j "${job}" 2>&1 || true
done
echo '=== A01 NODES ==='
sinfo -p a01 -N -o '%N %t %G %C' 2>&1 || true
