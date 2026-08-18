#!/usr/bin/env bash
set -u

for job in "$@"; do
  echo "========== JOB ${job} =========="
  scontrol show job -dd "${job}" 2>&1 || true
  echo '--- accounting ---'
  sacct -j "${job}" --format=JobID,JobName%24,State,Submit,Eligible,Start,Elapsed,Timelimit,ReqCPUS,ReqMem,AllocTRES%80,NodeList -n -P 2>&1 || true
done
