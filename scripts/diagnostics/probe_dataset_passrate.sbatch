#!/usr/bin/env bash
# One-GPU pass@1 probe. Usage:
#   salloc --partition=a01 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00
#   srun bash scripts/diagnostics/probe_dataset_passrate.sbatch
# Or: sbatch scripts/diagnostics/probe_dataset_passrate.sbatch
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=cspd-passrate-probe
#SBATCH --output=/home/fit/alex1/WORK/Meiqi.Gu/CSPD/output/probe_passrate-%j.out
#SBATCH --error=/home/fit/alex1/WORK/Meiqi.Gu/CSPD/output/probe_passrate-%j.err

set -euo pipefail
ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
PY=${PROBE_PYTHON:-/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full/bin/python}
export PYTHONPATH="$ROOT/sdpo_verl${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
mkdir -p "$ROOT/output"
cd "$ROOT"

N=${N:-128}
TEMP=${TEMP:-1.0}
MAX_NEW=${MAX_NEW:-2048}
BATCH=${BATCH:-4}

"$PY" probe_dataset_passrate.py \
  --model "${MODEL_PATH:-/WORK/PUBLIC/alex_work/Meiqi.Gu/models/Qwen3-1.7B}" \
  --n "$N" \
  --temperature "$TEMP" \
  --max-new "$MAX_NEW" \
  --batch-size "$BATCH" \
  --out "$ROOT/output/probe_passrate.json"
