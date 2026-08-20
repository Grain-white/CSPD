#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:-}"
shift || true
if [[ "${METHOD}" != "ppo" && "${METHOD}" != "cspd" && "${METHOD}" != "grpo" ]]; then
  echo "Usage: $0 {ppo|cspd|grpo} [Hydra overrides...]" >&2
  exit 2
fi

ROOT=/home/fit/alex1/WORK/Meiqi.Gu/CSPD
VERL=${ROOT}/sdpo_verl
ENV_PATH=/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full
MODEL_PATH=${MODEL_PATH:-/WORK/PUBLIC/alex_work/Meiqi.Gu/models/Qwen3-1.7B}
DATA_ROOT=/WORK/PUBLIC/alex_work/Meiqi.Gu/SDPO/datasets/dapo_hf/processed
TRAIN_FILE=${TRAIN_FILE:-${ROOT}/data/dapo-math-17k-seed42.parquet}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-17000}
VAL_FILES=${VAL_FILES:-"[${DATA_ROOT}/math-500.eval.parquet,${DATA_ROOT}/aime-2024.eval.parquet,${DATA_ROOT}/aime-2025.eval.parquet,/home/fit/alex1/WORK/Meiqi.Gu/SDPO/datasets/openthoughts_math/processed/hmmt-2025.eval.parquet]"}
GPUS=${GPUS_PER_NODE:-4}
STEPS=${TOTAL_TRAINING_STEPS:-300}
TRAIN_BATCH=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-12}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_TOP_K=${VAL_TOP_K:-20}
PPO_EPOCHS=${PPO_EPOCHS:-1}
MAX_PROMPT=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=$((MAX_PROMPT + MAX_RESPONSE))
EXP_NAME=${EXP_NAME:-CSPD-${METHOD}-qwen3-1.7b-seed42}
OUT=${ROOT}/output/${EXP_NAME}
TRAIN_LOG=${TRAIN_LOG:-${OUT}/train.log}

export PATH=${ENV_PATH}/bin:${PATH}
hash -r
export PYTHONNOUSERSITE=1
export PYTHONPATH=${VERL}:${ROOT}:${PYTHONPATH:-}
export RAY_ADDRESS=local
export RAY_num_prestart_python_workers=0
export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID:-$$}
export TMPDIR=${RAY_TMPDIR} TEMP=${RAY_TMPDIR} TMP=${RAY_TMPDIR}
export VLLM_CACHE_ROOT=${RAY_TMPDIR}/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=${RAY_TMPDIR}/torchinductor
export TRITON_CACHE_DIR=${RAY_TMPDIR}/triton
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=${ROOT}/output/hf_home
export HF_DATASETS_CACHE=${ROOT}/output/hf_datasets
export HUGGINGFACE_HUB_CACHE=${ROOT}/output/hf_hub
export SWANLAB_MODE=${SWANLAB_MODE:-online}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${ROOT}/swanlog}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
mkdir -p "${RAY_TMPDIR}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${OUT}" "${ROOT}/swanlog"

ENV_FILE=/home/fit/alex1/WORK/Meiqi.Gu/.env
[[ -f "${ENV_FILE}" ]] || ENV_FILE=/WORK/PUBLIC/alex_work/Meiqi.Gu/.env
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
LOGGER='["console","swanlab"]'
[[ -n "${SWANLAB_API_KEY:-}" ]] || LOGGER='["console"]'
[[ "${DISABLE_SWANLAB:-0}" == "1" ]] && LOGGER='["console"]'

LOSS_MODE=vanilla
[[ "${METHOD}" == cspd ]] && LOSS_MODE=cspd
ADV_ESTIMATOR=gae
CRITIC_ENABLE=True
if [[ "${METHOD}" == grpo ]]; then
  ADV_ESTIMATOR=grpo
  CRITIC_ENABLE=False
fi

cd "${ROOT}"
python -m verl.trainer.main_ppo \
  ++ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-8}}" \
  ++ray_kwargs.ray_init.include_dashboard=False \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.use_kl_in_reward=False \
  data.train_files="[${TRAIN_FILE}]" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.max_prompt_length="${MAX_PROMPT}" \
  data.max_response_length="${MAX_RESPONSE}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  data.seed=42 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
  +actor_rollout_ref.actor.policy_loss.cspd_topk="${CSPD_TOPK:-8}" \
  +actor_rollout_ref.actor.policy_loss.cspd_prefixes_per_response="${CSPD_PREFIXES:-8}" \
  actor_rollout_ref.actor.policy_loss.cspd_reward_range="${CSPD_REWARD_RANGE:-pm1}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.45}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-256}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_WORKERS:-8}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K}" \
  critic.enable="${CRITIC_ENABLE}" \
  critic.strategy=fsdp \
  critic.model.path="${MODEL_PATH}" \
  critic.model.use_remove_padding=True \
  critic.model.enable_gradient_checkpointing=True \
  critic.optim.lr=1e-5 \
  critic.ppo_mini_batch_size="${TRAIN_BATCH}" \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_epochs="${PPO_EPOCHS}" \
  critic.use_dynamic_bsz=True \
  critic.ppo_max_token_len_per_gpu="$((MAX_MODEL_LEN * 2))" \
  critic.model.fsdp_config.model_dtype=bfloat16 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  trainer.use_legacy_worker_impl=enable \
  trainer.critic_warmup=0 \
  trainer.logger="${LOGGER}" \
  trainer.project_name=CSPD-PPO-comparison \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${GPUS}" \
  trainer.nnodes=1 \
  trainer.total_training_steps="${STEPS}" \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.test_freq="${TEST_FREQ:-5}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.log_val_generations=16 \
  trainer.default_local_dir="${OUT}/checkpoints" \
  "$@" 2>&1 | tee "${TRAIN_LOG}"
