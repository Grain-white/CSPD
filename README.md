# CSPD

Critic-Induced Success-Posterior Distillation (CSPD) implemented on a local
SDPO/verl snapshot for a controlled comparison with PPO/GAE.

The canonical project directory on the cluster is:

```text
/home/fit/alex1/WORK/Meiqi.Gu/CSPD
```

Only files below this directory are written by the project. The shared model,
dataset, environment, and reference repositories are read-only inputs.

## Current algorithm

The verifier emits a terminal reward in `{-1, +1}`. Consequently, the PPO
critic predicts signed expected return rather than a probability. Before the
success posterior is constructed, state and successor values are converted by

```math
P(\text{success}\mid s)=\operatorname{clip}\left(\frac{V_{\pm}(s)+1}{2},0,1\right).
```

At each rollout iteration, the current CSPD implementation:

1. freezes the behavior actor and pre-update critic;
2. selects at most eight high-entropy response prefixes per response;
3. takes eight behavior-policy top-probability next-token candidates;
4. evaluates `V(s + a)` for every selected candidate;
5. builds a projected top-K plus tail success posterior; and
6. minimizes its frozen value-weighted forward cross-entropy, which has the
   same actor gradient as forward KL because target entropy is constant.

The implementation is a practical approximation:

- it uses behavior top-K directly (`K0 = K`) rather than proposing `K0 > K`
  and reranking by posterior mass;
- high-entropy prefix selection changes the state distribution;
- tail projection and renormalization can break the exact mass identity;
- successor critic calls currently rebuild full prefixes instead of reusing a
  critic KV cache; and
- PPO whitens GAE while CSPD currently leaves its loss unnormalized.

See [CONTRASTIVE_CSPD_DERIVATION.md](CONTRASTIVE_CSPD_DERIVATION.md) for the
success/failure posterior, Signed-KL, proximal interpretation, and bounded
pairwise contrastive CSPD derivations.

## Canonical code layout

| Path | Purpose |
| --- | --- |
| `sdpo_verl/` | Canonical verl source used by every current experiment. |
| `sdpo_verl/verl/trainer/ppo/cspd.py` | Prefix selection, value conversion, posterior construction, tail projection, and CSPD loss. |
| `sdpo_verl/verl/trainer/ppo/ray_trainer.py` | Builds frozen CSPD targets and logs GAE/posterior diagnostics. |
| `sdpo_verl/verl/trainer/ppo/core_algos.py` | Registers `loss_mode=cspd`. |
| `sdpo_verl/verl/workers/actor/dp_actor.py` | Gathers student top-K probabilities and reports actor gradient diagnostics. |
| `sdpo_verl/tests/trainer/ppo/test_cspd.py` | Canonical CSPD regression tests. |
| `run_math_ppo_cspd.sh` | Shared PPO/CSPD training entrypoint. |
| `data/` | DAPO-Math-17K materialization and checksum; parquet is not tracked. |
| `output/` | Checkpoints, logs, and Hugging Face caches; not tracked. |
| `swanlog/` | SwanLab local files; not tracked. |

`third_party/verl/` and `upstream_rlcsd/` are historical reference copies and
are not used by the current launcher. They are excluded from the top-level Git
repository. The root `cspd.py` is only a compatibility import; edit the
canonical module under `sdpo_verl/verl/trainer/ppo/cspd.py`.

## Environment and fixed inputs

```text
Environment: /WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full
Model:       /WORK/PUBLIC/alex_work/Meiqi.Gu/models/Qwen3-1.7B
Train data:  data/dapo-math-17k-seed42.parquet
Validation:  MATH-500, AIME 2024, AIME 2025, HMMT 2025
verl source: sdpo_verl/
```

Main defaults:

```text
train batch size       32
rollouts per prompt     8
max prompt length    2048
max response length  8192
actor learning rate   1e-6
critic learning rate  1e-5
PPO epochs               1
CSPD top-K                8
CSPD prefixes/response    8
validation samples       12
save frequency           50
validation frequency      5
total steps             300
```

These can be overridden through environment variables accepted by
`run_math_ppo_cspd.sh`.

## Tests

```bash
cd /home/fit/alex1/WORK/Meiqi.Gu/CSPD
export PATH=/WORK/PUBLIC/alex_work/miniconda3/envs/sdpo-full/bin:$PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD/sdpo_verl:$PWD

python -m pytest -q \
  sdpo_verl/tests/trainer/ppo/test_cspd.py \
  test_cspd_value_scale.py \
  test_root_cspd_import.py
```

The current expected result is `10 passed`.

Syntax-only checks:

```bash
bash -n run_math_ppo_cspd.sh submit_comparison.sh submit_cspd_valueprobfix.sh
python -m py_compile \
  sdpo_verl/verl/trainer/ppo/cspd.py \
  sdpo_verl/verl/trainer/ppo/ray_trainer.py \
  sdpo_verl/verl/workers/actor/dp_actor.py
```

## One-GPU smoke test

```bash
salloc --partition=a01 --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=8 --mem=120G --time=02:00:00

srun bash debug_1gpu.sh cspd
srun bash debug_1gpu.sh ppo
```

`debug_1gpu.sh` disables validation, checkpointing, and SwanLab; uses a small
batch, short responses, two candidates, and two prefixes; and runs one step.

## Training commands

### Run directly inside an allocation

```bash
bash run_math_ppo_cspd.sh ppo
bash run_math_ppo_cspd.sh cspd
```

Example CSPD overrides:

```bash
EXP_NAME=CSPD-cspd-qwen3-1.7b-custom \
TOTAL_TRAINING_STEPS=300 \
SAVE_FREQ=50 \
TEST_FREQ=5 \
VAL_N=12 \
CSPD_TOPK=8 \
CSPD_PREFIXES=8 \
bash run_math_ppo_cspd.sh cspd
```

### Submit matched PPO and CSPD

`submit_comparison.sh` submits two independent four-GPU jobs and does not
cancel existing jobs:

```bash
TAG=main-seed42 \
TIME_LIMIT=2-00:00:00 \
TOTAL_TRAINING_STEPS=300 \
SAVE_FREQ=50 \
TEST_FREQ=5 \
VAL_N=12 \
bash submit_comparison.sh
```

### Submit corrected CSPD with gradient diagnostics

```bash
bash submit_cspd_valueprobfix.sh
```

This submits only CSPD with the corrected signed-value conversion, 300 steps,
`save_freq=50`, `test_freq=5`, `mean@12`, and the latest diagnostics:

```text
diagnostics/raw_gae/{mean,std,rms,abs_mean,min,max}
diagnostics/whitened_gae/{mean,std,rms,abs_mean,min,max}
diagnostics/cspd_state_success/*
diagnostics/cspd_successor_success/*
diagnostics/cspd_signed_candidate_advantage/*
diagnostics/cspd_posterior_mass/*
diagnostics/cspd_state_to_mass_ratio/*
diagnostics/cspd_tail_projection_fraction
actor/cspd_grad_norm_if_raw_gae_std_scaled
actor/cspd_grad_norm_after_clip_if_raw_gae_std_scaled
```

The final two metrics are exact counterfactual norms for multiplying the whole
CSPD loss by `1 / std(raw_GAE)`; the actual loss remains unchanged.

## Validation and checkpoint evaluation

Step-zero verifier smoke:

```bash
bash submit_verify_v3.sh
```

Evaluate the hard-coded PPO and CSPD step-300 trainer checkpoints:

```bash
bash submit_eval_ckpts_v3.sh
```

Evaluate the hard-coded PPO Hugging Face actor checkpoint:

```bash
bash submit_eval_ppo_retry_v3.sh
```

This legacy retry disables resume and SwanLab and runs validation only. Inspect
its checkpoint path before reuse.

Evaluate the hard-coded merged PPO actor:

```bash
bash submit_eval_ppo_merged_v4.sh
```

This checks for `model.safetensors`, disables the critic, and runs validation
only. Inspect the model path before reuse.

## Monitoring and result extraction

```bash
bash inspect_slurm_jobs.sh JOB_ID [JOB_ID ...]
bash check_training_queue.sh JOB_ID [JOB_ID ...]
bash check_eval_status.sh JOB_ID

perl extract_val_progress.pl output/EXPERIMENT/train.log
perl summarize_training_reward.pl output/EXPERIMENT/train.log
```

`extract_val_progress.pl` prints the four validation `mean@12` series.
`summarize_training_reward.pl` groups signed training reward into 50-step bins
and reports implied accuracy `(mean_reward + 1) / 2`.

## Data and verifier utilities

| Script | Purpose |
| --- | --- |
| `make_dapo17k_subset.py` | Materializes the first 17,000 rows of the shared DAPO-Math parquet into `data/`. |
| `probe_parquet.py` | Prints parquet row and row-group counts. |
| `diagnose_math500.py` | Recomputes verifier scores for logged MATH-500 examples; paths are experiment-specific. |
| `run_cspd.py` | Thin Python entrypoint; the shell launcher is preferred. |
| `test_cspd_value_scale.py` | Tests the `{-1,+1}` critic to `[0,1]` conversion. |
| `test_root_cspd_import.py` | Ensures the root compatibility import resolves to the canonical signed-value implementation. |

```bash
python make_dapo17k_subset.py
python probe_parquet.py data/dapo-math-17k-seed42.parquet
sha256sum -c data/dapo-math-17k-seed42.sha256
```

## Complete shell-script reference

| Script | Effect |
| --- | --- |
| `run_math_ppo_cspd.sh` | Shared training/validation entrypoint; takes `ppo` or `cspd`. |
| `debug_1gpu.sh` | One-step one-GPU smoke configuration without save, validation, or SwanLab. |
| `submit_comparison.sh` | Submits matched four-GPU PPO and CSPD jobs; does not cancel jobs. |
| `submit_cspd_valueprobfix.sh` | Submits the latest corrected CSPD diagnostic job only. |
| `submit_verify_v3.sh` | Submits a step-zero/validation verifier smoke. |
| `submit_eval_ckpts_v3.sh` | Evaluates hard-coded step-300 PPO and CSPD trainer checkpoints. |
| `submit_eval_ppo_retry_v3.sh` | Legacy validation-only retry using a hard-coded PPO HF actor. |
| `submit_eval_ppo_merged_v4.sh` | Validation-only run for a hard-coded merged PPO actor. |
| `inspect_slurm_jobs.sh` | Shows detailed Slurm configuration and accounting for job IDs. |
| `check_training_queue.sh` | Shows queue, start estimates, priority, and partition nodes. |
| `check_eval_status.sh` | Shows queue/accounting/log tail for one legacy evaluation job. |
| `resubmit_comparison_2d.sh` | **Legacy/destructive:** may cancel only hard-coded pending jobs `461081` and `461082`, then submits a historical comparison. Do not reuse without reviewing it. |

## Output locations

```text
output/EXPERIMENT/train.log       per-experiment console log
output/EXPERIMENT/checkpoints/    verl checkpoints
output/logs/JOB_ID-*.log          Slurm stdout
output/logs/JOB_ID-*.err          Slurm stderr
swanlog/                          SwanLab local metadata
```

These paths, caches, parquet files, secrets, and reference repositories are
excluded by `.gitignore`.

## Source provenance

`SOURCE_VERL_COMMIT`, `SOURCE_VERL_WORKTREE.patch`, and
`sdpo_verl/SOURCE.txt` record the source snapshot and local changes used to
construct the canonical verl tree.

## GitHub upload

Create an empty GitHub repository, then review before the first commit:

```bash
cd /home/fit/alex1/WORK/Meiqi.Gu/CSPD
git status --short
git add .
git status --short
git diff --cached --stat
```

Commit and push:

```bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL"
git commit -m "Initial CSPD implementation"
git branch -M main
git remote add origin git@github.com:YOUR_GITHUB_USER/YOUR_REPOSITORY.git
git push -u origin main
```

Do not use `git add -f` on ignored output, checkpoint, cache, parquet, SwanLab,
or `.env` files.
