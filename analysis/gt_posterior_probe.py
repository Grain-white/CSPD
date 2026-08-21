#!/usr/bin/env python3
"""Monte-Carlo audit of CSPD's grouped success posterior on saved Math checkpoints.

The probe never mutates a checkpoint.  It merges FSDP shards in CPU memory, samples
continuations from the actor, scores them with verl's exact Math verifier, and then
evaluates either the CSPD or PPO critic on the same states and candidate actions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.utils.reward_score import default_compute_score


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--actor-method", choices=("cspd", "ppo"), required=True)
    p.add_argument("--critic-methods", nargs="+", choices=("cspd", "ppo"), default=("cspd", "ppo"))
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tag", default="math75k-valueprobfix-v5")
    p.add_argument("--dataset-label", default="math75k")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--mc-samples", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--num-prompts", type=int, default=3)
    p.add_argument("--prefixes-per-prompt", type=int, default=1)
    p.add_argument("--anchor-new-tokens", type=int, default=768)
    p.add_argument("--anchor-samples", type=int, default=8)
    p.add_argument("--scan-prompts", type=int, default=48)
    p.add_argument("--require-anchor-success", action="store_true")
    p.add_argument("--panel-source", choices=("generated", "solution"), default="generated")
    p.add_argument("--solution-fraction", type=float, default=0.70)
    p.add_argument("--generation-batch-size", type=int, default=32)
    p.add_argument("--reuse-actor-records", action="store_true")
    p.add_argument("--seed", type=int, default=20260820)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint(root: Path, method: str, tag: str, step: int, kind: str) -> Path:
    return root / "output" / f"CSPD-{method}-qwen3-1.7b-seed42-{tag}" / "checkpoints" / f"global_step_{step}" / kind


def merge_state_dict(shard_dir: Path) -> dict[str, torch.Tensor]:
    if not shard_dir.is_dir():
        raise FileNotFoundError(shard_dir)
    cfg = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(shard_dir),
        target_dir=str(shard_dir / ".unused-probe-target"),
        hf_model_config_path=str(shard_dir / "huggingface"),
    )
    merger = FSDPModelMerger(cfg)
    world_size = merger._get_world_size()
    rank0 = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_names = merger._extract_device_mesh_info(rank0, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_names)
    del rank0
    return merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_names)


def materialize_empty(model, state: dict[str, torch.Tensor], device: str):
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"state mismatch: {incompatible}")
    del state
    model.eval().requires_grad_(False)
    return model.to(device)


def load_actor(shard_dir: Path, device: str):
    state = merge_state_dict(shard_dir)
    cfg = AutoConfig.from_pretrained(shard_dir / "huggingface")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    return materialize_empty(model, state, device)


def load_critic(shard_dir: Path, device: str):
    state = merge_state_dict(shard_dir)
    cfg = AutoConfig.from_pretrained(shard_dir / "huggingface")
    cfg.num_labels = 1
    cfg.problem_type = "regression"
    cfg.classifier_dropout = 0.0
    with init_empty_weights():
        model = AutoModelForTokenClassification.from_config(cfg, torch_dtype=torch.bfloat16)
    return materialize_empty(model, state, device)


def release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prompt_ids(tokenizer, messages) -> list[int]:
    return tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


@torch.inference_mode()
def make_panel(args: argparse.Namespace, tokenizer) -> None:
    if args.panel.exists():
        return
    args.panel.parent.mkdir(parents=True, exist_ok=True)
    limit = args.scan_prompts if args.require_anchor_success else args.num_prompts
    data = pd.read_parquet(args.data).iloc[:limit]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(args.device).eval()
    panel = []
    for row_idx, row in data.iterrows():
        pids = prompt_ids(tokenizer, row["prompt"])
        if args.panel_source == "solution":
            solution = str(row["extra_info"]["solution"])
            solution_ids = tokenizer(solution, add_special_tokens=False)["input_ids"]
            cut = min(len(solution_ids) - 1, max(1, round(len(solution_ids) * args.solution_fraction)))
            generated = torch.tensor(pids + solution_ids[:cut], device=args.device)
        elif args.require_anchor_success:
            x = torch.tensor([pids], device=args.device)
            candidates_generated = model.generate(
                x,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                num_return_sequences=args.anchor_samples,
                max_new_tokens=args.anchor_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = None
            for candidate in candidates_generated:
                text = tokenizer.decode(candidate[len(pids) :], skip_special_tokens=True)
                score = default_compute_score(
                    data_source=str(row["data_source"]),
                    solution_str=text,
                    ground_truth=str(row["reward_model"]["ground_truth"]),
                    extra_info=None,
                )
                if isinstance(score, dict):
                    score = score.get("score", score.get("reward", score.get("acc", 0.0)))
                if float(score) > 0:
                    generated = candidate
                    break
            if generated is None:
                continue
        else:
            x = torch.tensor([pids], device=args.device)
            generated = model.generate(
                x,
                do_sample=False,
                max_new_tokens=args.anchor_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )[0]
        response = generated[len(pids) :]
        if len(response) < 12:
            raise RuntimeError(f"anchor response too short for row {row_idx}: {len(response)}")
        logits = model(generated.unsqueeze(0), use_cache=False).logits[0].float()
        if args.panel_source == "solution":
            candidates = [len(generated)]
        else:
            start = len(pids) + min(32, max(1, len(response) // 4))
            end = len(generated) - min(16, max(1, len(response) // 8))
            candidates = list(range(start, max(start + 1, end)))
        entropy = torch.distributions.Categorical(logits=logits[[i - 1 for i in candidates]]).entropy().cpu()
        order = torch.argsort(entropy, descending=True).tolist()
        chosen: list[int] = []
        min_gap = max(16, len(response) // 8)
        for j in order:
            pos = candidates[j]
            if all(abs(pos - old) >= min_gap for old in chosen):
                chosen.append(pos)
            if len(chosen) == args.prefixes_per_prompt:
                break
        gt = str(row["reward_model"]["ground_truth"])
        source = str(row["data_source"])
        for prefix_idx, pos in enumerate(sorted(chosen)):
            panel.append(
                {
                    "panel_id": f"row{row_idx}-p{prefix_idx}",
                    "row_index": int(row_idx),
                    "data_source": source,
                    "ground_truth": gt,
                    "prompt_len": len(pids),
                    "prefix_ids": generated[:pos].cpu().tolist(),
                    "anchor_response_tokens": int(len(response)),
                    "prefix_response_tokens": int(pos - len(pids)),
                    "base_entropy": float(torch.distributions.Categorical(logits=logits[pos - 1]).entropy()),
                    "panel_source": args.panel_source,
                }
            )
        if len(panel) >= args.num_prompts * args.prefixes_per_prompt:
            break
    if len(panel) < args.num_prompts * args.prefixes_per_prompt:
        raise RuntimeError(
            f"only found {len(panel)} panel prefixes; requested "
            f"{args.num_prompts * args.prefixes_per_prompt}"
        )
    args.panel.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    release(model)


def grouped_distribution(top_ids: np.ndarray, first_ids: np.ndarray, successes: np.ndarray):
    success_counts = np.zeros(len(top_ids) + 1, dtype=np.float64)
    action_counts = np.zeros(len(top_ids) + 1, dtype=np.float64)
    lookup = {int(token): i for i, token in enumerate(top_ids)}
    for token, reward in zip(first_ids, successes, strict=True):
        bucket = lookup.get(int(token), len(top_ids))
        action_counts[bucket] += 1
        success_counts[bucket] += reward
    total_success = success_counts.sum()
    q = success_counts / total_success if total_success > 0 else np.full_like(success_counts, np.nan)
    return q, success_counts, action_counts


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        return np.full_like(x, np.nan)
    x = np.maximum(x, 0.0)
    return x / x.sum() if x.sum() > 0 else np.full_like(x, np.nan)


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = normalize(p), normalize(q)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        return math.nan
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.maximum(q[mask], EPS)))))


def js(p: np.ndarray, q: np.ndarray) -> float:
    p, q = normalize(p), normalize(q)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    p, q = normalize(p), normalize(q)
    return float(0.5 * np.abs(p - q).sum()) if np.isfinite(p).all() and np.isfinite(q).all() else math.nan


@torch.inference_mode()
def sample_actor(args: argparse.Namespace, tokenizer, panel: list[dict]) -> list[dict]:
    actor_dir = checkpoint(args.root, args.actor_method, args.tag, args.step, "actor")
    model = load_actor(actor_dir, args.device)
    records = []
    for item_idx, item in enumerate(panel):
        ids = torch.tensor([item["prefix_ids"]], device=args.device)
        logits = model(ids, use_cache=False).logits[0, -1].float()
        probs = logits.softmax(-1)
        top_prob, top_ids = probs.topk(args.topk)
        generator_seed = args.seed + 100000 * args.step + 1000 * (args.actor_method == "ppo") + item_idx
        set_seed(generator_seed)
        generated_chunks = []
        remaining = args.mc_samples
        while remaining:
            n = min(remaining, args.generation_batch_size)
            generated_chunks.append(
                model.generate(
                    ids,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    num_return_sequences=n,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            )
            remaining -= n
        generated = torch.cat(generated_chunks, dim=0)
        prefix_len = ids.shape[1]
        first_ids = generated[:, prefix_len].cpu().numpy()
        successes = []
        texts = []
        for sequence in generated:
            response_text = tokenizer.decode(sequence[item["prompt_len"] :], skip_special_tokens=True)
            score = default_compute_score(
                data_source=item["data_source"],
                solution_str=response_text,
                ground_truth=item["ground_truth"],
                extra_info=None,
            )
            if isinstance(score, dict):
                score = score.get("score", score.get("reward", score.get("acc", 0.0)))
            successes.append(float(score > 0))
            texts.append(response_text[-300:])
        successes_np = np.asarray(successes, dtype=np.float64)
        top_np = top_ids.cpu().numpy()
        q_gt, success_counts, action_counts = grouped_distribution(top_np, first_ids, successes_np)
        records.append(
            {
                **item,
                "actor_method": args.actor_method,
                "dataset": args.dataset_label,
                "step": args.step,
                "mc_samples": args.mc_samples,
                "generator_seed": generator_seed,
                "top_ids": top_np.tolist(),
                "top_tokens": [tokenizer.decode([int(x)]) for x in top_np],
                "pi_top": top_prob.cpu().float().numpy().tolist(),
                "pi_tail": float(1.0 - top_prob.sum()),
                "first_token_counts": action_counts.tolist(),
                "success_counts": success_counts.tolist(),
                "q_gt": q_gt.tolist(),
                "v_gt": float(successes_np.mean()),
                "n_success": int(successes_np.sum()),
                "sample_tails": texts[:3],
            }
        )
    release(model)
    return records


@torch.inference_mode()
def critic_values(model, sequences: list[list[int]], device: str) -> np.ndarray:
    vals = []
    for seq in sequences:
        ids = torch.tensor([seq], device=device)
        value = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1, 0]
        vals.append(float(value.float().cpu()))
    return np.asarray(vals, dtype=np.float64)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return math.nan
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def vector_cosine(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > EPS else math.nan


@torch.inference_mode()
def evaluate_critic(args: argparse.Namespace, record: dict, critic_method: str, model) -> dict:
    prefix = record["prefix_ids"]
    top_ids = record["top_ids"]
    signed_v = critic_values(model, [prefix], args.device)[0]
    signed_q = critic_values(model, [prefix + [int(a)] for a in top_ids], args.device)
    v = float(np.clip((signed_v + 1.0) * 0.5, 0.0, 1.0))
    q = np.clip((signed_q + 1.0) * 0.5, 0.0, 1.0)
    pi_top = np.asarray(record["pi_top"], dtype=np.float64)
    pi_tail = float(record["pi_tail"])
    top_mass = pi_top * q
    raw_tail = v - top_mass.sum()
    projected_tail = float(np.clip(raw_tail, 0.0, pi_tail))
    posterior_mass = float(top_mass.sum() + projected_tail)
    q_cspd = normalize(np.r_[top_mass, projected_tail])
    q_actor = normalize(np.r_[pi_top, pi_tail])
    q_gt = np.asarray(record["q_gt"], dtype=np.float64)
    gt_success_mass = np.asarray(record["success_counts"], dtype=np.float64) / record["mc_samples"]
    projected_mass = np.r_[top_mass, projected_tail]
    loss_target_mass = v * q_cspd
    actor_grouped = np.r_[pi_top, pi_tail]
    gt_update = gt_success_mass - record["v_gt"] * actor_grouped
    cspd_update = loss_target_mass - v * actor_grouped
    gt_update_norm = float(np.linalg.norm(gt_update))
    cspd_update_norm = float(np.linalg.norm(cspd_update))
    direction_match = np.sign(cspd_update) == np.sign(gt_update)
    cspd_positive = cspd_update > 0
    gt_positive = gt_update > 0
    harmful_coordinate_fraction = float(
        np.abs(cspd_update[~direction_match]).sum() / max(np.abs(cspd_update).sum(), EPS)
    )
    action_counts = np.asarray(record["first_token_counts"], dtype=np.float64)
    success_counts = np.asarray(record["success_counts"], dtype=np.float64)
    observed = action_counts[:-1] > 0
    empirical_q = np.divide(success_counts[:-1], action_counts[:-1], out=np.zeros_like(q), where=observed)
    posterior_ok = np.isfinite(q_gt).all()
    return {
        "panel_id": record["panel_id"],
        "row_index": record["row_index"],
        "actor_method": args.actor_method,
        "dataset": args.dataset_label,
        "critic_method": critic_method,
        "step": args.step,
        "mc_samples": record["mc_samples"],
        "n_success": record["n_success"],
        "v_gt": record["v_gt"],
        "v_critic": v,
        "v_abs_error": abs(v - record["v_gt"]),
        "v_brier": (v - record["v_gt"]) ** 2,
        "top_mass_gt": float(gt_success_mass[:-1].sum()),
        "top_mass_critic": float(top_mass.sum()),
        "top_mass_l1": float(np.abs(top_mass - gt_success_mass[:-1]).sum()),
        "tail_mass_gt": float(gt_success_mass[-1]),
        "tail_mass_raw": float(raw_tail),
        "tail_mass_projected": projected_tail,
        "tail_mass_abs_error_raw": abs(raw_tail - gt_success_mass[-1]),
        "tail_mass_abs_error_projected": abs(projected_tail - gt_success_mass[-1]),
        "tail_projection": float(raw_tail < 0.0 or raw_tail > pi_tail),
        "pi_tail": pi_tail,
        "posterior_mass": posterior_mass,
        "state_to_mass_ratio": v / max(posterior_mass, EPS),
        "renorm_amplification": v / max(posterior_mass, EPS),
        "weight_mass_abs_gap": abs(v - posterior_mass),
        # Gradient-relevant diagnostics.  Unlike conditional-posterior KL,
        # these remain well-scaled when success probability is small.
        "mass_l1_projected": float(np.abs(projected_mass - gt_success_mass).sum()),
        "mass_l1_loss_target": float(np.abs(loss_target_mass - gt_success_mass).sum()),
        "update_cosine_gt": vector_cosine(cspd_update, gt_update),
        "update_dot_gt": float(np.dot(cspd_update, gt_update)),
        "update_norm_cspd": cspd_update_norm,
        "update_norm_gt": gt_update_norm,
        "update_norm_ratio_gt": cspd_update_norm / gt_update_norm if gt_update_norm > EPS else math.nan,
        "update_l2_error_gt": float(np.linalg.norm(cspd_update - gt_update)),
        "update_sign_agreement_gt": float(direction_match.mean()) if gt_update_norm > EPS else math.nan,
        "update_sign_agreement_topk_gt": float(direction_match[:-1].mean()) if gt_update_norm > EPS else math.nan,
        "harmful_coordinate_fraction": harmful_coordinate_fraction if gt_update_norm > EPS else math.nan,
        "push_precision_gt": float((cspd_positive & gt_positive).sum() / max(cspd_positive.sum(), 1)),
        "push_recall_gt": float((cspd_positive & gt_positive).sum() / max(gt_positive.sum(), 1)),
        # JSON vectors keep the per-action directions available in the detail
        # CSV instead of collapsing the whole comparison to one cosine.
        "bucket_token_ids": json.dumps([int(token) for token in top_ids] + [-1]),
        "bucket_tokens": json.dumps(list(record["top_tokens"]) + ["<TAIL>"], ensure_ascii=False),
        "actor_prob_by_bucket": json.dumps(actor_grouped.tolist()),
        "gt_success_mass_by_bucket": json.dumps(gt_success_mass.tolist()),
        "projected_mass_by_bucket": json.dumps(projected_mass.tolist()),
        "loss_target_mass_by_bucket": json.dumps(loss_target_mass.tolist()),
        "update_gt_by_bucket": json.dumps(gt_update.tolist()),
        "update_cspd_by_bucket": json.dumps(cspd_update.tolist()),
        "update_sign_match_by_bucket": json.dumps(direction_match.tolist()),
        "q_top_observed": int(observed.sum()),
        "q_mae_observed": float(np.abs(q[observed] - empirical_q[observed]).mean()) if observed.any() else math.nan,
        "q_rank_corr_observed": rank_correlation(q[observed], empirical_q[observed]),
        "posterior_valid": float(posterior_ok),
        "kl_gt_cspd": kl(q_gt, q_cspd),
        "js_gt_cspd": js(q_gt, q_cspd),
        "tv_gt_cspd": tv(q_gt, q_cspd),
        "kl_gt_actor": kl(q_gt, q_actor),
        "js_gt_actor": js(q_gt, q_actor),
        "tv_gt_actor": tv(q_gt, q_actor),
        "kl_cspd_actor": kl(q_cspd, q_actor),
        "js_cspd_actor": js(q_cspd, q_actor),
        "tv_cspd_actor": tv(q_cspd, q_actor),
        "cspd_kl_gain_over_actor": kl(q_gt, q_actor) - kl(q_gt, q_cspd),
        # CSPD's proposal loss is state-value weighted.  These metrics test the
        # hypothesis that an equally biased posterior is less harmful on a harder
        # dataset simply because V(s) is smaller.
        "v_weighted_kl_gt_cspd": v * kl(q_gt, q_cspd),
        "v_weighted_kl_cspd_actor": v * kl(q_cspd, q_actor),
        "v_weighted_tv_gt_cspd": v * tv(q_gt, q_cspd),
        "harmful_update_pressure": v * max(0.0, kl(q_gt, q_cspd) - kl(q_gt, q_actor)) if posterior_ok else math.nan,
        "beneficial_update_pressure": v * max(0.0, kl(q_gt, q_actor) - kl(q_gt, q_cspd)) if posterior_ok else math.nan,
        "net_update_pressure": v * (kl(q_gt, q_actor) - kl(q_gt, q_cspd)),
        "v_bin": min(3, int(v * 4.0)),
        "signed_v": signed_v,
        "signed_q_mean": float(signed_q.mean()),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    keys = ["dataset", "actor_method", "critic_method", "step"]
    numeric = [c for c in frame.columns if c not in keys + ["panel_id"] and pd.api.types.is_numeric_dtype(frame[c])]
    return frame.groupby(keys, as_index=False)[numeric].mean().to_dict("records")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    make_panel(args, tokenizer)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    actor_path = args.output_dir / f"actor-{args.actor_method}-step{args.step}.json"
    if args.reuse_actor_records and actor_path.exists():
        actor_records = json.loads(actor_path.read_text(encoding="utf-8"))
        for record in actor_records:
            counts = np.asarray(record["success_counts"], dtype=np.float64)
            record["q_gt"] = (counts / counts.sum()).tolist() if counts.sum() > 0 else [math.nan] * len(counts)
        actor_path.write_text(json.dumps(actor_records, indent=2), encoding="utf-8")
    else:
        actor_records = sample_actor(args, tokenizer, panel)
        actor_path.write_text(json.dumps(actor_records, indent=2), encoding="utf-8")

    all_rows = []
    for critic_method in args.critic_methods:
        critic_dir = checkpoint(args.root, critic_method, args.tag, args.step, "critic")
        critic = load_critic(critic_dir, args.device)
        all_rows.extend(evaluate_critic(args, record, critic_method, critic) for record in actor_records)
        release(critic)

    detail = pd.DataFrame(all_rows)
    detail_path = args.output_dir / f"detail-actor-{args.actor_method}-step{args.step}.csv"
    detail.to_csv(detail_path, index=False)
    summary = aggregate(all_rows)
    summary_path = args.output_dir / f"summary-actor-{args.actor_method}-step{args.step}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
