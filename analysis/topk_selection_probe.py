#!/usr/bin/env python3
"""Compare the implemented policy top-K with proposal's K0 -> posterior-mass top-K.

This is a read-only checkpoint probe.  At each frozen prefix it proposes K0
actions by actor probability, scores all K0 successor states with the critic,
and retains K actions by pi(a|s) * Q(s,a), as specified in proposal_v3.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from gt_posterior_probe import EPS, checkpoint, load_actor, load_critic, release, vector_cosine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--actor-method", choices=("cspd", "ppo"), default="cspd")
    p.add_argument("--critic-method", choices=("cspd", "ppo"), default="cspd")
    p.add_argument("--tag", required=True)
    p.add_argument("--dataset-label", required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--retain-k", type=int, default=8)
    p.add_argument("--proposal-k0", type=int, nargs="+", default=(16, 32, 64))
    p.add_argument("--critic-batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.inference_mode()
def critic_values_batched(model, sequences: list[list[int]], device: str, batch_size: int) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        max_len = max(map(len, chunk))
        ids = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for row, seq in enumerate(chunk):
            ids[row, -len(seq) :] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[row, -len(seq) :] = 1
        position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
        output = model(ids, attention_mask=mask, position_ids=position_ids, use_cache=False).logits[:, -1, 0]
        values.append(output.float().cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def selection_state(
    selected: np.ndarray,
    pi: np.ndarray,
    successor_q: np.ndarray,
    state_v: float,
) -> dict[str, object]:
    """Return target/update on common [max-K0 tokens + outside] coordinates.

    For tokens represented by the tail bucket, the bucket gradient is expanded
    according to their behavior-policy probability.  Thus the returned update
    is the exact grouped-logit direction at the frozen behavior policy.
    """
    chosen = np.zeros(len(pi), dtype=bool)
    chosen[selected] = True
    selected_pi = pi[selected]
    selected_mass = selected_pi * successor_q[selected]
    pi_outside = max(0.0, 1.0 - float(pi.sum()))
    pi_tail = max(0.0, 1.0 - float(selected_pi.sum()))
    raw_tail = state_v - float(selected_mass.sum())
    projected_tail = float(np.clip(raw_tail, 0.0, pi_tail))
    total_mass = float(selected_mass.sum() + projected_tail)

    grouped_pi = np.r_[pi, pi_outside]
    effective_q = np.zeros_like(grouped_pi)
    if total_mass > EPS:
        effective_q[selected] = selected_mass / total_mass
        tail_target = projected_tail / total_mass
        if pi_tail > EPS:
            omitted = np.flatnonzero(~chosen)
            effective_q[omitted] = tail_target * pi[omitted] / pi_tail
            effective_q[-1] = tail_target * pi_outside / pi_tail
    update = state_v * (effective_q - grouped_pi)
    return {
        "pi_mass": float(selected_pi.sum()),
        "success_mass": float(selected_mass.sum()),
        "pi_tail": pi_tail,
        "raw_tail": raw_tail,
        "projected_tail": projected_tail,
        "tail_projection": float(raw_tail < 0.0 or raw_tail > pi_tail),
        "posterior_mass": total_mass,
        "renorm_amplification": state_v / max(total_mass, EPS),
        "update": update,
    }


@torch.inference_mode()
def evaluate(args: argparse.Namespace, tokenizer, panel: list[dict]) -> list[dict]:
    actor = load_actor(checkpoint(args.root, args.actor_method, args.tag, args.step, "actor"), args.device)
    critic = load_critic(checkpoint(args.root, args.critic_method, args.tag, args.step, "critic"), args.device)
    max_k0 = max(args.proposal_k0)
    if args.retain_k > min(args.proposal_k0):
        raise ValueError("retain-k must not exceed proposal-k0")
    rows: list[dict] = []

    for item in panel:
        prefix = item["prefix_ids"]
        ids = torch.tensor([prefix], dtype=torch.long, device=args.device)
        probs_full = actor(ids, use_cache=False).logits[0, -1].float().softmax(-1)
        pi_t, token_ids_t = probs_full.topk(max_k0)
        pi = pi_t.cpu().numpy().astype(np.float64)
        token_ids = token_ids_t.cpu().numpy().astype(np.int64)
        signed = critic_values_batched(
            critic,
            [prefix] + [prefix + [int(token)] for token in token_ids],
            args.device,
            args.critic_batch_size,
        )
        state_v = float(np.clip((signed[0] + 1.0) * 0.5, 0.0, 1.0))
        successor_q = np.clip((signed[1:] + 1.0) * 0.5, 0.0, 1.0)
        current = np.arange(args.retain_k)
        current_state = selection_state(current, pi, successor_q, state_v)

        for k0 in args.proposal_k0:
            candidate_mass = pi[:k0] * successor_q[:k0]
            proposal = np.argsort(-candidate_mass, kind="stable")[: args.retain_k]
            full_k0 = np.arange(k0)
            proposal_state = selection_state(proposal, pi, successor_q, state_v)
            full_state = selection_state(full_k0, pi, successor_q, state_v)
            current_set = set(current.tolist())
            proposal_set = set(proposal.tolist())
            new = np.asarray(sorted(proposal_set - current_set), dtype=np.int64)
            dropped = np.asarray(sorted(current_set - proposal_set), dtype=np.int64)
            overlap = len(current_set & proposal_set)
            current_update = np.asarray(current_state["update"])
            proposal_update = np.asarray(proposal_state["update"])
            full_update = np.asarray(full_state["update"])

            rows.append(
                {
                    "dataset": args.dataset_label,
                    "actor_method": args.actor_method,
                    "critic_method": args.critic_method,
                    "step": args.step,
                    "panel_id": item["panel_id"],
                    "retain_k": args.retain_k,
                    "proposal_k0": k0,
                    "state_v": state_v,
                    "overlap_count": overlap,
                    "overlap_fraction": overlap / args.retain_k,
                    "replacement_count": args.retain_k - overlap,
                    "jaccard": overlap / (2 * args.retain_k - overlap),
                    "current_pi_mass": current_state["pi_mass"],
                    "proposal_pi_mass": proposal_state["pi_mass"],
                    "proposal_pi_mass_change": proposal_state["pi_mass"] - current_state["pi_mass"],
                    "current_success_mass": current_state["success_mass"],
                    "proposal_success_mass": proposal_state["success_mass"],
                    "proposal_success_mass_gain": proposal_state["success_mass"] - current_state["success_mass"],
                    "proposal_success_mass_ratio": proposal_state["success_mass"] / max(current_state["success_mass"], EPS),
                    "new_token_actor_rank_mean": float((new + 1).mean()) if len(new) else math.nan,
                    "new_token_actor_rank_max": float((new + 1).max()) if len(new) else math.nan,
                    "new_token_q_mean": float(successor_q[new].mean()) if len(new) else math.nan,
                    "dropped_token_q_mean": float(successor_q[dropped].mean()) if len(dropped) else math.nan,
                    "current_tail_projection": current_state["tail_projection"],
                    "proposal_tail_projection": proposal_state["tail_projection"],
                    "current_posterior_mass": current_state["posterior_mass"],
                    "proposal_posterior_mass": proposal_state["posterior_mass"],
                    "current_renorm_amplification": current_state["renorm_amplification"],
                    "proposal_renorm_amplification": proposal_state["renorm_amplification"],
                    "update_cosine_current_proposal": vector_cosine(current_update, proposal_update),
                    "update_norm_current": float(np.linalg.norm(current_update)),
                    "update_norm_proposal": float(np.linalg.norm(proposal_update)),
                    "update_norm_ratio_proposal_current": float(np.linalg.norm(proposal_update))
                    / max(float(np.linalg.norm(current_update)), EPS),
                    "update_l2_current_proposal": float(np.linalg.norm(current_update - proposal_update)),
                    "update_cosine_current_full_k0": vector_cosine(current_update, full_update),
                    "update_cosine_proposal_full_k0": vector_cosine(proposal_update, full_update),
                    "proposal_cosine_gain_to_full_k0": vector_cosine(proposal_update, full_update)
                    - vector_cosine(current_update, full_update),
                    "current_token_ids": json.dumps(token_ids[current].tolist()),
                    "proposal_token_ids": json.dumps(token_ids[proposal].tolist()),
                    "new_token_ids": json.dumps(token_ids[new].tolist()),
                    "dropped_token_ids": json.dumps(token_ids[dropped].tolist()),
                    "new_tokens": json.dumps([tokenizer.decode([int(token_ids[i])]) for i in new], ensure_ascii=False),
                    "dropped_tokens": json.dumps(
                        [tokenizer.decode([int(token_ids[i])]) for i in dropped], ensure_ascii=False
                    ),
                }
            )

    release(actor)
    release(critic)
    gc.collect()
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    rows = evaluate(args, tokenizer, panel)
    path = args.output_dir / (
        f"detail-{args.dataset_label}-actor-{args.actor_method}-critic-{args.critic_method}-step{args.step}.csv"
    )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).drop(columns=[c for c in pd.DataFrame(rows).columns if c.endswith("token_ids")]).to_string(index=False))


if __name__ == "__main__":
    main()
