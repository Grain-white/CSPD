"""Critic-Induced Success-Posterior Distillation helpers."""

from __future__ import annotations

import torch

from verl import DataProto


def select_prefixes(entropy: torch.Tensor, response_mask: torch.Tensor, max_prefixes: int) -> torch.Tensor:
    valid = response_mask.bool()
    if max_prefixes <= 0:
        return torch.zeros_like(valid)
    k = min(max_prefixes, entropy.size(-1))
    selected = torch.zeros_like(valid)
    selected.scatter_(1, entropy.float().masked_fill(~valid, -torch.inf).topk(k, dim=1).indices, True)
    return selected & valid


def values_to_success(values, reward_range: str = "pm1"):
    """Map critic values to success probabilities for CSPD.

    - ``pm1``: verifier rewards in {-1, +1}, so V = 2P(success) - 1.
    - ``01``: verifier rewards in {0, 1} (e.g. GSM8K), so V ≈ P(success).
    """
    v = values.float()
    if reward_range == "pm1":
        return ((v + 1.0) * 0.5).clamp(0, 1)
    if reward_range == "01":
        return v.clamp(0, 1)
    raise ValueError(f"Unknown CSPD reward_range={reward_range!r}; expected 'pm1' or '01'")


def build_success_posterior(
    behavior_logp,
    successor_values,
    state_values,
    selected_mask,
    eps=1e-6,
    return_diagnostics=False,
    reward_range: str = "pm1",
):
    """Equations (23)-(24), with the tail projected to its feasible interval."""
    successor_success = values_to_success(successor_values, reward_range)
    state_success = values_to_success(state_values, reward_range)
    pi_top = behavior_logp.float().exp()
    top_mass = pi_top * successor_success
    pi_tail = (1 - pi_top.sum(-1)).clamp(0, 1)
    raw_tail_mass = state_success - top_mass.sum(-1)
    tail_mass = raw_tail_mass.clamp_min(0)
    tail_mass = torch.minimum(tail_mass, pi_tail)
    masses = torch.cat((top_mass, tail_mass.unsqueeze(-1)), -1)
    total = masses.sum(-1)
    valid = selected_mask.bool() & (total > eps)
    target = torch.where(valid.unsqueeze(-1), masses / total.clamp_min(eps).unsqueeze(-1), 0)
    # Equation (10) is weighted by the frozen state value V_k(s), not by the
    # projected approximation's residual total mass. They coincide only for a
    # perfectly consistent critic.
    value_weight = torch.where(valid, state_success, 0)
    if not return_diagnostics:
        return target.detach(), value_weight.detach()

    diagnostics = {
        "state_success": state_success.detach(),
        "successor_success": successor_success.detach(),
        "signed_candidate_advantage": (
            successor_values.float() - state_values.float().unsqueeze(-1)
        ).detach(),
        "behavior_tail_probability": pi_tail.detach(),
        "raw_tail_mass": raw_tail_mass.detach(),
        "projected_tail_mass": tail_mass.detach(),
        "posterior_mass": total.detach(),
        "state_to_mass_ratio": (state_success / total.clamp_min(eps)).detach(),
        "tail_projection_mask": (
            selected_mask.bool() & ((raw_tail_mass < 0) | (raw_tail_mass > pi_tail))
        ).detach(),
        "valid_mask": valid.detach(),
    }
    return target.detach(), value_weight.detach(), diagnostics


def cspd_forward_kl(student_topk_logp, target, value_weight, selected_mask, eps=1e-6):
    top_prob = student_topk_logp.float().exp()
    tail_logp = (1 - top_prob.sum(-1)).clamp_min(eps).log()
    student_logp = torch.cat((student_topk_logp.float(), tail_logp.unsqueeze(-1)), -1)
    per_prefix = -(target.float() * student_logp).sum(-1)
    mask = selected_mask.float()
    weight = value_weight.float() * mask
    denom = mask.sum().clamp_min(1)
    loss = (per_prefix * weight).sum() / denom
    with torch.no_grad():
        entropy = -(target.float() * target.float().clamp_min(eps).log()).sum(-1)
        metrics = {
            "cspd_loss": loss.detach().item(),
            "cspd_selected_prefixes": mask.sum().item(),
            "cspd_value_weight": (weight.sum() / denom).item(),
            "cspd_target_entropy": ((entropy * mask).sum() / denom).item(),
            "cspd_tail_probability": ((target[..., -1] * mask).sum() / denom).item(),
        }
    return loss, metrics


def make_successor_batch(batch: DataProto, selected_mask: torch.Tensor, topk_ids: torch.Tensor, pad_token_id: int):
    """Make one-token critic queries whose prediction position is each s+a.

    A trailing dummy token makes verl's critic return the value at the candidate
    token position. It is causally invisible to that value.
    """
    rows, times = selected_mask.nonzero(as_tuple=True)
    k = topk_ids.size(-1)
    if rows.numel() == 0:
        return None, rows, times
    rows = rows.repeat_interleave(k)
    times = times.repeat_interleave(k)
    actions = topk_ids[selected_mask].reshape(-1)
    source_ids = batch.batch["input_ids"]
    source_mask = batch.batch["attention_mask"]
    responses = batch.batch["responses"]
    prompt_width = source_ids.size(1) - responses.size(1)
    prompt_tokens = [source_ids[i, :prompt_width][source_mask[i, :prompt_width].bool()] for i in range(source_ids.size(0))]
    lengths = [prompt_tokens[row].numel() + time + 2 for row, time in zip(rows.tolist(), times.tolist(), strict=True)]
    max_len = max(lengths)
    n = rows.numel()
    ids = torch.full((n, max_len), pad_token_id, dtype=source_ids.dtype)
    attn = torch.zeros((n, max_len), dtype=source_mask.dtype)
    pos = torch.zeros((n, max_len), dtype=batch.batch["position_ids"].dtype)
    for j, (row, time, action) in enumerate(zip(rows.tolist(), times.tolist(), actions.tolist(), strict=True)):
        prefix_ids = torch.cat((prompt_tokens[row], responses[row, :time]))
        sequence = torch.cat((prefix_ids, source_ids.new_tensor([action, pad_token_id])))
        length = sequence.numel()
        start = max_len - length
        ids[j, start:] = sequence
        attn[j, start:] = 1
        pos[j, start:] = torch.arange(length, dtype=pos.dtype)
    responses = torch.full((n, 1), pad_token_id, dtype=source_ids.dtype)
    response_mask = torch.ones((n, 1), dtype=source_mask.dtype)
    return DataProto.from_dict(
        tensors={"input_ids": ids, "attention_mask": attn, "position_ids": pos,
                 "responses": responses, "response_mask": response_mask}
    ), rows, times
