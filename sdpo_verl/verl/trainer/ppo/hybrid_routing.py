from collections import defaultdict
from typing import Any

import numpy as np
import torch


def compute_success_mask(reward_tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Match RayPPOTrainer._collect_solutions_by_uid success semantics."""
    return reward_tensor.sum(dim=-1) >= threshold


def compute_failed_mask(success_mask: torch.Tensor) -> torch.Tensor:
    return ~success_mask.bool()


def compute_group_has_success(success_mask: torch.Tensor, uids: np.ndarray | list[Any]) -> torch.Tensor:
    success_by_uid: dict[Any, bool] = defaultdict(bool)
    success_cpu = success_mask.detach().cpu().bool().tolist()
    for uid, is_success in zip(uids, success_cpu, strict=True):
        success_by_uid[uid] = success_by_uid[uid] or is_success
    return torch.tensor(
        [success_by_uid[uid] for uid in uids],
        dtype=torch.bool,
        device=success_mask.device,
    )


def compute_offpolicy_degree(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_log_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    behavior_log_probs = rollout_log_probs if rollout_log_probs is not None else old_log_probs
    token_count = response_mask.sum(dim=-1).clamp(min=1.0)
    return ((current_log_probs - behavior_log_probs).abs() * response_mask).sum(dim=-1) / token_count


def compute_far_gate(
    offpolicy_degree: torch.Tensor,
    tau_far: float,
    gate_temp: float = 0.1,
    use_soft_gate: bool = True,
) -> torch.Tensor:
    if use_soft_gate:
        return torch.sigmoid((offpolicy_degree - tau_far) / max(float(gate_temp), 1e-8))
    return (offpolicy_degree > tau_far).to(offpolicy_degree.dtype)


def compute_sdpo_route_mask(
    failed_mask: torch.Tensor,
    group_has_success: torch.Tensor,
    far_gate: torch.Tensor,
    mode: str = "far_failed_with_correct_group",
) -> torch.Tensor:
    if mode in {"none", None}:
        return torch.zeros_like(far_gate, dtype=far_gate.dtype)
    if mode == "far_all":
        return far_gate
    if mode in {"far_failed_with_correct_group", "far_failed_correct_group"}:
        return far_gate * failed_mask.to(far_gate.dtype) * group_has_success.to(far_gate.dtype)
    raise ValueError(f"Unsupported sdpo_routing.mode: {mode}")


def compute_far_correct_sft_loss(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    success_mask: torch.Tensor,
    far_gate: torch.Tensor,
    normalize_by_active_tokens: bool = True,
) -> torch.Tensor:
    token_weight = response_mask * success_mask.to(log_probs.dtype).unsqueeze(1) * far_gate.to(log_probs.dtype).unsqueeze(1)
    active_tokens = token_weight.sum()
    if active_tokens <= 0:
        return log_probs.sum() * 0.0
    loss_sum = (-log_probs * token_weight).sum()
    if normalize_by_active_tokens:
        return loss_sum / active_tokens
    return loss_sum / log_probs.shape[0]
