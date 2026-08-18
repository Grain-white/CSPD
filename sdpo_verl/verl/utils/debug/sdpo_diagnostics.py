"""Opt-in artifacts for SDPO/GRPO mechanism probes.

This module is deliberately independent from SwanLab.  Actor ranks write
compressed arrays for a small deterministic subset; only aggregate scalars are
returned through the normal trainer metrics path.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist


def should_capture(config: Any, global_step: int, ppo_epoch: int) -> bool:
    if config is None or not config.get("enable", False):
        return False
    if global_step not in set(config.get("capture_steps", [])):
        return False
    return bool(config.get("capture_all_ppo_epochs", True) or ppo_epoch == 0)


def masked_summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected = values.detach().float()[mask.bool()]
    if selected.numel() == 0:
        return {"mean": 0.0, "abs_mean": 0.0, "p95_abs": 0.0}
    return {
        "mean": selected.mean().item(),
        "abs_mean": selected.abs().mean().item(),
        "p95_abs": torch.quantile(selected.abs(), 0.95).item(),
    }


def output_gradient_diagnostics(
    *,
    sdpo_loss: torch.Tensor,
    grpo_loss: torch.Tensor,
    student_distill_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    student_topk_indices: torch.Tensor | None,
    responses: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare update vectors in the SDPO top-k+tail output space.

    Positive ``*_pressure`` means gradient descent raises that coordinate.
    GRPO is sparse in this space: its sampled-action coefficient is placed in
    the matching student-top-k bucket, or in the tail bucket when absent.
    """

    sdpo_grad = torch.autograd.grad(
        sdpo_loss,
        student_distill_log_probs,
        retain_graph=True,
        allow_unused=False,
    )[0]
    sampled_grpo_grad = torch.autograd.grad(
        grpo_loss,
        student_log_probs,
        retain_graph=True,
        allow_unused=False,
    )[0]
    sdpo_pressure = -sdpo_grad
    grpo_pressure = torch.zeros_like(sdpo_pressure)

    if student_topk_indices is None:
        sampled_bucket = responses.unsqueeze(-1)
        sampled_bucket = sampled_bucket.clamp(max=student_distill_log_probs.shape[-1] - 1)
    else:
        matches = student_topk_indices.eq(responses.unsqueeze(-1))
        found = matches.any(-1)
        sampled_bucket = matches.float().argmax(-1)
        tail_bucket = student_distill_log_probs.shape[-1] - 1
        sampled_bucket = torch.where(found, sampled_bucket, torch.full_like(sampled_bucket, tail_bucket))
    grpo_pressure.scatter_(-1, sampled_bucket.unsqueeze(-1), (-sampled_grpo_grad).unsqueeze(-1))

    dot = (sdpo_pressure * grpo_pressure).sum(-1)
    sdpo_norm = sdpo_pressure.square().sum(-1).sqrt()
    grpo_norm = grpo_pressure.square().sum(-1).sqrt()
    cosine = dot / (sdpo_norm * grpo_norm).clamp(min=1e-12)
    sampled_sdpo_pressure = sdpo_pressure.gather(-1, sampled_bucket.unsqueeze(-1)).squeeze(-1)
    sampled_grpo_pressure = -sampled_grpo_grad
    sign_conflict = (sampled_sdpo_pressure * sampled_grpo_pressure < 0).float()

    return {
        "sdpo_output_pressure_norm": sdpo_norm,
        "grpo_output_pressure_norm": grpo_norm,
        "output_update_cosine": cosine,
        "sampled_sdpo_pressure": sampled_sdpo_pressure,
        "sampled_grpo_pressure": sampled_grpo_pressure,
        "sampled_pressure_conflict": sign_conflict,
        "sampled_bucket": sampled_bucket,
    }


def iter_sketchable_named_parameters(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> list[tuple[str, torch.nn.Parameter]]:
    """Keep only locally materialized trainable params.

    Under FSDP FULL_SHARD, ``named_parameters()`` still exposes full logical shapes
    whose storage is empty until the parameter is unsharded.  Calling
    ``torch.autograd.grad`` on those placeholders raises ``setStorage`` and can
    corrupt the subsequent real ``loss.backward()``.  FlatParameter local shards
    (and single-GPU full params) pass this filter.
    """

    selected: list[tuple[str, torch.nn.Parameter]] = []
    seen: set[tuple[int, int, int]] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        try:
            storage = parameter.untyped_storage()
            storage_nbytes = int(storage.size())
            needed = int(parameter.numel()) * int(parameter.element_size())
            data_ptr = int(storage.data_ptr()) if storage_nbytes > 0 else 0
            storage_offset = int(parameter.storage_offset())
        except Exception:
            continue
        if storage_nbytes <= 0 or needed <= 0 or needed > storage_nbytes:
            continue
        key = (data_ptr, storage_offset, needed)
        if key in seen:
            continue
        seen.add(key)
        selected.append((name, parameter))
    return selected


def _count_sketch(
    grads: Iterable[torch.Tensor | None],
    dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sketch = torch.zeros(dim, dtype=torch.float32, device=device)
    norm_sq = torch.zeros((), dtype=torch.float64, device=device)
    offset = 0
    for grad in grads:
        if grad is None:
            continue
        flat = grad.detach().float().reshape(-1)
        norm_sq += flat.double().square().sum()
        chunk_size = 1_000_000
        for start in range(0, flat.numel(), chunk_size):
            stop = min(start + chunk_size, flat.numel())
            indices = torch.arange(start, stop, device=device, dtype=torch.int64) + offset
            buckets = torch.remainder(indices * 2654435761, dim)
            signs = torch.where(torch.remainder(indices * 2246822519, 2) == 0, 1.0, -1.0)
            sketch.scatter_add_(0, buckets, flat[start:stop] * signs)
        offset += flat.numel()
    return sketch, norm_sq


def parameter_gradient_sketch(
    *,
    sdpo_loss: torch.Tensor,
    grpo_loss: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    dim: int,
) -> dict[str, float | list[float]]:
    result = multi_parameter_gradient_sketch(
        losses={"sdpo": sdpo_loss, "grpo": grpo_loss},
        named_parameters=named_parameters,
        dim=dim,
    )
    pair = result["pairs"]["sdpo__grpo"]
    return {
        "parameter_sketch_cosine": pair["sketch_cosine"],
        "parameter_sketch_dot": pair["sketch_dot"],
        "sdpo_parameter_grad_norm": result["norms"]["sdpo"],
        "grpo_parameter_grad_norm": result["norms"]["grpo"],
        "final_norm_lm_head_exact_cosine": pair["exact_subset_cosine"],
        "sdpo_sketch": result["sketches"]["sdpo"],
        "grpo_sketch": result["sketches"]["grpo"],
    }


def multi_parameter_gradient_sketch(
    *,
    losses: dict[str, torch.Tensor],
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    dim: int,
) -> dict[str, Any]:
    """Sketch several objective gradients once and report all pairwise angles."""

    active_named = iter_sketchable_named_parameters(named_parameters)
    active = [parameter for _, parameter in active_named]
    device = next(iter(losses.values())).device
    zero_sketch = [0.0] * dim
    if not active:
        return {
            "norms": {label: 0.0 for label in losses},
            "sketches": {label: list(zero_sketch) for label in losses},
            "pairs": {},
            "skipped": True,
            "num_parameters": 0,
        }

    sketches: dict[str, torch.Tensor] = {}
    norm_squares: dict[str, torch.Tensor] = {}
    exact_grads: dict[str, dict[int, torch.Tensor]] = {}
    for label, loss in losses.items():
        grads = torch.autograd.grad(loss, active, retain_graph=True, allow_unused=True)
        sketches[label], norm_squares[label] = _count_sketch(grads, dim, device)
        exact_grads[label] = {
            index: grad.detach().float().clone()
            for index, ((name, parameter), grad) in enumerate(zip(active_named, grads, strict=True))
            if grad is not None
            and parameter.numel() <= 2_000_000
            and ("lm_head" in name or "norm" in name)
        }
        del grads
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sketches[label])
            dist.all_reduce(norm_squares[label])

    pairs: dict[str, dict[str, float]] = {}
    labels = list(losses)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            sketch_dot = torch.dot(sketches[left], sketches[right])
            sketch_cosine = sketch_dot / (
                sketches[left].norm() * sketches[right].norm()
            ).clamp(min=1e-12)
            exact_dot = torch.zeros((), dtype=torch.float64, device=device)
            exact_left_norm_sq = torch.zeros((), dtype=torch.float64, device=device)
            exact_right_norm_sq = torch.zeros((), dtype=torch.float64, device=device)
            common = exact_grads[left].keys() & exact_grads[right].keys()
            for index in common:
                left_grad = exact_grads[left][index].double()
                right_grad = exact_grads[right][index].double()
                exact_dot += (left_grad * right_grad).sum()
                exact_left_norm_sq += left_grad.square().sum()
                exact_right_norm_sq += right_grad.square().sum()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(exact_dot)
                dist.all_reduce(exact_left_norm_sq)
                dist.all_reduce(exact_right_norm_sq)
            exact_cosine = exact_dot / (
                exact_left_norm_sq.sqrt() * exact_right_norm_sq.sqrt()
            ).clamp(min=1e-12)
            pairs[f"{left}__{right}"] = {
                "sketch_cosine": sketch_cosine.item(),
                "sketch_dot": sketch_dot.item(),
                "exact_subset_cosine": exact_cosine.item(),
            }
    return {
        "norms": {
            label: math.sqrt(max(norm_square.item(), 0.0))
            for label, norm_square in norm_squares.items()
        },
        "sketches": {label: sketch.cpu().tolist() for label, sketch in sketches.items()},
        "pairs": pairs,
        "skipped": False,
        "num_parameters": len(active_named),
    }


def append_record(records: list[dict[str, Any]], tensors: dict[str, Any], metadata: dict[str, Any]) -> None:
    record: dict[str, Any] = {"metadata": metadata}
    for key, value in tensors.items():
        if torch.is_tensor(value):
            record[key] = value.detach().cpu()
        else:
            record[key] = value
    records.append(record)


def write_records(
    *,
    records: list[dict[str, Any]],
    gradient_records: list[dict[str, Any]],
    output_dir: str,
    global_step: int,
    rank: int,
) -> tuple[str, str]:
    """Write one torch artifact and one JSON gradient summary per actor rank."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    token_path = directory / f"step_{global_step:04d}_rank_{rank:03d}.pt"
    gradient_path = directory / f"step_{global_step:04d}_rank_{rank:03d}_gradients.json"
    torch.save(records, token_path)
    with gradient_path.open("w") as handle:
        json.dump(gradient_records, handle)
    return os.fspath(token_path), os.fspath(gradient_path)
