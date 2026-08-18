# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import math
import os
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_self_distillation_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.hybrid_routing import (
    compute_far_correct_sft_loss,
    compute_far_gate,
    compute_offpolicy_degree,
    compute_sdpo_route_mask,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.debug.sdpo_diagnostics import (
    append_record,
    iter_sketchable_named_parameters,
    masked_summary,
    multi_parameter_gradient_sketch,
    output_gradient_diagnostics,
    should_capture,
    write_records,
)
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "sdpo":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if teacher_regularization != "ema":
            return
        update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
        if update_rate == 0.0:
            return
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("EMA teacher requires a separate teacher_module in the actor worker.")
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_module.parameters(),
                self.actor_module.parameters(),
            ):
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")

        model = module or self.actor_module

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if compute_all_logps:
                    all_logps = full_all_logps[:, -response_length - 1 : -1, :]
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        cspd_topk = data.meta_info.get("cspd_topk")
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        topk_logps_lst = []
        topk_indices_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    distill_topk=cspd_topk,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])
            if cspd_topk is not None:
                topk_logps_lst.append(outputs["topk_logps"])
                topk_indices_lst.append(outputs["topk_indices"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)
        if cspd_topk is not None:
            cspd_topk_logps = torch.concat(topk_logps_lst, dim=0)
            cspd_topk_indices = torch.concat(topk_indices_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)
            if cspd_topk is not None:
                cspd_topk_logps = restore_dynamic_batch(cspd_topk_logps, batch_idx_list)
                cspd_topk_indices = restore_dynamic_batch(cspd_topk_indices, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        if cspd_topk is not None:
            outputs["cspd_topk_log_probs"] = cspd_topk_logps
            outputs["cspd_topk_indices"] = cspd_topk_indices
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        global_step = int(data.meta_info.get("global_steps", -1))
        cspd_raw_gae_std = float(data.meta_info.get("cspd_raw_gae_std", float("nan")))
        diagnostics_cfg = getattr(self.config, "sdpo_diagnostics", None)
        diagnostics_enabled = bool(
            diagnostics_cfg
            and diagnostics_cfg.get("enable", False)
            and global_step in set(diagnostics_cfg.get("capture_steps", []))
        )

        self_distillation_enabled = loss_mode == "sdpo"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        offpolicy_routing_cfg = getattr(self.config, "offpolicy_routing", None)
        far_correct_sft_cfg = getattr(self.config, "far_correct_sft", None)
        sdpo_routing_cfg = getattr(self.config, "sdpo_routing", None)
        kl_routing_cfg = getattr(self.config, "kl_routing", None)
        far_correct_sft_enabled = (
            not self_distillation_enabled
            and far_correct_sft_cfg is not None
            and far_correct_sft_cfg.get("enable", False)
        )
        sdpo_routing_enabled = (
            not self_distillation_enabled
            and sdpo_routing_cfg is not None
            and sdpo_routing_cfg.get("enable", False)
        )
        kl_routing_enabled = (
            not self_distillation_enabled
            and kl_routing_cfg is not None
            and kl_routing_cfg.get("enable", False)
        )
        if kl_routing_enabled and not self.config.use_kl_loss:
            raise ValueError("kl_routing.enable=True requires actor.use_kl_loss=True (ref policy).")
        if sdpo_routing_enabled and kl_routing_enabled:
            raise ValueError("sdpo_routing and kl_routing cannot both be enabled; pick one aux loss.")
        hybrid_loss_enabled = far_correct_sft_enabled or sdpo_routing_enabled or kl_routing_enabled
        if self_distillation_enabled or sdpo_routing_enabled or diagnostics_enabled:
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            }
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"
        if hybrid_loss_enabled:
            hybrid_required_keys = {
                "hybrid_success_mask",
                "hybrid_failed_mask",
                "hybrid_group_has_correct",
            }
            assert hybrid_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {hybrid_required_keys - set(data.batch.keys())}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled or sdpo_routing_enabled or diagnostics_enabled:
            select_keys.extend(list(self_distillation_required_keys))
            select_keys.extend(
                key
                for key in data.batch.keys()
                if key.startswith("diagnostic_") and key not in select_keys
            )
        if hybrid_loss_enabled:
            select_keys.extend(list(hybrid_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        for cspd_key in ("cspd_topk_indices", "cspd_target_probs", "cspd_value_weights", "cspd_selected_mask"):
            if cspd_key in data.batch.keys():
                select_keys.append(cspd_key)

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        if diagnostics_enabled:
            for key in ("uid", "data_source"):
                if key in data.non_tensor_batch.keys() and key not in non_tensor_select_keys:
                    non_tensor_select_keys.append(key)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        diagnostic_records = []
        diagnostic_gradient_records = []
        diagnostic_groups: set[str] = set()
        diagnostic_valid_gradient_count = 0
        diagnostic_gradient_budget = (
            int(diagnostics_cfg.get("gradient_max_records_per_step", 4)) if diagnostics_enabled else 0
        )
        diagnostic_gradient_min_norm = (
            float(diagnostics_cfg.get("gradient_min_norm", 1e-3)) if diagnostics_enabled else 0.0
        )
        did_update = False
        for ppo_epoch_index in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if (self_distillation_enabled or sdpo_routing_enabled or diagnostics_enabled) else None
                    if self_distillation_enabled or sdpo_routing_enabled or diagnostics_enabled:
                        assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for distillation"

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    if teacher_regularization == "trust-region" and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                    distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                    cspd_indices = model_inputs.get("cspd_topk_indices") if loss_mode == "cspd" else None
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                        topk_indices=cspd_indices,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    teacher_log_prob = None
                    teacher_all_logps = None
                    teacher_topk_logps = None
                    if self_distillation_enabled or sdpo_routing_enabled or diagnostics_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        # Hybrid M4 historically distills across reprompt contexts
                        # with the current actor; preserve that mechanism in probes.
                        teacher_model = (
                            self.actor_module
                            if sdpo_routing_enabled and not self_distillation_enabled
                            else self.teacher_module or self.actor_module
                        )
                        if teacher_regularization == "trust-region" and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                        with torch.no_grad():
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=teacher_model,
                            )
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                    sdpo_details = None
                    if self_distillation_enabled:
                        sdpo_result = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                            return_details=diagnostics_enabled,
                        )
                        if diagnostics_enabled:
                            pg_loss, pg_metrics, sdpo_details = sdpo_result
                        else:
                            pg_loss, pg_metrics = sdpo_result

                        pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                        micro_batch_metrics.update(pg_metrics)
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                            **({
                                "student_topk_log_probs": outputs["topk_logps"],
                                "cspd_target_probs": model_inputs["cspd_target_probs"],
                                "cspd_value_weights": model_inputs["cspd_value_weights"],
                                "cspd_selected_mask": model_inputs["cspd_selected_mask"],
                            } if loss_mode == "cspd" else {}),
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    far_gate = None
                    if offpolicy_routing_cfg is not None and (
                        offpolicy_routing_cfg.get("log_metrics", True)
                        or offpolicy_routing_cfg.get("enable", False)
                        or hybrid_loss_enabled
                    ):
                        with torch.no_grad():
                            offpolicy_degree = compute_offpolicy_degree(
                                current_log_probs=log_prob.detach(),
                                old_log_probs=old_log_prob.detach(),
                                response_mask=response_mask,
                                rollout_log_probs=rollout_log_prob.detach() if rollout_log_prob is not None else None,
                            )
                            far_gate = compute_far_gate(
                                offpolicy_degree=offpolicy_degree,
                                tau_far=offpolicy_routing_cfg.get("tau_far", 0.3),
                                gate_temp=offpolicy_routing_cfg.get("gate_temp", 0.1),
                                use_soft_gate=offpolicy_routing_cfg.get("use_soft_gate", True),
                            )
                        if offpolicy_routing_cfg.get("log_metrics", True):
                            micro_batch_metrics["hybrid_routing/offpolicy_degree_mean"] = offpolicy_degree.mean().item()
                            micro_batch_metrics["hybrid_routing/offpolicy_degree_max"] = offpolicy_degree.max().item()
                            micro_batch_metrics["hybrid_routing/far_gate_mean"] = far_gate.mean().item()
                            micro_batch_metrics["hybrid_routing/far_hard_fraction"] = (
                                offpolicy_degree > offpolicy_routing_cfg.get("tau_far", 0.3)
                            ).float().mean().item()
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if hybrid_loss_enabled:
                        if far_gate is None:
                            with torch.no_grad():
                                offpolicy_degree = compute_offpolicy_degree(
                                    current_log_probs=log_prob.detach(),
                                    old_log_probs=old_log_prob.detach(),
                                    response_mask=response_mask,
                                    rollout_log_probs=rollout_log_prob.detach() if rollout_log_prob is not None else None,
                                )
                                far_gate = compute_far_gate(
                                    offpolicy_degree=offpolicy_degree,
                                    tau_far=offpolicy_routing_cfg.get("tau_far", 0.3),
                                    gate_temp=offpolicy_routing_cfg.get("gate_temp", 0.1),
                                    use_soft_gate=offpolicy_routing_cfg.get("use_soft_gate", True),
                                )
                        success_mask = model_inputs["hybrid_success_mask"].bool()
                        failed_mask = model_inputs["hybrid_failed_mask"].bool()
                        group_has_correct = model_inputs["hybrid_group_has_correct"].bool()
                        if far_correct_sft_enabled:
                            sft_loss = compute_far_correct_sft_loss(
                                log_probs=log_prob,
                                response_mask=response_mask,
                                success_mask=success_mask,
                                far_gate=far_gate,
                                normalize_by_active_tokens=far_correct_sft_cfg.get(
                                    "normalize_by_active_tokens", True
                                ),
                            )
                            lambda_sft = far_correct_sft_cfg.get("lambda_sft", 0.05)
                            policy_loss = policy_loss + lambda_sft * sft_loss
                            micro_batch_metrics["hybrid_routing/far_correct_sft_loss"] = sft_loss.detach().item()
                            micro_batch_metrics["hybrid_routing/far_correct_sft_weight"] = lambda_sft
                        if sdpo_routing_enabled:
                            teacher_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["teacher_input_ids"],
                                "attention_mask": model_inputs["teacher_attention_mask"],
                                "position_ids": model_inputs["teacher_position_ids"],
                            }
                            teacher_model = self.actor_module
                            with torch.no_grad():
                                teacher_outputs = self._forward_micro_batch(
                                    teacher_inputs,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    return_all_logps=return_all_logps,
                                    distill_topk=distill_topk,
                                    topk_indices=student_topk_indices,
                                    module=teacher_model,
                                )
                            teacher_log_prob = teacher_outputs["log_probs"]
                            teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                            teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                            sdpo_route_mask = compute_sdpo_route_mask(
                                failed_mask=failed_mask,
                                group_has_success=group_has_correct,
                                far_gate=far_gate,
                                mode=sdpo_routing_cfg.get("mode", "none"),
                            )
                            sdpo_loss, sdpo_metrics = compute_self_distillation_loss(
                                student_log_probs=log_prob,
                                teacher_log_probs=teacher_log_prob,
                                response_mask=response_mask,
                                self_distillation_config=self_distillation_cfg,
                                old_log_probs=old_log_prob,
                                student_all_log_probs=student_all_logps,
                                teacher_all_log_probs=teacher_all_logps,
                                student_topk_log_probs=student_topk_logps,
                                teacher_topk_log_probs=teacher_topk_logps,
                                self_distillation_mask=self_distillation_mask,
                                sdpo_route_mask=sdpo_route_mask,
                                loss_agg_mode=loss_agg_mode,
                                rollout_is_weights=rollout_is_weights,
                            )
                            policy_loss = policy_loss + sdpo_loss
                            micro_batch_metrics["hybrid_routing/sdpo_aux_loss"] = sdpo_loss.detach().item()
                            micro_batch_metrics["hybrid_routing/sdpo_route_weight_mean"] = (
                                sdpo_route_mask.detach().float().mean().item()
                            )
                            micro_batch_metrics.update(sdpo_metrics)
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss_mask = response_mask
                        kl_coef = self.config.kl_loss_coef
                        if kl_routing_enabled:
                            if far_gate is None:
                                with torch.no_grad():
                                    offpolicy_degree = compute_offpolicy_degree(
                                        current_log_probs=log_prob.detach(),
                                        old_log_probs=old_log_prob.detach(),
                                        response_mask=response_mask,
                                        rollout_log_probs=rollout_log_prob.detach()
                                        if rollout_log_prob is not None
                                        else None,
                                    )
                                    far_gate = compute_far_gate(
                                        offpolicy_degree=offpolicy_degree,
                                        tau_far=offpolicy_routing_cfg.get("tau_far", 0.3),
                                        gate_temp=offpolicy_routing_cfg.get("gate_temp", 0.1),
                                        use_soft_gate=offpolicy_routing_cfg.get("use_soft_gate", True),
                                    )
                            success_mask = model_inputs["hybrid_success_mask"].bool()
                            failed_mask = model_inputs["hybrid_failed_mask"].bool()
                            group_has_correct = model_inputs["hybrid_group_has_correct"].bool()
                            kl_route_mask = compute_sdpo_route_mask(
                                failed_mask=failed_mask,
                                group_has_success=group_has_correct,
                                far_gate=far_gate,
                                mode=kl_routing_cfg.get("mode", "none"),
                            )
                            kl_loss_mask = response_mask * kl_route_mask.unsqueeze(1).to(response_mask.dtype)
                            kl_coef = float(kl_routing_cfg.get("lambda_kl", 1.0))
                            micro_batch_metrics["hybrid_routing/kl_route_weight_mean"] = (
                                kl_route_mask.detach().float().mean().item()
                            )
                            micro_batch_metrics["hybrid_routing/kl_lambda"] = kl_coef
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=kl_loss_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * kl_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = kl_coef

                    if should_capture(diagnostics_cfg, global_step, ppo_epoch_index):
                        vanilla_loss_fn = get_policy_loss_fn("vanilla")
                        grpo_probe_loss, _ = vanilla_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        if sdpo_details is None:
                            sdpo_probe_loss, _, sdpo_details = compute_self_distillation_loss(
                                student_log_probs=log_prob,
                                teacher_log_probs=teacher_log_prob,
                                response_mask=response_mask,
                                self_distillation_config=self_distillation_cfg,
                                old_log_probs=old_log_prob,
                                student_all_log_probs=student_all_logps,
                                teacher_all_log_probs=teacher_all_logps,
                                student_topk_log_probs=student_topk_logps,
                                teacher_topk_log_probs=teacher_topk_logps,
                                self_distillation_mask=self_distillation_mask,
                                loss_agg_mode=loss_agg_mode,
                                rollout_is_weights=rollout_is_weights,
                                return_details=True,
                            )
                        else:
                            sdpo_probe_loss = pg_loss

                        output_diagnostics = output_gradient_diagnostics(
                            sdpo_loss=sdpo_probe_loss,
                            grpo_loss=grpo_probe_loss,
                            student_distill_log_probs=sdpo_details["student_distill_log_probs"],
                            student_log_probs=log_prob,
                            student_topk_indices=student_topk_indices,
                            responses=model_inputs["responses"],
                            loss_mask=sdpo_details["loss_mask"],
                        )
                        diagnostic_mask = sdpo_details["loss_mask"].bool()
                        delta_summary = masked_summary(
                            sdpo_details["selected_token_logprob_delta"], diagnostic_mask
                        )
                        cosine_summary = masked_summary(
                            output_diagnostics["output_update_cosine"], diagnostic_mask
                        )
                        micro_batch_metrics.update(
                            {
                                "sdpo_diagnostics/delta_mean": delta_summary["mean"],
                                "sdpo_diagnostics/delta_abs_p95": delta_summary["p95_abs"],
                                "sdpo_diagnostics/output_gradient_cosine": cosine_summary["mean"],
                                "sdpo_diagnostics/conflict_rate": (
                                    output_diagnostics["sampled_pressure_conflict"][diagnostic_mask]
                                    .float()
                                    .mean()
                                    .item()
                                    if diagnostic_mask.any()
                                    else 0.0
                                ),
                                "sdpo_diagnostics/teacher_tail_mass": (
                                    sdpo_details["teacher_tail_mass"][diagnostic_mask].mean().item()
                                    if diagnostic_mask.any()
                                    else 0.0
                                ),
                            }
                        )

                        if (
                            diagnostics_cfg.get("gradient_compare", True)
                            and diagnostic_valid_gradient_count < diagnostic_gradient_budget
                        ):
                            # Filter before autograd.grad: FSDP empty-storage placeholders
                            # raise setStorage and can break the real loss.backward().
                            sketch_parameters = iter_sketchable_named_parameters(
                                self.actor_module.named_parameters()
                            )
                            adv_abs_mean = advantages.detach().float().abs().mean().item()
                            if not sketch_parameters:
                                logger.warning(
                                    "SDPO parameter gradient sketch skipped: no locally materialized parameters"
                                )
                                # Exhaust budget so we do not retry every microbatch uselessly.
                                diagnostic_valid_gradient_count = diagnostic_gradient_budget
                            elif adv_abs_mean < diagnostic_gradient_min_norm:
                                # Flat GRPO advantages → zero GRPO grad; try a later microbatch.
                                micro_batch_metrics["sdpo_diagnostics/gradient_skip_flat_adv"] = 1.0
                            else:
                                try:
                                    probe_losses = {
                                        "sdpo": sdpo_probe_loss,
                                        "grpo": grpo_probe_loss,
                                    }
                                    if hybrid_loss_enabled:
                                        probe_losses["m4_actual"] = policy_loss
                                    gradient_record = multi_parameter_gradient_sketch(
                                        losses=probe_losses,
                                        named_parameters=sketch_parameters,
                                        dim=int(diagnostics_cfg.get("gradient_sketch_dim", 2048)),
                                    )
                                    norms = gradient_record.get("norms", {})
                                    sdpo_norm = float(norms.get("sdpo", 0.0))
                                    grpo_norm = float(norms.get("grpo", 0.0))
                                    gradient_record.update(
                                        {
                                            "global_step": global_step,
                                            "ppo_epoch_index": ppo_epoch_index,
                                            "mini_batch_index": batch_idx,
                                            "advantage_abs_mean": adv_abs_mean,
                                            "valid": (
                                                not gradient_record.get("skipped")
                                                and "sdpo__grpo" in gradient_record.get("pairs", {})
                                                and sdpo_norm >= diagnostic_gradient_min_norm
                                                and grpo_norm >= diagnostic_gradient_min_norm
                                            ),
                                            "rank": torch.distributed.get_rank()
                                            if torch.distributed.is_initialized()
                                            else 0,
                                        }
                                    )
                                    if not gradient_record["valid"]:
                                        micro_batch_metrics["sdpo_diagnostics/gradient_skip_low_norm"] = 1.0
                                        micro_batch_metrics["sdpo_diagnostics/grpo_grad_norm"] = grpo_norm
                                        micro_batch_metrics["sdpo_diagnostics/sdpo_grad_norm"] = sdpo_norm
                                    else:
                                        diagnostic_gradient_records.append(gradient_record)
                                        diagnostic_valid_gradient_count += 1
                                        micro_batch_metrics["sdpo_diagnostics/parameter_gradient_cosine"] = (
                                            gradient_record["pairs"]["sdpo__grpo"]["sketch_cosine"]
                                        )
                                        micro_batch_metrics["sdpo_diagnostics/grpo_grad_norm"] = grpo_norm
                                        micro_batch_metrics["sdpo_diagnostics/sdpo_grad_norm"] = sdpo_norm
                                        micro_batch_metrics["sdpo_diagnostics/valid_gradient_count"] = float(
                                            diagnostic_valid_gradient_count
                                        )
                                        if hybrid_loss_enabled:
                                            micro_batch_metrics[
                                                "sdpo_diagnostics/m4_vs_grpo_gradient_cosine"
                                            ] = gradient_record["pairs"]["grpo__m4_actual"]["sketch_cosine"]
                                            micro_batch_metrics[
                                                "sdpo_diagnostics/m4_vs_sdpo_gradient_cosine"
                                            ] = gradient_record["pairs"]["sdpo__m4_actual"]["sketch_cosine"]
                                except RuntimeError as error:
                                    logger.warning("SDPO parameter gradient sketch failed: %s", error)

                        raw_uids = model_inputs.get("uid")
                        if raw_uids is None:
                            raw_uids = [f"rank-local-{len(diagnostic_records) + i}" for i in range(response_mask.shape[0])]
                        raw_sources = model_inputs.get("data_source")
                        if raw_sources is None:
                            raw_sources = ["unknown"] * response_mask.shape[0]
                        max_groups = int(diagnostics_cfg.get("max_prompt_groups", 8))
                        student_distribution = sdpo_details["student_distill_log_probs"]
                        teacher_distribution = sdpo_details["teacher_distill_log_probs"]
                        student_entropy = -(student_distribution.exp() * student_distribution).sum(-1)
                        teacher_entropy = -(teacher_distribution.exp() * teacher_distribution).sum(-1)
                        top1_agreement = student_distribution.argmax(-1).eq(teacher_distribution.argmax(-1))
                        for sample_idx, raw_uid in enumerate(raw_uids):
                            uid = str(raw_uid)
                            if uid not in diagnostic_groups and len(diagnostic_groups) >= max_groups:
                                continue
                            diagnostic_groups.add(uid)
                            tensor_record = {
                                "token_ids": model_inputs["responses"][sample_idx],
                                "response_mask": response_mask[sample_idx],
                                "old_log_probs": old_log_prob[sample_idx],
                                "advantages": advantages[sample_idx],
                                "student_log_probs": log_prob[sample_idx],
                                "teacher_log_probs": teacher_log_prob[sample_idx],
                                "logprob_delta": sdpo_details["selected_token_logprob_delta"][sample_idx],
                                "sdpo_per_token_loss": sdpo_details["per_token_loss"][sample_idx],
                                "student_entropy_topk_tail": student_entropy[sample_idx],
                                "teacher_entropy_on_student_topk_tail": teacher_entropy[sample_idx],
                                "top1_bucket_agreement": top1_agreement[sample_idx],
                                "student_topk_mass": sdpo_details["student_topk_mass"][sample_idx],
                                "teacher_mass_on_student_topk": sdpo_details["teacher_mass_on_student_topk"][sample_idx],
                                "student_tail_mass": sdpo_details["student_tail_mass"][sample_idx],
                                "teacher_tail_mass": sdpo_details["teacher_tail_mass"][sample_idx],
                            }
                            tensor_record.update(
                                {key: value[sample_idx] for key, value in output_diagnostics.items()}
                            )
                            for key, value in model_inputs.items():
                                if key.startswith("diagnostic_") and torch.is_tensor(value):
                                    tensor_record[key] = value[sample_idx]
                            append_record(
                                diagnostic_records,
                                tensor_record,
                                {
                                    "uid": uid,
                                    "data_source": str(raw_sources[sample_idx]),
                                    "global_step": global_step,
                                    "ppo_epoch_index": ppo_epoch_index,
                                    "actual_objective": "m4_hard"
                                    if sdpo_routing_enabled
                                    and not offpolicy_routing_cfg.get("use_soft_gate", True)
                                    else loss_mode,
                                    "teacher_mode": "current_actor_context"
                                    if sdpo_routing_enabled and not self_distillation_enabled
                                    else teacher_regularization,
                                },
                            )

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if (
                    loss_mode == "cspd"
                    and torch.isfinite(grad_norm).item()
                    and math.isfinite(cspd_raw_gae_std)
                    and cspd_raw_gae_std > 0
                ):
                    normalized_norm = grad_norm.detach().item() / cspd_raw_gae_std
                    mini_batch_metrics.update(
                        {
                            "actor/cspd_grad_norm_if_raw_gae_std_scaled": normalized_norm,
                            "actor/cspd_grad_norm_after_clip_if_raw_gae_std_scaled": min(
                                normalized_norm, float(self.config.grad_clip)
                            ),
                        }
                    )
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if did_update:
            self._update_teacher()
        if diagnostics_enabled and diagnostics_cfg.get("output_dir"):
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            write_records(
                records=diagnostic_records,
                gradient_records=diagnostic_gradient_records,
                output_dir=diagnostics_cfg.output_dir,
                global_step=global_step,
                rank=rank,
            )
        return metrics
