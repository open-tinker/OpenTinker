#!/usr/bin/env python3
# Copyright 2025 OpenTinker
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
"""OPSD Worker Extension: patches ActorRolloutRefWorker with OPSD RPCs.

Importing this module monkey-patches `verl.workers.fsdp_workers.ActorRolloutRefWorker`
to add:
  - `compute_opsd_vocab_log_probs` (legacy, full-vocab debug path)
  - `compute_opsd_jsd_per_token` (primary path for memory-safe OPSD)

The patch is idempotent (safe to import multiple times) and does NOT require any
config flag — the methods are always present on the class; they will simply never
be called when OPSD is disabled.

This avoids the previous subclass approach, which broke because
`build_role_worker_mapping` cannot add the subclass unless `use_opsd=True` reaches it,
but verl's structured OmegaConf config rejects unknown keys in the algorithm block.
"""

import math
import torch
import logging

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.profiler import log_gpu_memory_usage
from opentinker.server.opsd_config import validate_opsd_modes

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Full-vocab log-prob computation (standalone helper)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_vocab_log_probs(actor, data: DataProto) -> torch.Tensor:
    """Run the actor forward pass and return log-softmax over full vocab.

    Args:
        actor: DataParallelPPOActor instance (actor.actor_module is the FSDP model).
        data: DataProto containing input_ids, attention_mask, position_ids, responses.

    Returns:
        Tensor of shape [batch, response_len, vocab_size] on CPU, dtype=float32.
    """
    from verl.utils.device import get_device_id, get_device_name
    from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch

    actor.actor_module.eval()

    micro_batch_size = data.meta_info.get("micro_batch_size", None)
    if micro_batch_size is None or micro_batch_size <= 0:
        # For OPSD full-vocab paths, defaulting to 1 avoids catastrophic OOM when
        # log_prob_micro_batch_size_per_gpu is not configured.
        micro_batch_size = 1
        data.meta_info["micro_batch_size"] = micro_batch_size
    temperature = data.meta_info["temperature"]
    use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

    select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
    data = data.select(batch_keys=select_keys, non_tensor_batch_keys=[])

    if use_dynamic_bsz:
        max_token_len = data.meta_info.get("max_token_len", 4096) * getattr(actor, "ulysses_sequence_parallel_size", 1)
        micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
    else:
        micro_batches = data.split(micro_batch_size)

    device_name = get_device_name()
    all_vocab_log_probs = []

    for micro_batch in micro_batches:
        micro_batch = micro_batch.to(get_device_id())
        model_inputs = {**micro_batch.batch}
        responses = model_inputs["responses"]            # [mb, response_len]
        response_length = responses.size(-1)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        position_ids = model_inputs["position_ids"]

        with torch.no_grad():
            with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                output = actor.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )

            # logits: [mb, full_seq_len, vocab]
            # response positions: last response_length positions in the shifted sequence
            logits = output.logits  # [mb, seq_len, vocab]
            logits = logits[:, -response_length - 1 : -1, :]  # [mb, response_len, vocab]
            temp = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)
            logits = logits / temp

            # Keep logits dtype to avoid fp32 blow-up on [B, R, V].
            vocab_log_probs = torch.log_softmax(logits, dim=-1)  # [mb, response_len, vocab]

        all_vocab_log_probs.append(vocab_log_probs.cpu())

    result = torch.cat(all_vocab_log_probs, dim=0)  # [batch, response_len, vocab]

    if use_dynamic_bsz:
        result = restore_dynamic_batch(result, batch_idx_list)

    return result


def _compute_opsd_distill_stats(
    student_actor,
    teacher_actor,
    student_data: DataProto,
    teacher_data: DataProto,
    beta: float,
    loss_mode: str,
    vocab_chunk_size: int = 4096,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Compute OPSD distillation stats for full-JSD or sampled-KL modes.

    This function avoids materializing/storing full-vocab log-probs [B, R, V] on CPU
    and avoids sending those tensors through Ray. It computes exact JSD in streaming
    chunks over the vocab dimension (for full_jsd mode), and always returns sampled
    token log-probs for student/teacher.
    """
    from verl.utils.device import get_device_id, get_device_name

    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    if vocab_chunk_size <= 0:
        raise ValueError(f"vocab_chunk_size must be > 0, got {vocab_chunk_size}")
    if loss_mode not in {"full_jsd", "sampled_kl"}:
        raise ValueError(
            f"loss_mode must be one of ['full_jsd', 'sampled_kl'], got {loss_mode!r}"
        )

    student_actor.actor_module.eval()
    teacher_actor.actor_module.eval()

    student_select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
    teacher_select_keys = ["input_ids", "attention_mask", "position_ids"]
    student_data = student_data.select(batch_keys=student_select_keys, non_tensor_batch_keys=[])
    teacher_data = teacher_data.select(batch_keys=teacher_select_keys, non_tensor_batch_keys=[])

    if len(student_data) != len(teacher_data):
        raise ValueError(
            f"student and teacher batch size mismatch: {len(student_data)} vs {len(teacher_data)}"
        )

    micro_batch_size = student_data.meta_info.get("micro_batch_size", None)
    if micro_batch_size is None or micro_batch_size <= 0:
        micro_batch_size = 1
        student_data.meta_info["micro_batch_size"] = micro_batch_size
    teacher_data.meta_info["micro_batch_size"] = micro_batch_size

    temperature = student_data.meta_info["temperature"]
    student_mbs = student_data.split(micro_batch_size)
    teacher_mbs = teacher_data.split(micro_batch_size)
    if len(student_mbs) != len(teacher_mbs):
        raise RuntimeError(
            f"student/teacher split mismatch: {len(student_mbs)} vs {len(teacher_mbs)}"
        )

    log_beta = math.log(beta)
    log_one_minus_beta = math.log(1.0 - beta)
    device_name = get_device_name()

    all_jsd_per_token = [] if loss_mode == "full_jsd" else None
    all_student_sampled_log_probs = []
    all_teacher_sampled_log_probs = []

    for student_mb, teacher_mb in zip(student_mbs, teacher_mbs, strict=True):
        student_mb = student_mb.to(get_device_id())
        teacher_mb = teacher_mb.to(get_device_id())

        student_inputs = {**student_mb.batch}
        teacher_inputs = {**teacher_mb.batch}

        responses = student_inputs["responses"]  # [mb, response_len]
        response_length = responses.size(-1)

        with torch.no_grad():
            with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                student_out = student_actor.actor_module(
                    input_ids=student_inputs["input_ids"],
                    attention_mask=student_inputs["attention_mask"],
                    position_ids=student_inputs["position_ids"],
                    use_cache=False,
                )
                teacher_out = teacher_actor.actor_module(
                    input_ids=teacher_inputs["input_ids"],
                    attention_mask=teacher_inputs["attention_mask"],
                    position_ids=teacher_inputs["position_ids"],
                    use_cache=False,
                )

            student_logits = student_out.logits[:, -response_length - 1 : -1, :]  # [mb, response_len, vocab]
            teacher_logits = teacher_out.logits[:, -response_length - 1 : -1, :]  # [mb, response_len, vocab]
            temp = torch.as_tensor(temperature, dtype=student_logits.dtype, device=student_logits.device)
            student_logits = student_logits / temp
            teacher_logits = teacher_logits / temp

            mb, resp_len, vocab_size = student_logits.shape
            student_flat = student_logits.reshape(-1, vocab_size)
            teacher_flat = teacher_logits.reshape(-1, vocab_size)

            # log p(token*) for sampled rollout tokens: logit(token*) - logsumexp(logits)
            sampled_token_ids = responses.reshape(-1, 1).to(student_flat.device)
            student_lse = torch.logsumexp(student_flat.float(), dim=-1, keepdim=True)
            teacher_lse = torch.logsumexp(teacher_flat.float(), dim=-1, keepdim=True)
            sampled_log_probs = (
                student_flat.gather(dim=-1, index=sampled_token_ids).float().squeeze(-1) - student_lse.squeeze(-1)
            )  # [mb * response_len]
            teacher_sampled_log_probs = (
                teacher_flat.gather(dim=-1, index=sampled_token_ids).float().squeeze(-1)
                - teacher_lse.squeeze(-1)
            )  # [mb * response_len]

            if loss_mode == "full_jsd":
                # Exact JSD with chunked vocab accumulation:
                # JSD_beta = beta * KL(p_t || m) + (1-beta) * KL(p_s || m)
                # m = beta * p_t + (1-beta) * p_s
                kl_teacher = torch.zeros(
                    (student_flat.shape[0],),
                    device=student_flat.device,
                    dtype=torch.float32,
                )
                kl_student = torch.zeros_like(kl_teacher)
                for start in range(0, vocab_size, vocab_chunk_size):
                    end = min(start + vocab_chunk_size, vocab_size)

                    student_logp_chunk = student_flat[:, start:end].float() - student_lse
                    teacher_logp_chunk = teacher_flat[:, start:end].float() - teacher_lse

                    log_m_chunk = torch.logaddexp(
                        teacher_logp_chunk + log_beta,
                        student_logp_chunk + log_one_minus_beta,
                    )

                    kl_teacher += (
                        teacher_logp_chunk.exp() * (teacher_logp_chunk - log_m_chunk)
                    ).sum(dim=-1)
                    kl_student += (
                        student_logp_chunk.exp() * (student_logp_chunk - log_m_chunk)
                    ).sum(dim=-1)

                jsd_per_token = (
                    beta * kl_teacher + (1.0 - beta) * kl_student
                ).view(mb, resp_len)  # [mb, response_len]
            sampled_log_probs = sampled_log_probs.view(mb, resp_len)  # [mb, response_len]
            teacher_sampled_log_probs = teacher_sampled_log_probs.view(
                mb, resp_len
            )  # [mb, response_len]

        if loss_mode == "full_jsd":
            all_jsd_per_token.append(jsd_per_token.cpu())
        all_student_sampled_log_probs.append(sampled_log_probs.cpu())
        all_teacher_sampled_log_probs.append(teacher_sampled_log_probs.cpu())

    jsd_out = (
        torch.cat(all_jsd_per_token, dim=0)
        if loss_mode == "full_jsd"
        else None
    )
    return (
        jsd_out,
        torch.cat(all_student_sampled_log_probs, dim=0),
        torch.cat(all_teacher_sampled_log_probs, dim=0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Monkey-patch: add compute_opsd_vocab_log_probs to ActorRolloutRefWorker
# ──────────────────────────────────────────────────────────────────────────────

def _compute_opsd_vocab_log_probs(self, data: DataProto) -> DataProto:
    """Compute full-vocabulary log-probabilities for OPSD.

    Args:
        data: DataProto with keys:
            - input_ids: [batch, prompt_len + response_len]  (teacher OR student prompt + rollout)
            - attention_mask: [batch, prompt_len + response_len]
            - position_ids: [batch, prompt_len + response_len]
            - responses: [batch, response_len]  (the student rollout tokens)
            Temperature and micro_batch_size are read from data.meta_info.

    Returns:
        DataProto with key ``opsd_vocab_log_probs``:
            shape [batch, response_len, vocab_size], dtype=float32 (log-softmax).
    """
    assert self._is_actor, "compute_opsd_vocab_log_probs requires actor role"
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load actor for OPSD vocab log probs", logger=logger)

    data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
    data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
    data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
    if "temperature" not in data.meta_info:
        data.meta_info["temperature"] = self.config.rollout.temperature

    with self.ulysses_sharding_manager:
        vocab_log_probs = _compute_vocab_log_probs(
            actor=self.actor,
            data=data,
        )
        output = DataProto.from_dict(
            tensors={"opsd_vocab_log_probs": vocab_log_probs},
        )

    output = output.to("cpu")

    if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
        self.actor.actor_module._handle.reshard(True)

    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    return output


def _compute_opsd_jsd_per_token(self, student_data: DataProto, teacher_data: DataProto) -> DataProto:
    """Compute OPSD distillation stats for full-JSD or sampled-KL modes.

    Args:
        student_data: DataProto with student prompt tensors + responses.
        teacher_data: DataProto with teacher prompt tensors aligned to student_data.
        `student_data.meta_info["opsd_beta"]`: JSD mixture coefficient in (0, 1).

    Returns:
        DataProto with:
            - full_jsd mode:
                * opsd_jsd_per_token: [batch, response_len] float32 on CPU
                * opsd_student_log_probs: [batch, response_len] float32 on CPU
            - sampled_kl mode:
                * opsd_student_log_probs: [batch, response_len] float32 on CPU
                * opsd_teacher_log_probs: [batch, response_len] float32 on CPU
    """
    assert self._is_actor, "compute_opsd_jsd_per_token requires actor role"
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load actor for OPSD JSD per-token", logger=logger)

    micro_batch_size = self.config.rollout.log_prob_micro_batch_size_per_gpu
    if micro_batch_size is None or micro_batch_size <= 0:
        micro_batch_size = 1
        logger.warning(
            "rollout.log_prob_micro_batch_size_per_gpu is unset/invalid; "
            "falling back to micro_batch_size=1 for OPSD JSD computation."
        )

    student_data.meta_info["micro_batch_size"] = micro_batch_size
    teacher_data.meta_info["micro_batch_size"] = micro_batch_size

    if "temperature" not in student_data.meta_info:
        student_data.meta_info["temperature"] = self.config.rollout.temperature
    if "temperature" not in teacher_data.meta_info:
        teacher_data.meta_info["temperature"] = student_data.meta_info["temperature"]
    beta = float(student_data.meta_info.get("opsd_beta", 0.5))
    teacher_source, loss_mode = validate_opsd_modes(student_data.meta_info)
    if teacher_source == "initial_frozen":
        if not self._is_ref or not hasattr(self, "ref_policy"):
            raise RuntimeError(
                "opsd_teacher_source='initial_frozen' requires actor role "
                "'actor_rollout_ref' with initialized ref_policy."
            )
        teacher_actor = self.ref_policy
    else:
        teacher_actor = self.actor

    with self.ulysses_sharding_manager:
        jsd_per_token, student_sampled_log_probs, teacher_sampled_log_probs = _compute_opsd_distill_stats(
            student_actor=self.actor,
            teacher_actor=teacher_actor,
            student_data=student_data,
            teacher_data=teacher_data,
            beta=beta,
            loss_mode=loss_mode,
        )
        tensors = {
            "opsd_student_log_probs": student_sampled_log_probs,
        }
        if loss_mode == "full_jsd":
            tensors["opsd_jsd_per_token"] = jsd_per_token
        else:
            tensors["opsd_teacher_log_probs"] = teacher_sampled_log_probs
        output = DataProto.from_dict(
            tensors=tensors,
            meta_info={
                "opsd_beta": beta,
                "opsd_teacher_source": teacher_source,
                "opsd_loss_mode": loss_mode,
            },
        )

    output = output.to("cpu")

    if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
        self.actor.actor_module._handle.reshard(True)
    if teacher_source == "initial_frozen" and self.world_size > 1:
        if fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)
        elif fsdp_version(self.ref_policy.actor_module) == 2:
            self.ref_policy.actor_module.reshard()

    if self._is_offload_param:
        offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    return output


def patch_actor_rollout_ref_worker():
    """Idempotently patch OPSD RPC methods onto ActorRolloutRefWorker."""
    if not hasattr(ActorRolloutRefWorker, "compute_opsd_vocab_log_probs"):
        registered_vocab = register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)(_compute_opsd_vocab_log_probs)
        ActorRolloutRefWorker.compute_opsd_vocab_log_probs = registered_vocab
        logger.info("Patched ActorRolloutRefWorker with compute_opsd_vocab_log_probs")

    if not hasattr(ActorRolloutRefWorker, "compute_opsd_jsd_per_token"):
        registered_jsd = register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)(_compute_opsd_jsd_per_token)
        ActorRolloutRefWorker.compute_opsd_jsd_per_token = registered_jsd
        logger.info("Patched ActorRolloutRefWorker with compute_opsd_jsd_per_token")


# Apply patch at import time
patch_actor_rollout_ref_worker()

# Keep OPSDActorRolloutRefWorker for backwards compat (now identical to base)
OPSDActorRolloutRefWorker = ActorRolloutRefWorker
