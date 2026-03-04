# Copyright 2026 OpenTinker
#
# Licensed under the Apache License, Version 2.0.
"""FSDP worker patch: add OPSD full-vocab JSD actor update."""

from __future__ import annotations

from typing import Any, Optional

import psutil
import torch
from codetiming import Timer

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker as _OrigActorRolloutRefWorker,
)
from opentinker.backend_patch.verl.trainer.ppo.full_vocab_jsd import (
    backward_full_vocab_jsd_on_logits,
    backward_topk_tail_reverse_kl_on_logits,
    compute_teacher_topk_log_probs_from_logits,
)


class ActorRolloutRefWorker(_OrigActorRolloutRefWorker):
    """Patched worker with full-vocab JSD actor update path."""

    def _resolve_log_prob_micro_batch_size(self, configured_micro_batch_size) -> int:
        """Resolve a valid micro batch size for log-prob recomputation."""
        if configured_micro_batch_size is not None:
            return int(configured_micro_batch_size)
        if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
            return int(self.config.actor.ppo_micro_batch_size_per_gpu)
        if self.config.actor.ppo_mini_batch_size is not None:
            return int(self.config.actor.ppo_mini_batch_size)
        raise ValueError(
            "log_prob micro batch size is not set. Please configure either "
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu or "
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu / ppo_mini_batch_size."
        )

    def _forward_response_logits(
        self,
        module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response_length: int,
        temperature: float,
    ) -> torch.Tensor:
        if position_ids.dim() == 3:
            # qwen2vl mrope layout
            position_ids = position_ids.transpose(0, 1)

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            # Request only tail logits used by response loss when model supports it.
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": False,
                "logits_to_keep": int(response_length) + 1,
            }
            try:
                output = module(**forward_kwargs)
            except TypeError:
                # Some model implementations do not expose logits_to_keep.
                forward_kwargs.pop("logits_to_keep", None)
                output = module(**forward_kwargs)
            logits = output.logits
            logits = logits[:, -response_length - 1 : -1, :]
            logits = logits / float(temperature)
        return logits

    def _update_policy_full_vocab_jsd(self, data: DataProto) -> dict:
        """Actor update for OPSD distillation (full-vocab JSD or top-K+tail reverse-KL)."""
        self.actor_module_fsdp.train()
        self.ref_module_fsdp.eval()
        temperature = data.meta_info.get("temperature", self.config.rollout.temperature)
        beta = float(data.meta_info.get("opsd_jsd_beta", 0.5))
        coef = float(data.meta_info.get("opsd_jsd_coef", 1.0))
        vocab_chunk_size = int(data.meta_info.get("opsd_jsd_vocab_chunk_size", 4096))
        token_chunk_size = int(data.meta_info.get("opsd_jsd_token_chunk_size", 0))
        distill_mode = str(
            data.meta_info.get("opsd_distill_mode", "full_vocab_jsd")
        ).lower()
        topk = int(data.meta_info.get("opsd_topk", 32))
        teacher_logits_cpu_offload = bool(
            data.meta_info.get("opsd_jsd_teacher_logits_cpu_offload", True)
        )
        if distill_mode not in {"full_vocab_jsd", "topk_reverse_kl_tail"}:
            raise ValueError(
                f"Unsupported OPSD distill mode: {distill_mode!r}. "
                "Expected one of {'full_vocab_jsd', 'topk_reverse_kl_tail'}."
            )
        if distill_mode == "topk_reverse_kl_tail" and topk <= 0:
            raise ValueError(f"opsd_topk must be > 0 for top-K distillation, got {topk}")

        required = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_position_ids",
        ]
        missing = [k for k in required if k not in data.batch.keys()]
        if missing:
            raise ValueError(
                f"Missing required keys for OPSD distill update: {missing}"
            )

        metrics = {}
        mini_batches = data.split(self.config.actor.ppo_mini_batch_size)

        for _ in range(self.config.actor.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.actor.use_dynamic_bsz:
                    max_token_len = (
                        self.config.actor.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                    grad_acc_steps = None
                else:
                    micro_bsz = self.config.actor.ppo_micro_batch_size_per_gpu
                    if micro_bsz is None:
                        micro_bsz = self.config.actor.ppo_mini_batch_size
                    grad_acc_steps = max(
                        1, self.config.actor.ppo_mini_batch_size // micro_bsz
                    )
                    micro_batches = mini_batch.split(micro_bsz)

                self.actor.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    mb = micro_batch.batch
                    response_mask = mb["response_mask"]
                    response_len = mb["responses"].size(1)

                    with torch.no_grad():
                        teacher_logits = self._forward_response_logits(
                            module=self.ref_module_fsdp,
                            input_ids=mb["teacher_input_ids"],
                            attention_mask=mb["teacher_attention_mask"],
                            position_ids=mb["teacher_position_ids"],
                            response_length=response_len,
                            temperature=temperature,
                        )
                    if distill_mode == "topk_reverse_kl_tail":
                        teacher_topk_log_probs, teacher_topk_indices = (
                            compute_teacher_topk_log_probs_from_logits(
                                teacher_logits=teacher_logits,
                                topk=topk,
                                vocab_chunk_size=vocab_chunk_size,
                            )
                        )
                        del teacher_logits
                        if teacher_logits_cpu_offload and teacher_topk_log_probs.is_cuda:
                            teacher_topk_log_probs = teacher_topk_log_probs.to(
                                "cpu", non_blocking=False
                            )
                            teacher_topk_indices = teacher_topk_indices.to(
                                "cpu", non_blocking=False
                            )
                            # Promptly release cached GPU blocks from teacher forward.
                            get_torch_device().empty_cache()
                    else:
                        if teacher_logits_cpu_offload and teacher_logits.is_cuda:
                            teacher_logits = teacher_logits.to("cpu", non_blocking=False)
                            # Promptly release cached GPU blocks from teacher forward.
                            get_torch_device().empty_cache()

                    student_logits = self._forward_response_logits(
                        module=self.actor_module_fsdp,
                        input_ids=mb["input_ids"],
                        attention_mask=mb["attention_mask"],
                        position_ids=mb["position_ids"],
                        response_length=response_len,
                        temperature=temperature,
                    )

                    if self.config.actor.use_dynamic_bsz:
                        loss_scale_factor = (
                            response_mask.shape[0]
                            / self.config.actor.ppo_mini_batch_size
                        )
                    else:
                        loss_scale_factor = 1 / float(grad_acc_steps)

                    if distill_mode == "topk_reverse_kl_tail":
                        distill_loss, effective_token_chunk_size = (
                            backward_topk_tail_reverse_kl_on_logits(
                                student_logits=student_logits,
                                teacher_topk_log_probs=teacher_topk_log_probs,
                                teacher_topk_indices=teacher_topk_indices,
                                response_mask=response_mask,
                                loss_agg_mode=self.config.actor.loss_agg_mode,
                                loss_scale=coef * loss_scale_factor,
                                vocab_chunk_size=vocab_chunk_size,
                                token_chunk_size=token_chunk_size,
                            )
                        )
                    else:
                        distill_loss, effective_token_chunk_size = (
                            backward_full_vocab_jsd_on_logits(
                                student_logits=student_logits,
                                teacher_logits=teacher_logits,
                                response_mask=response_mask,
                                beta=beta,
                                loss_agg_mode=self.config.actor.loss_agg_mode,
                                loss_scale=coef * loss_scale_factor,
                                vocab_chunk_size=vocab_chunk_size,
                                token_chunk_size=token_chunk_size,
                            )
                        )

                    # Release large tensors as early as possible.
                    if distill_mode == "topk_reverse_kl_tail":
                        del teacher_topk_log_probs
                        del teacher_topk_indices
                    else:
                        del teacher_logits
                    del student_logits

                    metric_payload = {
                        "distill/full_vocab_jsd_coef": coef,
                        "distill/full_vocab_jsd_vocab_chunk_size": float(
                            vocab_chunk_size
                        ),
                        "distill/full_vocab_jsd_token_chunk_size": float(
                            effective_token_chunk_size
                        ),
                        "distill/opsd_distill_mode_full_vocab_jsd": float(
                            distill_mode == "full_vocab_jsd"
                        ),
                        "distill/opsd_distill_mode_topk_reverse_kl_tail": float(
                            distill_mode == "topk_reverse_kl_tail"
                        ),
                    }
                    if distill_mode == "topk_reverse_kl_tail":
                        metric_payload.update(
                            {
                                # Keep existing key for dashboard compatibility.
                                "distill/full_vocab_jsd_loss": distill_loss.detach().item()
                                * loss_scale_factor,
                                "distill/topk_tail_reverse_kl_loss": distill_loss.detach().item()
                                * loss_scale_factor,
                                "distill/topk_tail_reverse_kl_k": float(topk),
                            }
                        )
                    else:
                        metric_payload.update(
                            {
                                "distill/full_vocab_jsd_loss": distill_loss.detach().item()
                                * loss_scale_factor,
                                "distill/full_vocab_jsd_beta": beta,
                            }
                        )
                    append_to_dict(metrics, metric_payload)

                grad_norm = self.actor._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor.actor_optimizer.zero_grad()
        return metrics

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        self.config.rollout.log_prob_micro_batch_size_per_gpu = (
            self._resolve_log_prob_micro_batch_size(
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
        )
        return super().compute_log_prob(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        self.config.ref.log_prob_micro_batch_size_per_gpu = (
            self._resolve_log_prob_micro_batch_size(
                self.config.ref.log_prob_micro_batch_size_per_gpu
            )
        )
        return super().compute_ref_log_prob(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        # Fallback to original PPO/GRPO update path.
        if not bool(data.meta_info.get("opsd_full_vocab_jsd_enable", False)):
            return super().update_actor(data)

        assert self._is_actor, "full-vocab JSD update requires actor role"
        if not hasattr(self, "ref_module_fsdp"):
            raise RuntimeError(
                "full-vocab JSD requires actor worker role=actor_rollout_ref "
                "(ref module loaded in actor worker)."
            )

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            load_fsdp_model_to_gpu(self.ref_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            with Timer(name="update_policy_full_vocab_jsd", logger=None) as timer:
                metrics = self._update_policy_full_vocab_jsd(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = (
                get_torch_device().max_memory_allocated() / (1024**3)
            )
            metrics["perf/max_memory_reserved_gb"] = (
                get_torch_device().max_memory_reserved() / (1024**3)
            )
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
            log_gpu_memory_usage("After offload actor/ref model during jsd update", logger=None)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during jsd update", logger=None)

        return output


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        return await self.rollout.chat_completion(json_request)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        return await self.rollout.generate(
            prompt_ids, sampling_params, request_id, image_data=image_data
        )
