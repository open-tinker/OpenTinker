# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Per-Turn Agent Loop Worker and Manager.

This module provides patched versions of AgentLoopWorkerBase and AgentLoopManager
that support expanding multi-turn rollout outputs into individual per-turn training
samples. This avoids context length issues from concatenating all turns into one
long sequence and aligns training context with inference context.

When an agent loop (e.g., AndroidAgentLoop) stores per-turn data in
extra_fields['per_turn_outputs'], this worker expands each turn into
a separate training sample with its own prompt, response, mask, and reward.
"""

import asyncio
import logging
import os
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorkerBase,
    AsyncLLMServerManager,
    _DummyConfig,
    _InternalAgentLoopOutput,
    _agent_loop_registry,
    get_trajectory_info,
)
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import rollout_trace_attr
from verl.utils.transferqueue_utils import tqbridge

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PerTurnAgentLoopWorkerBase(AgentLoopWorkerBase):
    """Agent loop worker that expands per-turn outputs into individual training samples.

    When the agent loop returns per-turn data in extra_fields['per_turn_outputs'],
    each turn is converted into a separate training sample with its own prompt,
    response, mask, and reward. For agent loops that don't produce per-turn outputs,
    the behavior is identical to the base AgentLoopWorkerBase.

    This solves two problems:
    1. Context length: Concatenated multi-turn sequences quickly exceed the model's
       context window, especially for environments with long observations.
    2. Train/inference mismatch: During rollout, AndroidAgentLoop generates each turn
       using only [system + latest observation] as context. But the concatenated
       training sequence exposes the model to ALL previous turns, creating a
       distribution mismatch. Per-turn training aligns training and inference contexts.
    """

    async def _pad_single_output(
        self,
        output: AgentLoopOutput,
    ) -> _InternalAgentLoopOutput:
        """Pad and convert a single AgentLoopOutput to _InternalAgentLoopOutput.

        This extracts the padding/conversion logic from the parent's _run_agent_loop
        so it can be reused for each per-turn output.
        """
        prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        response_length = self.config.actor_rollout_ref.rollout.response_length

        # Truncate prompt_ids if they exceed prompt_length (left-truncate to keep recent context)
        prompt_ids = output.prompt_ids
        if len(prompt_ids) > prompt_length:
            logger.warning(
                f"[PerTurnAgentLoop] Truncating per-turn prompt from {len(prompt_ids)} to {prompt_length} tokens"
            )
            prompt_ids = prompt_ids[-prompt_length:]

        # Truncate response_ids if they exceed response_length
        response_ids = output.response_ids[:response_length]
        response_mask = output.response_mask[:response_length]

        # Left-pad prompt
        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": prompt_ids},
            padding="max_length",
            max_length=prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        # Right-pad response
        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": response_ids},
            padding="max_length",
            max_length=response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        # Pad response mask
        response_mask_output = self.tokenizer.pad(
            {"input_ids": response_mask},
            padding="max_length",
            max_length=response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        # Pad logprobs
        response_logprobs_tensor = None
        if output.response_logprobs is not None:
            logprobs = output.response_logprobs[:response_length]
            pad_size = response_length - len(logprobs)
            response_logprobs_tensor = torch.tensor(logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask_final = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        # Handle multi-modal inputs and position_ids calculation
        multi_modal_inputs = None
        if (
            self.processor is not None
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            images = getattr(output, "multi_modal_data", {}).get("image", None)
            current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
            multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)
            multi_modal_inputs = dict(multi_modal_inputs)

            image_grid_thw = multi_modal_inputs.get("image_grid_thw")
            video_grid_thw = multi_modal_inputs.get("video_grid_thw")
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids.squeeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask.squeeze(0),
            ).unsqueeze(0)

            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            text_position_ids = text_position_ids.unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask_final,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs_tensor,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalAgentLoopOutput | list[_InternalAgentLoopOutput]:
        """Run agent loop and optionally expand per-turn outputs.

        If the agent loop produces per-turn outputs (via extra_fields['per_turn_outputs']),
        each turn is converted into a separate _InternalAgentLoopOutput. Otherwise,
        the behavior is identical to the base class.

        Returns:
            Single _InternalAgentLoopOutput or list of _InternalAgentLoopOutput (one per turn).
        """
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

            # Check for per-turn outputs from agent loops that support per-turn training
            per_turn_outputs = output.extra_fields.pop('per_turn_outputs', None)

            if per_turn_outputs and len(per_turn_outputs) > 0:
                # Expand per-turn outputs into separate training samples
                results = []
                for turn_output in per_turn_outputs:
                    internal = await self._pad_single_output(turn_output)
                    results.append(internal)

                logger.info(
                    f"[PerTurnAgentLoop] Expanded 1 episode into {len(results)} per-turn training samples"
                )
                return results
            else:
                # Standard single output processing (same as parent class)
                enable_async_reward = (
                    self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
                ) or not self.config.reward_model.enable

                internal = await self._pad_single_output(output)

                if output.reward_score is None and enable_async_reward:
                    batch = TensorDict(
                        {
                            "prompts": internal.prompt_ids,
                            "responses": internal.response_ids,
                            "attention_mask": internal.attention_mask,
                            "input_ids": internal.input_ids,
                            "position_ids": internal.position_ids,
                        },
                        batch_size=1,
                    )
                    non_tensor_batch = {
                        **{k: np.array([v]) for k, v in kwargs.items()},
                        "__num_turns__": np.array([output.num_turns]),
                        "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                    }

                    data = DataProto(
                        batch=batch,
                        non_tensor_batch=non_tensor_batch,
                    )
                    result = await self.reward_manager_worker.compute_score.remote(data)
                    output.reward_score = result["reward_score"]
                    output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
                    internal.reward_score = output.reward_score
                    internal.extra_fields = output.extra_fields

                return internal

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences with per-turn expansion support.

        This overrides the parent's generate_sequences to flatten per-turn outputs
        from _run_agent_loop before passing them to _postprocess. When _run_agent_loop
        returns a list (per-turn mode), the lists are flattened into a single list of
        training samples. The effective batch size may increase (e.g., 4 episodes * 15
        turns/episode = 60 training samples).

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch with per-turn expanded samples.
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # Override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # By default, assume single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))
        outputs = await asyncio.gather(*tasks)

        # Flatten per-turn outputs: _run_agent_loop may return a list (per-turn) or single output
        # Also build an expansion index that maps each flat output back to its source episode.
        # This allows the training server to expand the original batch (with reward fields etc.)
        # to match the expanded batch size.
        flat_outputs = []
        expansion_index = []  # expansion_index[j] = i means flat output j came from episode i
        for i, o in enumerate(outputs):
            if isinstance(o, list):
                flat_outputs.extend(o)
                expansion_index.extend([i] * len(o))
            else:
                flat_outputs.append(o)
                expansion_index.append(i)

        output = self._postprocess(flat_outputs)

        if len(flat_outputs) != len(outputs):
            # Per-turn expansion happened: store the index so the training server
            # can expand the original batch to match.
            output.meta_info['per_turn_expansion_index'] = expansion_index
            logger.info(
                f"[PerTurnAgentLoop] Expanded {len(outputs)} episodes into {len(flat_outputs)} per-turn training samples"
            )

        return output


@ray.remote
class PerTurnAgentLoopWorker(PerTurnAgentLoopWorkerBase):
    """Ray actor wrapper for PerTurnAgentLoopWorkerBase."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        super().__init__(config, server_handles, reward_router_address)


class PerTurnAgentLoopManager(AgentLoopManager):
    """Agent loop manager that uses PerTurnAgentLoopWorker for per-turn training support.

    Drop-in replacement for AgentLoopManager. The only difference is that this manager
    creates PerTurnAgentLoopWorker instances instead of AgentLoopWorker instances,
    enabling per-turn output expansion when agent loops produce per-turn data.

    When per_turn_training is disabled in the config (or the agent loop doesn't produce
    per-turn outputs), the behavior is identical to the standard AgentLoopManager.
    """

    def __init__(self, config: DictConfig, worker_group=None, rm_wg=None):
        # Set the worker class BEFORE calling super().__init__(), which calls _init_agent_loop_workers
        self.agent_loop_workers_class = PerTurnAgentLoopWorker
        super().__init__(config, worker_group, rm_wg)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Override to handle per-turn expansion indices across workers.

        Each worker may produce per_turn_expansion_index in meta_info, but
        DataProto.concat() requires matching meta_info values. We extract the
        per-worker expansion indices, adjust them to global offsets, remove
        them from meta_info before concat, and attach the combined global
        index to the final output.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()

        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
            ]
        )

        # Collect per-turn expansion indices from each worker and build a
        # global expansion index before concat (which would fail on conflicting
        # meta_info values).
        chunk_sizes = [len(c) for c in chunks]
        has_expansion = any('per_turn_expansion_index' in o.meta_info for o in outputs)
        global_expansion_index = None

        if has_expansion:
            global_expansion_index = []
            offset = 0
            for o, cs in zip(outputs, chunk_sizes):
                local_idx = o.meta_info.pop('per_turn_expansion_index', None)
                if local_idx is not None:
                    global_expansion_index.extend([idx + offset for idx in local_idx])
                else:
                    # No expansion for this worker — identity mapping
                    global_expansion_index.extend(range(offset, offset + len(o)))
                offset += cs
        else:
            for o in outputs:
                o.meta_info.pop('per_turn_expansion_index', None)

        output = DataProto.concat(outputs)

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()

        # Calculate performance metrics (same as parent)
        metrics = [output.meta_info.pop("metrics") for output in outputs]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}

        if global_expansion_index is not None:
            output.meta_info['per_turn_expansion_index'] = global_expansion_index
            logger.info(
                f"[PerTurnAgentLoop] Global expansion: {sum(chunk_sizes)} episodes -> "
                f"{len(global_expansion_index)} per-turn training samples"
            )

        return output
