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
"""Android Agent Loop for LLM-Environment Interaction.

This module provides a simplified agent loop for multi-turn interactions
between LLMs and external environments (e.g., OpenAI Gym-like APIs).
Unlike ToolAgentLoop, this does not handle tool calls - the external
environment is treated as a conversational API that returns observations.
"""

import copy
import json
import logging
import os
import re
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import (
    initialize_interactions_from_config,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _deserialize_images(image_data):
    """Deserialize PIL Images if they're still in serialized dict form.

    This handles the case where images arrive as {'__type__': 'PIL.Image', '__data__': base64...}
    instead of actual PIL Image objects.

    Args:
        image_data: List of images (either PIL Images or serialized dicts)

    Returns:
        List of PIL Image objects
    """
    if not image_data:
        return image_data

    from PIL import Image
    import base64
    import io

    result = []
    for img in image_data:
        if isinstance(img, dict) and img.get("__type__") == "PIL.Image":
            # Deserialize from base64-encoded PNG
            data_bytes = base64.b64decode(img["__data__"])
            buffer = io.BytesIO(data_bytes)
            result.append(Image.open(buffer).copy())
        elif hasattr(img, "save") and hasattr(img, "mode"):
            # Already a PIL Image
            result.append(img)
        else:
            # Unknown type, keep as is
            result.append(img)
    return result


class AndroidAgentState(Enum):
    """States for the Android agent loop."""

    PENDING = "pending"  # Initial state, preparing the prompt
    GENERATING = "generating"  # LLM is generating response
    INTERACTING = "interacting"  # Interacting with external environment
    TERMINATED = "terminated"  # Rollout complete


class AndroidAgentData:
    """Encapsulates all state variables for the Android agent loop.

    This is similar to AgentData in tool_agent_loop.py but without tool-specific fields.
    """

    def __init__(
        self,
        messages: list[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        image_data: Optional[list[Any]] = None,
    ):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # Multimodal data (images/videos for VL models)
        self.image_data = image_data

        # Token sequences
        # prompt_ids: full accumulated sequence (system, user, assistant, user, ...) for final output/loss
        self.prompt_ids: list[int] = []
        # generation_prompt_ids: what the model sees each time (system + latest user only)
        self.generation_prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []

        # Turn tracking
        self.user_turns = 0
        self.assistant_turns = 0

        # Reward tracking (for turn-level rewards, accumulated for final reward)
        self.turn_scores: list[float] = []

        # Per-turn data for per-turn training mode
        # Each entry: {generation_prompt_ids, response_ids, response_logprobs, reward}
        self.per_turn_data: list[dict[str, Any]] = []

        # Extra fields for additional data
        self.extra_fields: dict[str, Any] = {}


# @register("Android_agent")
class AndroidAgentLoop(AgentLoopBase):
    """Android agent loop for LLM-environment interaction.

    This agent loop handles multi-turn conversations between an LLM and
    an external environment. The environment is accessed through a
    BaseInteraction subclass that implements the generate_response method.

    State Machine:
        PENDING -> GENERATING -> INTERACTING -> GENERATING -> ... -> TERMINATED

    Response Mask:
        - mask=1: LLM generated tokens (included in loss computation)
        - mask=0: Environment observations, system prompt, padding (excluded from loss)

    Reward Attribution:
        - Final Reward: The cumulative reward is placed at the last response token position
    """

    # Trace saving configuration
    _trace_output_dir: Optional[str] = None
    _trace_count: int = 0
    _trace_lock = threading.Lock()  # Thread-safe counter (within single process)
    _save_traces: bool = False
    _process_id: Optional[int] = None  # Track process ID for multi-process training

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level AndroidAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = (
            config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        )

        cls.apply_chat_template_kwargs = config.data.get(
            "apply_chat_template_kwargs", {}
        )
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

        # Per-turn token limit (optional, None means no per-turn limit)
        cls.max_tokens_per_turn = config.actor_rollout_ref.rollout.multi_turn.get(
            "max_tokens_per_turn", None
        )

        # Per-turn training mode: each interaction turn becomes a separate training sample
        # instead of concatenating all turns into one long sequence
        cls.per_turn_training = config.actor_rollout_ref.rollout.multi_turn.get(
            "per_turn_training", False
        )
        # Per-turn reward gamma: when > 0, replace immediate turn rewards with
        # discounted cumulative returns so earlier turns can sense final outcome
        cls.per_turn_reward_gamma = config.actor_rollout_ref.rollout.multi_turn.get(
            "per_turn_reward_gamma", 0.0
        )
        if cls.per_turn_training:
            print(
                f"[AndroidAgentLoop] Per-turn training mode ENABLED: each turn becomes a separate training sample"
                f" (reward_gamma={cls.per_turn_reward_gamma})"
            )

        # Pre-compute system prompt tokens for later stripping
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}],
            add_generation_prompt=False,
            tokenize=True,
            **cls.apply_chat_template_kwargs,
        )

        # Initialize interactions from config
        # CROSS-NODE FIX: If interaction_config_content is available, recreate the temp file locally
        # because the original path may point to a file on a different node's /tmp/
        cls.interaction_config_file = (
            config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        )
        interaction_config_content = config.actor_rollout_ref.rollout.multi_turn.get(
            "interaction_config_content", None
        )

        if interaction_config_content:
            import tempfile

            # Create a local temp file with the content on THIS worker's node
            fd, local_path = tempfile.mkstemp(
                suffix=".yaml", prefix="interaction_config_worker_"
            )
            with os.fdopen(fd, "w") as f:
                f.write(interaction_config_content)
            cls.interaction_config_file = local_path
            print(
                f"[AndroidAgentLoop] Created local interaction config from content: {local_path}"
            )

        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = (
                cls._initialize_interactions(cls.interaction_config_file)
            )
        else:
            cls.interaction_map = {}

        # Initialize trace saving
        cls._trace_output_dir = os.environ.get("ROLLOUT_TRACE_DIR", None)
        if cls._trace_output_dir:
            # Create per-job subdirectory to isolate traces from different client tasks
            job_id = os.environ.get("ROLLOUT_TRACE_JOB_ID", None)
            if job_id:
                cls._trace_output_dir = str(Path(cls._trace_output_dir) / f"job_{job_id}")
            cls._save_traces = True
            cls._process_id = os.getpid()  # Store process ID for unique trace naming
            Path(cls._trace_output_dir).mkdir(parents=True, exist_ok=True)
            print(
                f"[AndroidAgentLoop] Rollout trace saving ENABLED: {cls._trace_output_dir} (PID: {cls._process_id})"
            )
        else:
            cls._save_traces = False
            print(
                "[AndroidAgentLoop] Rollout trace saving DISABLED (set ROLLOUT_TRACE_DIR to enable)"
            )

        # Initialize Weave tracing on server side
        # Enabled via WEAVE_PROJECT env var (e.g., "opentinker/Android-env")
        # or via config.actor_rollout_ref.rollout.multi_turn.weave_project
        weave_project = os.environ.get("WEAVE_PROJECT", None)
        if weave_project is None:
            weave_project = config.actor_rollout_ref.rollout.multi_turn.get(
                "weave_project", None
            )

        if weave_project:
            try:
                from verl.utils.rollout_trace import RolloutTraceConfig

                experiment_name = config.actor_rollout_ref.rollout.multi_turn.get(
                    "experiment_name", "default"
                )
                RolloutTraceConfig.init(
                    project_name=weave_project,
                    experiment_name=experiment_name,
                    backend="weave",
                    token2text=True,
                )
                print(
                    f"[AndroidAgentLoop] Weave tracing ENABLED: project={weave_project}, experiment={experiment_name}"
                )
            except ImportError:
                print(
                    "[AndroidAgentLoop] WARNING: Weave not installed (pip install weave)"
                )
            except Exception as e:
                logger.warning(f"Failed to init Weave: {e}")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the agent loop for a single trajectory.

        Args:
            sampling_params: LLM sampling parameters (temperature, top_p, etc.)
            **kwargs: Dataset fields including 'raw_prompt', 'extra_info', etc.

        Returns:
            AgentLoopOutput containing prompt_ids, response_ids, response_mask, etc.
        """
        # breakpoint()
        # Extract step if available (for trace naming)
        step = kwargs.get("step", None)
        # Extract messages from kwargs
        if "raw_prompt" not in kwargs:
            raise KeyError("raw_prompt is required in kwargs for agent loop")

        raw_prompt_value = kwargs["raw_prompt"]
        # CRITICAL: Deep copy to prevent GRPO n-sample message accumulation bug!
        # When GRPO samples the same prompt N times, each rollout MUST have its own
        # independent messages list. Without deepcopy, all N rollouts share the same
        # list reference, causing conversation history from all samples to accumulate.
        if isinstance(raw_prompt_value, list):
            messages = copy.deepcopy(raw_prompt_value)
        elif isinstance(raw_prompt_value, dict):
            messages = [copy.deepcopy(raw_prompt_value)]
        else:
            raise TypeError(
                f"raw_prompt must be a list or dict, got {type(raw_prompt_value)}"
            )

        metrics = {}

        # Use a stable request_id if available to allow environment reuse on the server.
        # extra_info.sample_id is often a stable index for the worker/sample.
        stable_id = kwargs.get("extra_info", {}).get("sample_id")
        if stable_id is not None:
            request_id = str(stable_id)
        else:
            request_id = uuid4().hex

        # CRITICAL: Extract multimodal data (images) from kwargs for VL models
        # This follows the verl pattern from single_turn_agent_loop.py
        multi_modal_data_raw = kwargs.get("multi_modal_data")
        print(
            f"[AndroidAgentLoop DEBUG] multi_modal_data type: {type(multi_modal_data_raw)}, value: {multi_modal_data_raw!r:.200}"
        )

        if isinstance(multi_modal_data_raw, dict):
            image_data = copy.deepcopy(multi_modal_data_raw.get("image", None))
        else:
            image_data = None

        # Deserialize images if they're still in serialized dict form
        # This handles cases where HTTP deserialization didn't fully complete
        if image_data:
            image_data = _deserialize_images(image_data)

        print(
            f"[AndroidAgentLoop DEBUG] image_data type: {type(image_data)}, is_list: {isinstance(image_data, list)}"
        )
        if image_data:
            print(
                f"[AndroidAgentLoop DEBUG] image_data[0] type: {type(image_data[0]) if len(image_data) > 0 else 'empty'}"
            )

        # Debug: Save images if SAVE_DEBUG_IMAGES is set
        if image_data and os.environ.get("SAVE_DEBUG_IMAGES"):
            await self._save_debug_images(image_data, request_id)

        # Initialize interaction if configured
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs.get("extra_info", {}).get(
                "interaction_kwargs", {}
            )
            if not interaction_kwargs:
                interaction_kwargs = kwargs.get("interaction_kwargs", {})

            # Get interaction name - use default if not provided in data
            if "name" not in interaction_kwargs:
                # Use the first interaction from config as default
                if self.interaction_map:
                    default_interaction_name = list(self.interaction_map.keys())[0]
                    interaction_kwargs["name"] = default_interaction_name
                    logger.info(
                        f"Using default interaction: {default_interaction_name}"
                    )
                else:
                    raise ValueError(
                        "No interactions configured in interaction_config_file"
                    )

            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found. Available: {list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

            # Capture initial board state ONLY for Gomoku environment (not other environments)
            initial_board_state = None
            if interaction_name == "gomoku":  # Only for Gomoku
                if (
                    hasattr(interaction, "_instance_dict")
                    and request_id in interaction._instance_dict
                ):
                    initial_board_state = interaction._instance_dict[request_id].get(
                        "initial_board_state"
                    )

        # Create agent data to track state
        agent_data = AndroidAgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
            image_data=image_data,
        )

        # breakpoint()

        # Store initial board state if available
        if initial_board_state:
            agent_data.extra_fields["initial_board_state"] = initial_board_state

        # State machine loop
        state = AndroidAgentState.PENDING
        try:
            while state != AndroidAgentState.TERMINATED:
                if state == AndroidAgentState.PENDING:
                    state = await self._handle_pending_state(
                        agent_data, sampling_params
                    )
                elif state == AndroidAgentState.GENERATING:
                    state = await self._handle_generating_state(
                        agent_data, sampling_params
                    )
                elif state == AndroidAgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AndroidAgentState.TERMINATED
        finally:
            # CRITICAL: Always finalize interaction to release resources
            if agent_data.interaction is not None:
                await agent_data.interaction.finalize_interaction(agent_data.request_id)

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[
            : len(agent_data.prompt_ids) - len(agent_data.response_mask)
        ]

        # Truncate prompt_ids if they exceed steps * prompt_length (left truncate to keep recent context).
        # Each generation uses prompt <= prompt_length; total prefix is at most steps * prompt_length.
        max_total_prompt = self.prompt_length * max(
            self.max_assistant_turns or 1, self.max_user_turns or 1
        )
        if len(prompt_ids) > max_total_prompt:
            logger.warning(
                f"[AndroidAgentLoop] Truncating prompt from {len(prompt_ids)} to {max_total_prompt} tokens (steps * prompt_length)"
            )
            prompt_ids = prompt_ids[-max_total_prompt:]

        # Calculate final reward (sum of all turn scores)
        # Return 0.0 if no turn scores collected - this prevents fallback to naive reward loop
        # which expects ground_truth data that gym environments don't provide
        final_reward = sum(agent_data.turn_scores) if agent_data.turn_scores else 0.0

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data={"image": agent_data.image_data}
            if agent_data.image_data
            else {},
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            reward_score=final_reward,
            extra_fields={},
        )
        # Explicitly set reward_extra_info with FIXED keys to ensure consistency
        # across all workers. This prevents meta_info['reward_extra_keys'] conflicts
        # when DataProto.concat() merges outputs from different workers.
        # Using the same keys as GSM8K/Math reward functions for compatibility.
        output.extra_fields["reward_extra_info"] = {
            "acc": None,  # Placeholder - will be filtered out in metrics computation
        }
        # Ensure env_info exists for all samples (even if empty) for consistent DataProto.concat
        output.extra_fields["env_info"] = agent_data.extra_fields.get("env_info", [])
        output.extra_fields["turn_scores"] = agent_data.turn_scores
        # Add any other extra fields (except the ones we already set)
        for key, value in agent_data.extra_fields.items():
            if key not in output.extra_fields:
                output.extra_fields[key] = value

        # Build per-turn outputs for per-turn training mode
        # Each turn becomes a separate training sample with its own prompt, response, and reward.
        # This avoids concatenating all turns into one long sequence that exceeds context limits.
        if self.per_turn_training and agent_data.per_turn_data:
            # Drop the last per_turn_data entry if it has reward=0.0 (never went through INTERACTING).
            # This happens when the loop terminates in GENERATING state (e.g., max_assistant_turns exceeded).
            # A reward=0.0 sample provides no gradient signal and dilutes the actual rewards.
            if (
                len(agent_data.per_turn_data) > 0
                and agent_data.per_turn_data[-1]['reward'] == 0.0
                and len(agent_data.per_turn_data) > len(agent_data.turn_scores)
            ):
                dropped = agent_data.per_turn_data.pop()
                logger.info(
                    f"[AndroidAgentLoop] Dropped last per-turn sample with reward=0.0 "
                    f"(never reached INTERACTING state)"
                )
            # Compute per-turn rewards: either raw immediate rewards or cumulative
            # discounted returns (when per_turn_reward_gamma > 0).
            # Cumulative returns propagate the final outcome signal (e.g., +10.0 for
            # success) back to earlier turns, enabling cross-turn credit assignment
            # that GAE alone cannot provide in per-turn mode.
            immediate_rewards = [t['reward'] for t in agent_data.per_turn_data]
            if self.per_turn_reward_gamma > 0 and len(immediate_rewards) > 1:
                gamma = self.per_turn_reward_gamma
                cumulative_returns = [0.0] * len(immediate_rewards)
                G = 0.0
                for i in reversed(range(len(immediate_rewards))):
                    G = immediate_rewards[i] + gamma * G
                    cumulative_returns[i] = G
                per_turn_rewards = cumulative_returns
            else:
                per_turn_rewards = immediate_rewards

            per_turn_agent_outputs = []
            for turn_data, reward in zip(agent_data.per_turn_data, per_turn_rewards):
                turn_response_ids = turn_data['response_ids'][:self.response_length]
                turn_logprobs = turn_data['response_logprobs'][:self.response_length] if turn_data['response_logprobs'] else None
                turn_output = AgentLoopOutput(
                    prompt_ids=turn_data['generation_prompt_ids'],
                    response_ids=turn_response_ids,
                    response_mask=[1] * len(turn_response_ids),
                    response_logprobs=turn_logprobs,
                    multi_modal_data={"image": agent_data.image_data} if agent_data.image_data else {},
                    reward_score=reward,
                    num_turns=1,
                    metrics=agent_data.metrics,
                    extra_fields={
                        "reward_extra_info": {"acc": None},
                        "env_info": [],
                        "turn_scores": [turn_data['reward']],  # Keep original immediate reward for logging
                    },
                )
                per_turn_agent_outputs.append(turn_output)
            output.extra_fields['per_turn_outputs'] = per_turn_agent_outputs
            logger.info(
                f"[AndroidAgentLoop] Per-turn training: {len(per_turn_agent_outputs)} turns "
                f"(immediate_rewards: {[round(r, 4) for r in immediate_rewards]}, "
                f"training_rewards: {[round(r, 4) for r in per_turn_rewards]})"
            )

        # Save rollout trace for verification (if enabled via ROLLOUT_TRACE_DIR env var)
        if self._save_traces:
            await self._save_rollout_trace(agent_data, output, request_id, step)

        return output

    async def _handle_pending_state(
        self,
        agent_data: AndroidAgentData,
        sampling_params: dict[str, Any],
    ) -> AndroidAgentState:
        """Handle the pending state: tokenize the initial prompt."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # CRITICAL: Pass images to processor for VL models
            model_inputs = self.processor(
                text=[raw_prompt],
                images=agent_data.image_data if agent_data.image_data else None,
                return_tensors="pt",
            )
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        # First turn: generation sees the same prompt as full (system + initial user)
        agent_data.generation_prompt_ids = list(agent_data.prompt_ids)
        return AndroidAgentState.GENERATING

    async def _handle_generating_state(
        self,
        agent_data: AndroidAgentData,
        sampling_params: dict[str, Any],
    ) -> AndroidAgentState:
        """Handle the generating state: generate LLM response.

        The generated tokens are marked with mask=1 (included in loss computation).
        """
        import time

        # CONTEXT OVERFLOW PROTECTION: Check if we have enough room for generation
        # This prevents the "max_tokens must be at least 1, got -X" error from vLLM
        # which occurs when prompt_len exceeds max_model_len (especially for VL models
        # where image tokens can be very large)
        total_context_budget = self.prompt_length + self.response_length
        min_generation_tokens = 16  # Minimum tokens needed for meaningful generation

        if len(agent_data.generation_prompt_ids) + min_generation_tokens > total_context_budget:
            logger.warning(
                f"[AndroidAgentLoop] Context overflow detected: prompt_len={len(agent_data.generation_prompt_ids)}, "
                f"total_budget={total_context_budget}. Terminating early to avoid negative max_tokens error."
            )
            # Add a placeholder response if none exists yet (so we have valid output)
            if not agent_data.response_ids:
                # Add EOS token as minimal response
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is not None:
                    agent_data.response_ids = [eos_token_id]
                    agent_data.prompt_ids.append(eos_token_id)
                    agent_data.response_mask.append(1)
            return AndroidAgentState.TERMINATED

        print(
            f"[AndroidAgentLoop DEBUG] _handle_generating_state START: request_id={agent_data.request_id}, prompt_len={len(agent_data.prompt_ids)}"
        )
        start_time = time.time()
        with simple_timer("generate_sequences", agent_data.metrics):
            print(
                f"[AndroidAgentLoop DEBUG] Calling server_manager.generate() with image_data={agent_data.image_data is not None}..."
            )
            # CRITICAL: Pass image_data to vLLM for VL model inference
            try:
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.generation_prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                )
            except Exception as e:
                # Before re-raising: decode and print the full input for debugging
                import sys
                n_tokens = len(agent_data.generation_prompt_ids)
                msg = (
                    f"[AndroidAgentLoop] generate() failed: {type(e).__name__}: {e}\n"
                    f"[AndroidAgentLoop] prompt_ids len={n_tokens} (full accumulated multi-turn prompt; each BEGIN/END block below = one worker's dump)\n"
                    "--- BEGIN full decoded prompt (for debug) ---"
                )
                sys.stderr.write(msg)
                sys.stderr.flush()
                try:
                    decoded = self.tokenizer.decode(
                        agent_data.generation_prompt_ids, skip_special_tokens=True
                    )
                    sys.stderr.write(decoded)
                    sys.stderr.write("\n--- END full decoded prompt ---\
")
                    sys.stderr.flush()
                except Exception as decode_err:
                    sys.stderr.write(f"[AndroidAgentLoop] decode failed: {decode_err}\n")
                    sys.stderr.flush()
                raise
            elapsed = time.time() - start_time
            print(
                f"[AndroidAgentLoop DEBUG] server_manager.generate() COMPLETED in {elapsed:.2f}s, response_tokens={len(output.token_ids) if output else 0}"
            )

        agent_data.assistant_turns += 1
        response_token_ids = output.token_ids
        response_log_probs = output.log_probs

        # Apply per-turn token limit if configured
        if (
            self.max_tokens_per_turn
            and len(response_token_ids) > self.max_tokens_per_turn
        ):
            logger.debug(
                f"Truncating turn response from {len(response_token_ids)} to {self.max_tokens_per_turn} tokens"
            )
            response_token_ids = response_token_ids[: self.max_tokens_per_turn]
            if response_log_probs:
                response_log_probs = response_log_probs[: self.max_tokens_per_turn]

        agent_data.response_ids = response_token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(
            agent_data.response_ids
        )  # mask=1 for LLM tokens

        if response_log_probs:
            agent_data.response_logprobs += response_log_probs

        # Collect per-turn data for per-turn training mode
        if self.per_turn_training:
            agent_data.per_turn_data.append({
                'generation_prompt_ids': list(agent_data.generation_prompt_ids),
                'response_ids': list(response_token_ids),
                'response_logprobs': list(response_log_probs) if response_log_probs else [],
                'reward': 0.0,  # Will be updated in INTERACTING state when env provides reward
            })

        # Check termination conditions
        if len(agent_data.response_ids) >= self.response_length:
            return AndroidAgentState.TERMINATED
        # Use > instead of >= so that max_assistant_turns=1 allows 1 generation + 1 step
        # before terminating (instead of terminating immediately after first generation)
        if (
            self.max_assistant_turns
            and agent_data.assistant_turns > self.max_assistant_turns
        ):
            return AndroidAgentState.TERMINATED
        # Similarly, max_user_turns=1 means user can ask once, then terminate after next generation
        if self.max_user_turns and agent_data.user_turns > self.max_user_turns:
            return AndroidAgentState.TERMINATED

        # Add assistant message to conversation history
        if agent_data.interaction is not None:
            assistant_message = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    agent_data.response_ids, skip_special_tokens=True
                ),
            )
            agent_data.messages.append(
                {"role": "assistant", "content": assistant_message}
            )
            return AndroidAgentState.INTERACTING
        else:
            # No interaction configured, terminate after first generation
            return AndroidAgentState.TERMINATED

    async def _handle_interacting_state(
        self,
        agent_data: AndroidAgentData,
    ) -> AndroidAgentState:
        """Handle the interacting state: get response from external environment.

        The environment observation is tokenized and marked with mask=0
        (excluded from loss computation).
        """
        # Call the interaction to get environment response
        (
            should_terminate,
            observation,
            reward,
            info,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        # Record turn-level reward (will be summed for final reward)
        if reward is not None:
            agent_data.turn_scores.append(reward)

            # Update per-turn reward for the last generation
            if self.per_turn_training and agent_data.per_turn_data:
                agent_data.per_turn_data[-1]['reward'] = reward

        # Store environment info under a SINGLE key to ensure consistent structure
        # across all samples (avoids DataProto.concat assertion errors when different
        # samples return different info keys)
        if info:
            # Append to list instead of overwriting (for multi-turn)
            if "env_info" not in agent_data.extra_fields:
                agent_data.extra_fields["env_info"] = []
            agent_data.extra_fields["env_info"].append(info)

        # observation may be "long_part<obs>short_part": long part for generation prompt, short for prompt_ids/mask
        if "<obs>" in observation:
            observation_long, observation_short = observation.split("<obs>", 1)
        else:
            observation_long = observation
            observation_short = observation

        # Construct user message from full observation (for message history)
        add_messages: list[dict[str, Any]] = [{"role": "user", "content": observation_short}]
        agent_data.messages.extend(add_messages)

        # Tokenize the user message (environment observation)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    add_messages, add_generation_prompt=True, tokenize=True
                ),
            )

        # Strip the system prompt tokens (they are duplicated from the full conversation)
        response_ids = response_ids[len(self.system_prompt) :]

        # Check if adding these tokens would exceed response length
        # In per-turn training mode, skip this check since each turn is trained independently
        # and the accumulated response_mask length is irrelevant.
        if not self.per_turn_training and len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AndroidAgentState.TERMINATED

        # Update full prompt_ids and response_mask (accumulated for final output/loss)
        # mask=0 for environment observation tokens (not included in loss)
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)

        if agent_data.response_logprobs:
            # Pad logprobs with 0.0 for observation tokens
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # Next generation: model sees system + LONG part of observation (full context for generation)
        system_msg = next(
            (m for m in agent_data.messages if m.get("role") == "system"),
            agent_data.messages[0],
        )
        minimal_messages = [system_msg, {"role": "user", "content": observation_long}]
        if self.processor is not None:
            raw_minimal = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    minimal_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(
                text=[raw_minimal],
                images=agent_data.image_data if agent_data.image_data else None,
                return_tensors="pt",
            )
            agent_data.generation_prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.generation_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    minimal_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        if should_terminate:
            return AndroidAgentState.TERMINATED
        else:
            return AndroidAgentState.GENERATING

    async def _save_debug_images(self, image_data: list, request_id: str):
        """Save debug images to disk when SAVE_DEBUG_IMAGES env var is set.

        This helps verify that images are being correctly passed to the model.
        Images are saved to the ROLLOUT_TRACE_DIR or /tmp/debug_images/ folder.
        """
        import os
        from pathlib import Path

        output_dir = Path(os.environ.get("ROLLOUT_TRACE_DIR", "/tmp/debug_images"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(image_data):
            try:
                # Handle PIL Images
                if hasattr(img, "save"):
                    img_path = output_dir / f"debug_image_{request_id[:8]}_{idx}.png"
                    await self.loop.run_in_executor(
                        None, lambda p=img_path, im=img: im.save(str(p))
                    )
                    logger.debug(f"Saved debug image to {img_path}")
                else:
                    logger.debug(
                        f"Image {idx} is of type {type(img)}, cannot save"
                    )
            except Exception as e:
                logger.debug(f"Failed to save image {idx}: {e}")

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.

        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialized interactions: {list(interaction_map.keys())}")
        return interaction_map

    async def _save_rollout_trace(
        self,
        agent_data: AndroidAgentData,
        output: AgentLoopOutput,
        request_id: str,
        step: Optional[int] = None,
    ):
        """Save rollout trace to JSON file for algorithm verification.

        Trace includes:
        - Full conversation messages
        - Token IDs and response mask
        - Rewards and env info
        - Decoded text (readable format)
        - Per-turn board states (for Gomoku and similar environments)
        """
        try:
            # Use a sequential counter for ordered filenames
            # Include request_id prefix to avoid collisions across workers
            # (each worker process has its own _trace_count starting from 0)
            req_short = request_id[:8] if request_id else "unknown"
            with self._trace_lock:
                self.__class__._trace_count += 1
                seq = self.__class__._trace_count

            # Format: <seq>_<request_id_short>  (e.g. trace_000001_ab20ea37.json)
            if step is not None:
                trace_id = f"{seq:06d}_step{step:06d}_{req_short}"
            else:
                trace_id = f"{seq:06d}_{req_short}"

            # Decode text for readability
            prompt_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    output.prompt_ids, skip_special_tokens=True
                ),
            )
            response_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    output.response_ids, skip_special_tokens=True
                ),
            )

            # Extract per-turn board states from messages
            per_turn_board_states = self._extract_per_turn_board_states(
                agent_data.messages, agent_data
            )

            # Build trace data
            trace_data = {
                "trace_id": trace_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                # Conversation
                "messages": agent_data.messages,
                "initial_prompt": agent_data.messages[0]
                if agent_data.messages
                else None,
                # Decoded text (readable)
                "prompt_text": prompt_text,
                "response_text": response_text,
                # Token-level data
                "prompt_ids": output.prompt_ids,
                "response_ids": output.response_ids,
                "response_mask": output.response_mask,
                # Response mask analysis
                "response_mask_analysis": {
                    "total_tokens": len(output.response_mask),
                    "llm_tokens": sum(output.response_mask),  # mask=1
                    "env_tokens": len(output.response_mask)
                    - sum(output.response_mask),  # mask=0
                    "llm_ratio": sum(output.response_mask) / len(output.response_mask)
                    if output.response_mask
                    else 0,
                },
                # Per-turn board states (for Gomoku verification)
                "per_turn_board_states": per_turn_board_states,
                # Rewards
                "reward_score": output.reward_score,
                "turn_scores": agent_data.turn_scores,
                "env_info": agent_data.extra_fields.get("env_info", []),
                # Turn tracking
                "num_user_turns": agent_data.user_turns,
                "num_assistant_turns": agent_data.assistant_turns,
                "total_turns": output.num_turns,
                # Configuration (for verification)
                # NOTE: response_length is the TOTAL response budget for entire multi-turn trajectory
                # NOT per-turn! Each generation call gets max_tokens = max_model_len - current_prompt_len
                "config": {
                    "response_length": self.response_length,  # Total response budget (NOT per-turn)
                    "prompt_length": self.prompt_length,
                    "max_user_turns": self.max_user_turns,
                    "max_assistant_turns": self.max_assistant_turns,
                    "max_tokens_per_turn": self.max_tokens_per_turn,  # Per-turn limit (None = no limit)
                },
                # Metrics
                "metrics": agent_data.metrics,
            }

            # Save trace file to per-job directory
            trace_file = Path(self._trace_output_dir) / f"trace_{trace_id}.json"
            await self.loop.run_in_executor(
                None,
                lambda: trace_file.write_text(
                    json.dumps(trace_data, indent=2, default=str)
                ),
            )

            # Also append to streaming JSONL file for easy batch processing
            jsonl_file = Path(self._trace_output_dir) / "traces.jsonl"
            json_line = json.dumps(trace_data, default=str) + "\n"
            await self.loop.run_in_executor(
                None,
                lambda: jsonl_file.open("a").write(json_line),
            )

            logger.info(f"Saved trace {trace_id} to {trace_file}")

        except Exception as e:
            import traceback

            logger.warning(f"Failed to save rollout trace: {e}")
            print(traceback.format_exc())

    def _extract_per_turn_board_states(
        self,
        messages: list[dict[str, Any]],
        agent_data: AndroidAgentData,
    ) -> list[dict[str, Any]]:
        """Extract board states from each message for verification.

        This is ONLY for Gomoku environment. Other environments will get empty board states.

        For Gomoku, this verifies that the board state in the prompt matches the actual game state.

        Combines:
        - Board states from env_info (ground truth from environment)
        - Visual board extracted from message content
        - Initial board state if available

        Returns:
            List of dicts with turn info, role, board_state, and verification status.
            Empty list for non-Gomoku environments.
        """
        # Only extract board states for Gomoku environment
        interaction_name = agent_data.interaction_kwargs.get("name", "")
        if interaction_name != "gomoku":
            return []  # No board state tracking for other environments

        per_turn_states = []

        # Get env_info list (board states from environment)
        env_info_list = agent_data.extra_fields.get("env_info", [])
        initial_board_state = agent_data.extra_fields.get("initial_board_state")

        # Track which env_info entry corresponds to which user message
        # env_info is added after each interaction (user turn)
        env_info_idx = 0

        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            # Extract visual board from message content
            visual_board = self._extract_board_from_content(content)

            # Get structured board state from environment
            structured_board = None
            if role == "system" and initial_board_state:
                # System message may reference initial state
                structured_board = initial_board_state
            elif role == "user" and env_info_idx < len(env_info_list):
                # User messages (env observations) have board states in env_info
                env_info = env_info_list[env_info_idx]
                structured_board = (
                    env_info.get("board_state") if isinstance(env_info, dict) else None
                )
                env_info_idx += 1

            # Verification: compare visual board with structured board
            verification_status = None
            if visual_board and structured_board:
                expected_visual = structured_board.get("board_visual", "")
                # Simple comparison: normalize whitespace
                visual_normalized = " ".join(visual_board.split())
                expected_normalized = " ".join(expected_visual.split())
                verification_status = (
                    "match" if visual_normalized == expected_normalized else "mismatch"
                )

            turn_info = {
                "turn": i,
                "role": role,
                "visual_board_extracted": visual_board,
                "structured_board_state": structured_board,
                "verification_status": verification_status,
                "message_content_preview": content[:300] + "..."
                if len(content) > 300
                else content,
            }

            per_turn_states.append(turn_info)

        return per_turn_states

    def _extract_board_from_content(self, content: str) -> str | None:
        """Extract ASCII board visualization from message content.

        Looks for Gomoku-style board patterns like:
            0 1 2 3 4 5 6 7 8
          0 . . . . . . . . .
          1 . . . . X . . . .
          ...

        Returns:
            The extracted board string, or None if no board found.
        """
        if not content:
            return None

        lines = content.split("\n")
        board_lines = []
        in_board = False

        for line in lines:
            stripped = line.strip()

            # Detect column header line (e.g., "  0 1 2 3 4 5 6 7 8")
            if re.match(r"^\s*\d(\s+\d)+\s*$", stripped):
                in_board = True
                board_lines.append(line)
                continue

            # Detect board row lines (e.g., "0 . . . X . . . . ." or "0 . X O . . . . . .")
            if in_board and re.match(r"^\s*\d\s+[.XO](\s+[.XO])*\s*$", stripped):
                board_lines.append(line)
                continue

            # End board detection on non-matching line after we've started
            if in_board and stripped and not re.match(r"^\s*\d\s+[.XO]", stripped):
                # Check if this is still a valid board line with row number
                if not re.match(r"^\s*\d", stripped):
                    break

        if board_lines:
            return "\n".join(board_lines)

        return None