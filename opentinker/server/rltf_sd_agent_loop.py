# Copyright 2026 OpenTinker
#
# Licensed under the Apache License, Version 2.0.
"""RLTF-SD agent loop with same-state dual sampling.

Flow at each decision:
1) sample draft y0 from x_t
2) generate critique c_t + draft score r0 (without stepping env)
3) sample revised action y1 from x'_t = f(x_t, y0, c_t)
4) execute only y1 in environment to obtain r1
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

from verl.utils.profiler import simple_timer

from opentinker.server.generic_agent_loop import (
    GenericAgentData,
    GenericAgentLoop,
    GenericAgentState,
)
from opentinker.server.rltf_sd_feedback import RLTFSdFeedbackGenerator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RLTFSDAgentLoop(GenericAgentLoop):
    """Agent loop implementing RLTF-SD dual-sampling interaction."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        super().init_class(config, tokenizer, processor, **kwargs)

        rltf_sd_cfg = config.algorithm.get("rltf_sd", {})
        cls.rltf_sd_enable = bool(rltf_sd_cfg.get("enable", False))
        # Avoid request_id collisions in vLLM when rollout_n > 1.
        # GenericAgentLoop may use extra_info.sample_id as request_id; this can be
        # duplicated across concurrent samples in on-policy rollout.
        cls.rltf_sd_use_stable_request_id = bool(
            rltf_sd_cfg.get("use_stable_request_id", False)
        )
        cls.rltf_sd_max_pairs_per_episode = int(
            rltf_sd_cfg.get("max_pairs_per_episode", 8)
        )
        cls.rltf_sd_feedback_generator = RLTFSdFeedbackGenerator(
            config=rltf_sd_cfg.get("feedback", {})
        )

        logger.info(
            "[RLTFSDAgentLoop] Initialized (enable=%s, max_pairs_per_episode=%s, use_stable_request_id=%s)",
            cls.rltf_sd_enable,
            cls.rltf_sd_max_pairs_per_episode,
            cls.rltf_sd_use_stable_request_id,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        """Attach RLTF-SD group uid while keeping request_id collision-safe."""
        uid = kwargs.get("uid")
        extra_info = kwargs.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info = copy.deepcopy(extra_info)

        # By default, do not pass stable sample_id to request_id because duplicate
        # sample_id under rollout_n can trigger "Request id ... already running."
        if not self.rltf_sd_use_stable_request_id:
            extra_info.pop("sample_id", None)

        if uid is not None:
            interaction_kwargs = extra_info.get("interaction_kwargs")
            if not isinstance(interaction_kwargs, dict):
                interaction_kwargs = {}
            interaction_kwargs = copy.deepcopy(interaction_kwargs)
            interaction_kwargs.setdefault("rltf_sd_group_uid", str(uid))
            extra_info["interaction_kwargs"] = interaction_kwargs

        kwargs["extra_info"] = extra_info

        return await super().run(sampling_params, **kwargs)

    async def _handle_generating_state(
        self, agent_data: GenericAgentData, sampling_params: dict[str, Any]
    ) -> GenericAgentState:
        """Generate draft -> critique -> revised; only revised is executed."""
        if not self.rltf_sd_enable or agent_data.interaction is None:
            return await super()._handle_generating_state(agent_data, sampling_params)

        total_context_budget = self.prompt_length + self.response_length
        min_generation_tokens = 16
        if len(agent_data.prompt_ids) + min_generation_tokens > total_context_budget:
            logger.warning(
                "[RLTFSDAgentLoop] Context overflow detected: prompt_len=%s, budget=%s",
                len(agent_data.prompt_ids),
                total_context_budget,
            )
            if not agent_data.response_ids:
                eos_token_id = self.tokenizer.eos_token_id
                if eos_token_id is not None:
                    agent_data.response_ids = [eos_token_id]
                    agent_data.prompt_ids.append(eos_token_id)
                    agent_data.response_mask.append(1)
            return GenericAgentState.TERMINATED

        x_t_prompt_ids = list(agent_data.prompt_ids)

        with simple_timer("rltf_sd_generate_draft", agent_data.metrics):
            draft_output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=x_t_prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )
        draft_token_ids = draft_output.token_ids

        if self.max_tokens_per_turn and len(draft_token_ids) > self.max_tokens_per_turn:
            draft_token_ids = draft_token_ids[: self.max_tokens_per_turn]

        draft_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(draft_token_ids, skip_special_tokens=True),
        )

        interaction_name = str(agent_data.interaction_kwargs.get("name", ""))
        feedback = await self.loop.run_in_executor(
            None,
            lambda: self.rltf_sd_feedback_generator.generate_feedback(
                interaction_name=interaction_name,
                messages=agent_data.messages,
                draft_response=draft_text,
                interaction_kwargs=agent_data.interaction_kwargs,
            ),
        )

        revision_prompt = self.rltf_sd_feedback_generator.build_revision_prompt(
            critique=feedback.critique,
            interaction_name=interaction_name,
        )

        revised_messages = copy.deepcopy(agent_data.messages)
        revised_messages.append({"role": "assistant", "content": draft_text})
        revised_messages.append({"role": "user", "content": revision_prompt})

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    revised_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(
                text=[raw_prompt],
                images=agent_data.image_data if agent_data.image_data else None,
                return_tensors="pt",
            )
            revised_prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            revised_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    revised_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        with simple_timer("rltf_sd_generate_revised", agent_data.metrics):
            revised_output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=revised_prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        response_token_ids = revised_output.token_ids
        response_log_probs = revised_output.log_probs
        if self.max_tokens_per_turn and len(response_token_ids) > self.max_tokens_per_turn:
            response_token_ids = response_token_ids[: self.max_tokens_per_turn]
            if response_log_probs:
                response_log_probs = response_log_probs[: self.max_tokens_per_turn]

        agent_data.assistant_turns += 1
        agent_data.response_ids = response_token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if response_log_probs:
            agent_data.response_logprobs += response_log_probs

        pair_list = agent_data.extra_fields.setdefault("rltf_sd_pair", [])
        if len(pair_list) < self.rltf_sd_max_pairs_per_episode:
            turn_id = len(pair_list) + 1
            group_uid = agent_data.interaction_kwargs.get(
                "rltf_sd_group_uid", agent_data.request_id
            )
            feedback_source = None
            feedback_model = None
            if isinstance(feedback.metadata, dict):
                feedback_source = feedback.metadata.get("source")
                feedback_model = feedback.metadata.get("model")
            pair_list.append(
                {
                    "group_id": f"{group_uid}::turn_{turn_id}",
                    "turn_id": turn_id,
                    "interaction_name": interaction_name,
                    "r0": float(feedback.r0),
                    "r1": None,
                    "x_t_token_ids": x_t_prompt_ids,
                    "x_t_prime_token_ids": list(revised_prompt_ids),
                    "y1_token_ids": list(response_token_ids),
                    "feedback": feedback.critique,
                    "feedback_source": feedback_source,
                    "feedback_model": feedback_model,
                }
            )

        if len(agent_data.response_mask) >= self.response_length:
            return GenericAgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns > self.max_assistant_turns:
            return GenericAgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns > self.max_user_turns:
            return GenericAgentState.TERMINATED

        revised_message = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(
                agent_data.response_ids,
                skip_special_tokens=True,
            ),
        )
        agent_data.messages.append({"role": "assistant", "content": revised_message})
        return GenericAgentState.INTERACTING

    async def _handle_interacting_state(
        self, agent_data: GenericAgentData
    ) -> GenericAgentState:
        """Execute only revised action and attach step reward as r1."""
        (
            should_terminate,
            observation,
            reward,
            info,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        if reward is not None:
            agent_data.turn_scores.append(reward)
            pair_list = agent_data.extra_fields.get("rltf_sd_pair", [])
            for pair in reversed(pair_list):
                if pair.get("r1") is None:
                    pair["r1"] = float(reward)
                    break

        if info:
            if "env_info" not in agent_data.extra_fields:
                agent_data.extra_fields["env_info"] = []
            agent_data.extra_fields["env_info"].append(info)

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": observation}]
        agent_data.messages.extend(add_messages)

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
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                ),
            )

        response_ids = response_ids[len(self.system_prompt) :]

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return GenericAgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)

        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        if should_terminate:
            return GenericAgentState.TERMINATED
        return GenericAgentState.GENERATING
