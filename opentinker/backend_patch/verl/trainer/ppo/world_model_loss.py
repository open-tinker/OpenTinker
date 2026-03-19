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
"""
World Model SFT Loss — trains the policy to also predict observation tokens.

Joint loss = ppo_loss(action tokens) + wm_coeff * sft_loss(observation tokens)

Implementation:
    The WM SFT loss is computed inside dp_actor.update_policy() (verl modification).
    It is gated by config.world_model_coeff > 0 AND observation_mask being present
    in the batch.

    observation_mask is computed in http_training_server.py before update_actor:
        obs_mask = attention_mask[:, -resp_len:] & ~response_mask

    To enable, set in your config yaml:
        actor_rollout_ref:
          actor:
            world_model_coeff: 0.1
"""

import torch


def compute_observation_mask(batch) -> torch.Tensor:
    """Compute observation_mask from attention_mask and response_mask.

    observation tokens = real tokens in the response portion that are NOT
    action (LLM-generated) tokens.

    Args:
        batch: DataProto with batch["attention_mask"] and batch["response_mask"]

    Returns:
        observation_mask: (batch_size, response_length) float tensor
    """
    resp_len = batch.batch["response_mask"].shape[1]
    attn_response = batch.batch["attention_mask"][:, -resp_len:]
    return (attn_response.bool() & ~batch.batch["response_mask"].bool()).float()