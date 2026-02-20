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
Per-Step Core Algorithms for Multi-Turn GRPO.

This module implements per-step/per-turn advantage estimation for multi-turn
agentic tasks, inspired by GTPO (Group Turn Policy Optimization) from arXiv:2511.17052.

Key Algorithm: grpo_per_step
- Uses cumulative returns from each turn to the end of the trajectory
- Provides fine-grained credit assignment for multi-turn interactions
- Falls back to standard GRPO when turn_scores is not available (single-turn case)
"""

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch


def compute_turn_boundaries(
    response_mask: torch.Tensor,
) -> List[List[Tuple[int, int]]]:
    """
    Detect turn boundaries from response_mask.

    The response_mask contains:
    - 1 for LLM-generated tokens
    - 0 for environment observation tokens and padding

    Each contiguous segment of 1s represents one LLM generation turn.

    Args:
        response_mask: Shape (batch_size, response_length), binary mask

    Returns:
        List of lists of (start, end) tuples for each sample.
        Each tuple represents the start (inclusive) and end (exclusive) indices
        of a turn's tokens in the response.
    """
    batch_size = response_mask.shape[0]
    all_boundaries = []

    for i in range(batch_size):
        mask = response_mask[i].cpu().numpy()
        boundaries = []
        in_turn = False
        start_idx = 0

        for j, val in enumerate(mask):
            if val == 1 and not in_turn:
                # Start of a new turn
                in_turn = True
                start_idx = j
            elif val == 0 and in_turn:
                # End of current turn
                in_turn = False
                boundaries.append((start_idx, j))

        # Handle case where sequence ends with mask=1
        if in_turn:
            boundaries.append((start_idx, len(mask)))

        all_boundaries.append(boundaries)

    return all_boundaries


def compute_cumulative_returns(
    turn_scores: List[float],
    gamma: float = 1.0,
) -> List[float]:
    """
    Compute cumulative returns from each turn to the end.

    Return_t = r_t + gamma * Return_{t+1}

    Args:
        turn_scores: List of per-turn rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor (default 1.0 for no discounting)

    Returns:
        List of cumulative returns [R_0, R_1, ..., R_T]
    """
    if not turn_scores:
        return []

    returns = [0.0] * len(turn_scores)
    cumulative = 0.0

    for t in reversed(range(len(turn_scores))):
        cumulative = turn_scores[t] + gamma * cumulative
        returns[t] = cumulative

    return returns


def compute_grpo_per_step_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    turn_scores: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO with per-step (per-turn) credit assignment.

    This implements a return-based advantage estimation inspired by GTPO
    (arXiv:2511.17052), where each turn's advantage is based on the cumulative
    return from that turn to the end of the trajectory.

    For single-turn trajectories or when turn_scores is not available,
    this falls back to standard GRPO behavior.

    Algorithm:
    1. Compute cumulative returns for each turn: R_t = r_t + γ*R_{t+1}
    2. Group samples by prompt (index)
    3. Normalize returns within each group: A_t = (R_t - μ_g) / (σ_g + ε)
    4. Broadcast turn-level advantages to token positions

    Args:
        token_level_rewards: Shape (batch_size, response_length)
            Token-level rewards (used for fallback to standard GRPO)
        response_mask: Shape (batch_size, response_length)
            Binary mask: 1 for LLM tokens, 0 for env/padding tokens
        index: Shape (batch_size,)
            Group index for each sample (samples with same index are from same prompt)
        turn_scores: Shape (batch_size,), dtype=object
            Array of lists, where each list contains per-turn rewards for that sample.
            If None or empty, falls back to standard GRPO.
        gamma: Discount factor for computing cumulative returns (default 1.0)
        epsilon: Small value for numerical stability
        norm_adv_by_std_in_grpo: Whether to normalize by std (True) or just subtract mean (False)

    Returns:
        advantages: Shape (batch_size, response_length)
            Per-token advantages (same value for all tokens within a turn)
        returns: Shape (batch_size, response_length)
            Same as advantages (for consistency with GRPO interface)
    """
    batch_size, response_length = token_level_rewards.shape
    device = token_level_rewards.device

    # Check if turn_scores is available and contains valid multi-turn data
    has_valid_turn_scores = (
        turn_scores is not None
        and len(turn_scores) == batch_size
        and any(
            isinstance(ts, (list, np.ndarray)) and len(ts) > 0 for ts in turn_scores
        )
    )

    if not has_valid_turn_scores:
        # Fallback to standard GRPO (outcome-based)
        # This ensures single-turn equivalence
        return _fallback_to_standard_grpo(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )

    # Detect turn boundaries from response_mask
    turn_boundaries = compute_turn_boundaries(response_mask)

    # Compute cumulative returns for each sample
    all_returns = []
    for i in range(batch_size):
        ts = turn_scores[i]
        if isinstance(ts, (list, np.ndarray)) and len(ts) > 0:
            returns = compute_cumulative_returns(list(ts), gamma=gamma)
        else:
            # Single-turn or no turn scores: use total reward from token_level_rewards
            total_reward = (token_level_rewards[i] * response_mask[i]).sum().item()
            returns = [total_reward]
        all_returns.append(returns)

    # Group samples by index and compute group statistics
    # We use the FIRST return (R_0) as the "outcome" for grouping
    # This represents the total return of the trajectory
    id2returns = defaultdict(list)
    id2indices = defaultdict(list)

    for i in range(batch_size):
        idx = index[i]
        # Use the first return (R_0 = total return) for group normalization
        outcome_return = all_returns[i][0] if all_returns[i] else 0.0
        id2returns[idx].append(outcome_return)
        id2indices[idx].append(i)

    # Compute group mean and std
    id2mean = {}
    id2std = {}
    for idx in id2returns:
        returns_tensor = torch.tensor(
            id2returns[idx], dtype=torch.float32, device=device
        )
        if len(returns_tensor) == 1:
            id2mean[idx] = torch.tensor(0.0, device=device)
            id2std[idx] = torch.tensor(1.0, device=device)
        else:
            id2mean[idx] = returns_tensor.mean()
            id2std[idx] = returns_tensor.std()

    # Compute per-token advantages
    advantages = torch.zeros_like(token_level_rewards)

    for i in range(batch_size):
        idx = index[i]
        group_mean = id2mean[idx]
        group_std = id2std[idx]

        boundaries = turn_boundaries[i]
        returns = all_returns[i]

        # Match turns with their returns
        # If fewer returns than turns, use the last return for remaining turns
        # If fewer turns than returns, ignore extra returns
        for t, (start, end) in enumerate(boundaries):
            if t < len(returns):
                return_t = returns[t]
            else:
                # Use last available return
                return_t = returns[-1] if returns else 0.0

            # Normalize return to get advantage
            if norm_adv_by_std_in_grpo:
                adv = (return_t - group_mean) / (group_std + epsilon)
            else:
                adv = return_t - group_mean

            # Broadcast to all tokens in this turn
            advantages[i, start:end] = adv

    # Apply response mask
    advantages = advantages * response_mask

    return advantages, advantages


def _fallback_to_standard_grpo(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback to standard GRPO when turn_scores is not available.

    This ensures single-turn equivalence with the standard GRPO algorithm.
    """
    scores = token_level_rewards.sum(dim=-1)  # (batch_size,)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    device = token_level_rewards.device

    with torch.no_grad():
        batch_size = scores.shape[0]
        for i in range(batch_size):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=device)
                id2std[idx] = torch.tensor(1.0, device=device)
            else:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = scores_tensor.mean()
                id2std[idx] = scores_tensor.std()

        for i in range(batch_size):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        advantages = scores.unsqueeze(-1) * response_mask

    return advantages, advantages


def incorporate_kl_penalty_in_advantage(
    data,
    kl_penalty_coef: float,
):
    """Add per-token KL penalty to advantages for on-policy distillation.

    Can be applied on top of any base advantage estimator (GRPO, GAE/PPO, etc.):
        A_final_t = A_base_t - α * (log π_student(a_t) - log π_teacher(a_t))

    Adding KL *after* advantage normalization keeps the task signal scale intact
    while the KL term acts as an independent penalty for teacher divergence.

    Args:
        data: DataProto with keys:
            - advantages: base advantages from any estimator (batch, response_len)
            - old_log_probs: student log-probs at training time (batch, response_len)
            - ref_log_prob: teacher/reference log-probs (batch, response_len)
            - response_mask: 1 for LLM tokens, 0 for env/padding tokens
        kl_penalty_coef: α — weight of KL penalty relative to task advantage

    Returns:
        data with batch["advantages"] and batch["returns"] updated in-place.
    """
    advantages = data.batch["advantages"]
    old_log_probs = data.batch["old_log_probs"]   # log π_student
    ref_log_probs = data.batch["ref_log_prob"]     # log π_teacher
    response_mask = data.batch["response_mask"]

    # Reverse KL per token: positive when student diverges from teacher
    kl = (old_log_probs - ref_log_probs) * response_mask

    # Subtract α * KL from advantages (divergence lowers advantage)
    data.batch["advantages"] = advantages - kl_penalty_coef * kl
    data.batch["returns"] = data.batch["advantages"]

    return data
