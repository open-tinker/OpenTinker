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
WMC-ERC: World Model-Conditioned Entropy Regularized Co-evolution.

Dynamic entropy clipping for multi-turn agentic RL. Uses the LLM's own
prediction entropy at environment token positions as a World Model uncertainty
signal (H_WM) to dynamically gate policy gradient updates, preventing
entropy collapse in well-understood regions while permitting exploration
in uncertain ones.

Core idea:
- S_* measures per-turn "blind confidence" of the policy (entropy collapse momentum)
- H_WM measures per-turn World Model uncertainty (prediction entropy at env tokens)
- Dynamic mask m_t: when WM is confident but policy is overconfident → block update
                    when WM is uncertain → allow exploration regardless of confidence
- Entropy floor: when per-turn action entropy falls below a threshold, force mask
  to prevent entropy collapse even when z-score gating degrades

Reference: World Model-Conditioned Entropy Regularized Co-evolution (WMC-ERC)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import (
    compute_turn_boundaries,
)


def compute_s_star(
    old_log_probs: torch.Tensor,
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn policy blind confidence S_*.

    S_*^t = mean over action tokens in turn t of: p_k * (H + log p_k)

    where p_k is the probability of the chosen action token and H is the
    full-distribution entropy. This quantity measures how strongly the
    policy's token-level probability mass is driving entropy downward:
    high S_* means the policy is aggressively collapsing toward a single
    action, creating momentum for entropy collapse.

    Based on first-order Taylor expansion of the entropy discriminator
    (see WMC-ERC algorithm specification).

    Args:
        old_log_probs: (batch_size, response_length) log probs of chosen tokens
        entropys: (batch_size, response_length) entropy of policy distribution
        response_mask: (batch_size, response_length) 1=action, 0=env/pad
        turn_boundaries: per-sample list of (start, end) tuples for action turns

    Returns:
        List of lists of scalar tensors, one S_* per turn per sample.
    """
    batch_size = old_log_probs.shape[0]
    device = old_log_probs.device
    s_star_per_sample = []

    for i in range(batch_size):
        s_star_turns = []
        for start, end in turn_boundaries[i]:
            log_p = old_log_probs[i, start:end]
            H = entropys[i, start:end]
            mask = response_mask[i, start:end]
            count = mask.sum()

            if count > 0:
                p_k = torch.exp(log_p)
                s_token = p_k * (H + log_p)
                s_t = (s_token * mask).sum() / count
            else:
                s_t = torch.tensor(0.0, device=device)

            s_star_turns.append(s_t)
        s_star_per_sample.append(s_star_turns)

    return s_star_per_sample


def compute_per_turn_entropy(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute mean action-token entropy per turn.

    Args:
        entropys: (batch_size, response_length) entropy at all positions
        response_mask: (batch_size, response_length) 1=action, 0=env/pad
        turn_boundaries: per-sample list of (start, end) for action turns

    Returns:
        List of lists of scalar tensors, one mean entropy per turn per sample.
    """
    batch_size = entropys.shape[0]
    device = entropys.device
    ent_per_sample = []

    for i in range(batch_size):
        ent_turns = []
        for start, end in turn_boundaries[i]:
            H = entropys[i, start:end]
            mask = response_mask[i, start:end]
            count = mask.sum()
            if count > 0:
                ent_t = (H * mask).sum() / count
            else:
                ent_t = torch.tensor(0.0, device=device)
            ent_turns.append(ent_t)
        ent_per_sample.append(ent_turns)

    return ent_per_sample


def compute_h_wm(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask_response: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn World Model uncertainty H_WM.

    H_WM^t = mean prediction entropy at env token positions following action turn t.

    Env tokens after turn t represent the environment's response to action a_t.
    The model's entropy at these positions measures how uncertain it is about
    predicting the next state — i.e., the World Model's "cognitive blind spot".

    Higher H_WM → model doesn't understand this environment transition well.
    Lower H_WM → model has seen similar transitions and is confident.

    Args:
        entropys: (batch_size, response_length) entropy at all positions
        response_mask: (batch_size, response_length) 1=action, 0=env/pad
        attention_mask_response: (batch_size, response_length) 1=real, 0=padding
        turn_boundaries: per-sample list of (start, end) for action turns

    Returns:
        List of lists of scalar tensors, one H_WM per turn per sample.
    """
    batch_size = entropys.shape[0]
    seq_len = entropys.shape[1]
    device = entropys.device
    env_mask = attention_mask_response * (1.0 - response_mask)

    h_wm_per_sample = []

    for i in range(batch_size):
        boundaries = turn_boundaries[i]
        h_wm_turns = []

        for t, (start, end) in enumerate(boundaries):
            # Env tokens after this turn: [end, next_turn_start) or [end, seq_len)
            if t + 1 < len(boundaries):
                env_end = boundaries[t + 1][0]
            else:
                env_end = seq_len

            region_mask = env_mask[i, end:env_end]
            region_entropy = entropys[i, end:env_end]
            count = region_mask.sum()

            if count > 0:
                h_wm_t = (region_entropy * region_mask).sum() / count
            else:
                h_wm_t = torch.tensor(0.0, device=device)

            h_wm_turns.append(h_wm_t)

        h_wm_per_sample.append(h_wm_turns)

    return h_wm_per_sample


def compute_dynamic_mask(
    s_star_per_sample: List[List[torch.Tensor]],
    h_wm_per_sample: List[List[torch.Tensor]],
    mu_base: float = 1.0,
    lambda_wm: float = 1.0,
    per_turn_entropy: Optional[List[List[torch.Tensor]]] = None,
    entropy_floor: float = 0.0,
) -> List[List[float]]:
    """Compute per-turn dynamic entropy clipping mask.

    Two-stage gating:

    Stage 1 (z-score): m_t = 1 if |S_*^t - S_bar| <= mu * (1 + lambda * H_WM^t) * sigma
                             0 otherwise

    Stage 2 (entropy floor): if per-turn action entropy < entropy_floor → m_t = 0
      This prevents the degenerate case where z-score gating fails because all
      S_* values cluster near zero during entropy collapse.

    Args:
        s_star_per_sample: per-sample, per-turn S_* tensors
        h_wm_per_sample: per-sample, per-turn H_WM tensors
        mu_base: base clipping coefficient
        lambda_wm: WM uncertainty weight
        per_turn_entropy: per-sample, per-turn mean action entropy (optional)
        entropy_floor: minimum allowed per-turn entropy; turns below are masked

    Returns:
        List of lists of floats (0.0 or 1.0), one mask per turn per sample.
    """
    # Flatten all S_* for batch statistics
    all_s = []
    for turns in s_star_per_sample:
        for s in turns:
            all_s.append(s.detach())

    if len(all_s) == 0:
        return [[] for _ in s_star_per_sample]

    all_s_tensor = torch.stack(all_s)
    s_bar = all_s_tensor.mean()

    # Guard for single-element: std is 0, threshold = mu_base * (1 + lambda * h_wm) * 0
    # → everything would be masked. Use 1.0 as default sigma for single element.
    if len(all_s) <= 1:
        sigma = torch.tensor(1.0, device=all_s_tensor.device)
    else:
        sigma = all_s_tensor.std(unbiased=False) + 1e-8

    mask_per_sample = []
    n_zscore_masked = 0
    n_entropy_floor_masked = 0

    for i in range(len(s_star_per_sample)):
        masks = []
        for t in range(len(s_star_per_sample[i])):
            s_t = s_star_per_sample[i][t].detach()
            h_t = h_wm_per_sample[i][t].detach()

            # Stage 1: z-score gating
            threshold = mu_base * (1.0 + lambda_wm * h_t) * sigma
            if torch.abs(s_t - s_bar) > threshold:
                m_t = 0.0
                n_zscore_masked += 1
            # Stage 2: entropy floor gating
            elif (
                per_turn_entropy is not None
                and entropy_floor > 0
                and i < len(per_turn_entropy)
                and t < len(per_turn_entropy[i])
                and per_turn_entropy[i][t].detach().item() < entropy_floor
            ):
                m_t = 0.0
                n_entropy_floor_masked += 1
            else:
                m_t = 1.0

            masks.append(m_t)
        mask_per_sample.append(masks)

    return mask_per_sample, n_zscore_masked, n_entropy_floor_masked


def apply_wmc_erc(
    batch,
    entropys: torch.Tensor,
    wmc_erc_config,
) -> Tuple[object, Dict[str, float]]:
    """Apply WMC-ERC dynamic entropy clipping to batch advantages.

    Pipeline:
    1. Compute turn boundaries from response_mask
    2. Compute S_* (policy blind confidence) per turn
    3. Compute per-turn action entropy
    4. Compute H_WM (world model uncertainty) per turn from env token entropys
    5. Compute dynamic mask m_t per turn (z-score + entropy floor)
    6. Apply mask to advantages: A_masked = A * m_t (broadcast to tokens)
    7. Return metrics for logging

    Args:
        batch: DataProto or compatible object with batch dict containing
               advantages, response_mask, old_log_probs, attention_mask
        entropys: (batch_size, response_length) stored before pop in train_step
        wmc_erc_config: OmegaConf DictConfig or dict with mu_base, lambda_wm, enable

    Returns:
        (batch, metrics) where batch has masked advantages and metrics dict
    """
    enable = wmc_erc_config.get("enable", True) if hasattr(wmc_erc_config, 'get') else getattr(wmc_erc_config, 'enable', True)
    if not enable:
        return batch, {}

    response_mask = batch.batch["response_mask"]
    old_log_probs = batch.batch["old_log_probs"]
    advantages = batch.batch["advantages"]

    # Compute attention mask for response region
    response_length = advantages.shape[1]
    attention_mask = batch.batch["attention_mask"]
    attention_mask_response = attention_mask[:, -response_length:]

    # 1. Turn boundaries
    turn_boundaries = compute_turn_boundaries(response_mask)

    # 2. S_* per turn
    s_star = compute_s_star(old_log_probs, entropys, response_mask, turn_boundaries)

    # 3. Per-turn action entropy
    turn_entropy = compute_per_turn_entropy(entropys, response_mask, turn_boundaries)

    # 4. H_WM per turn
    h_wm = compute_h_wm(entropys, response_mask, attention_mask_response, turn_boundaries)

    # 5. Dynamic mask (z-score + entropy floor)
    _get = lambda key, default: (
        wmc_erc_config.get(key, default) if hasattr(wmc_erc_config, 'get')
        else getattr(wmc_erc_config, key, default)
    )
    mu_base = float(_get("mu_base", 1.0))
    lambda_wm = float(_get("lambda_wm", 1.0))
    entropy_floor = float(_get("entropy_floor", 0.0))

    mask, n_zscore_masked, n_entropy_floor_masked = compute_dynamic_mask(
        s_star, h_wm, mu_base, lambda_wm,
        per_turn_entropy=turn_entropy,
        entropy_floor=entropy_floor,
    )

    # 6. Apply mask to advantages (in-place)
    batch_size = advantages.shape[0]
    for i in range(batch_size):
        for t, (start, end) in enumerate(turn_boundaries[i]):
            if t < len(mask[i]):
                advantages[i, start:end] *= mask[i][t]
    batch.batch["advantages"] = advantages

    # 7. Adaptive entropy control via beta_token
    # Instead of fixed entropy_coeff, use per-token beta based on turn entropy.
    # - turns with entropy < target → beta = entropy_coeff (encourage exploration)
    # - turns with entropy >= target → beta = 0 (don't push entropy higher)
    # This prevents both entropy collapse AND entropy explosion.
    entropy_target = float(_get("entropy_target", 0.0))
    base_entropy_coeff = float(_get("base_entropy_coeff", 0.0))

    if entropy_target > 0 and base_entropy_coeff > 0:
        beta_token = torch.zeros_like(advantages)
        for i in range(batch_size):
            for t, (start, end) in enumerate(turn_boundaries[i]):
                if t < len(turn_entropy[i]):
                    turn_ent = turn_entropy[i][t].detach().item()
                    if turn_ent < entropy_target:
                        # Linear scaling: full coeff at entropy=0, zero at target
                        scale = max(0.0, 1.0 - turn_ent / entropy_target)
                        beta_token[i, start:end] = base_entropy_coeff * scale
                    # else: beta stays 0 (no entropy bonus for high-entropy turns)
        batch.batch["beta_token"] = beta_token

    # 8. Metrics
    all_s = [s.item() for turns in s_star for s in turns]
    all_h = [h.item() for turns in h_wm for h in turns]
    all_m = [m for turns in mask for m in turns]
    all_te = [e.item() for turns in turn_entropy for e in turns]

    # WM NLL (monitoring only — not in backward pass for this prototype)
    env_mask = attention_mask_response * (1.0 - response_mask)
    env_count = env_mask.sum()
    wm_nll = (-(old_log_probs * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0

    # Beta token stats for logging
    beta_mean = 0.0
    if entropy_target > 0 and base_entropy_coeff > 0:
        active_beta = beta_token[response_mask.bool()]
        beta_mean = active_beta.mean().item() if active_beta.numel() > 0 else 0.0

    metrics = {
        "wmc_erc/s_star_mean": float(np.mean(all_s)) if all_s else 0.0,
        "wmc_erc/s_star_std": float(np.std(all_s)) if all_s else 0.0,
        "wmc_erc/h_wm_mean": float(np.mean(all_h)) if all_h else 0.0,
        "wmc_erc/mask_ratio": float(np.mean(all_m)) if all_m else 1.0,
        "wmc_erc/num_masked_turns": sum(1 for m in all_m if m == 0.0),
        "wmc_erc/total_turns": len(all_m),
        "wmc_erc/wm_nll": wm_nll,
        "wmc_erc/turn_entropy_mean": float(np.mean(all_te)) if all_te else 0.0,
        "wmc_erc/n_zscore_masked": n_zscore_masked,
        "wmc_erc/n_entropy_floor_masked": n_entropy_floor_masked,
        "wmc_erc/adaptive_beta_mean": beta_mean,
    }

    return batch, metrics
