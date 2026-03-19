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

Reference: World Model-Conditioned Entropy Regularized Co-evolution (WMC-ERC)
"""

from typing import Dict, List, Tuple

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
    mu_base: float,
    mu_exp: float,
    eta_wm: float,
    lambda_wm: float,
    s_bar: float,
    sigma: float,
    clipping_method: str = "mask",
) -> List[List[float]]:
    """Compute per-turn dynamic entropy clipping mask or coefficient.

    Logic:
    - WM uncertainty signal: f(H_WM) = eta_wm * exp(-lambda_wm * H_WM)
    - Threshold: threshold = mu * f(H_WM) * sigma
    - Masking: m_t = 0.0 if violation, 1.0 otherwise
    - Clipping: m_t = threshold / deviation if violation, 1.0 otherwise (PPO-style)

    Args:
        s_star_per_sample: per-sample, per-turn S_* tensors
        h_wm_per_sample: per-sample, per-turn H_WM tensors
        mu_base: clipping coefficient for collapsing side
        mu_exp: clipping coefficient for exploration side
        eta_wm: base multiplier for WM uncertainty signal
        lambda_wm: exponential decay factor for WM uncertainty
        s_bar: mean of S_* (batch or global)
        sigma: std of S_* (batch or global)
        clipping_method: "mask" or "clip"

    Returns:
        List of lists of floats, one mask/coeff per turn per sample.
    """
    mask_per_sample = []
    for i in range(len(s_star_per_sample)):
        masks = []
        for t in range(len(s_star_per_sample[i])):
            s_t = s_star_per_sample[i][t].detach().item()
            h_t = h_wm_per_sample[i][t].detach().item()
            
            # WM uncertainty signal: f(H_WM) = eta_wm * exp(-lambda_wm * H_WM)
            h_factor = eta_wm * np.exp(-lambda_wm * h_t)
            
            # Asymmetric threshold calculation
            if s_t > s_bar:
                # Collapsing side
                threshold = mu_base * h_factor * sigma
                diff = s_t - s_bar
                if clipping_method == "mask":
                    m_t = 1.0 if diff <= threshold else 0.0
                else: # PPO-style clipping
                    # If diff > threshold, we scale the advantage by threshold/diff
                    # such that the effective update is capped at threshold
                    m_t = min(1.0, threshold / (diff + 1e-8))
            else:
                # Exploration side
                threshold = mu_exp * h_factor * sigma
                diff = s_bar - s_t
                if clipping_method == "mask":
                    m_t = 1.0 if diff <= threshold else 0.0
                else: # PPO-style clipping
                    m_t = min(1.0, threshold / (diff + 1e-8))
                
            masks.append(m_t)
        mask_per_sample.append(masks)

    return mask_per_sample


def apply_wmc_erc(
    batch,
    entropys: torch.Tensor,
    wmc_erc_config,
    running_stats: Dict[str, float],
) -> Tuple[object, Dict[str, float]]:
    """Apply WMC-ERC dynamic entropy clipping to batch advantages.

    Args:
        batch: DataProto or compatible object
        entropys: (batch_size, response_length)
        wmc_erc_config: OmegaConf DictConfig or dict
        running_stats: Dictionary for global running statistics

    Returns:
        (batch, metrics)
    """
    enable = wmc_erc_config.get("enable", True) if hasattr(wmc_erc_config, 'get') else getattr(wmc_erc_config, 'enable', True)
    if not enable:
        return batch, {}

    clipping_type = wmc_erc_config.get("clipping_type", "batch") if hasattr(wmc_erc_config, 'get') else getattr(wmc_erc_config, 'clipping_type', "batch")
    clipping_method = wmc_erc_config.get("clipping_method", "mask") if hasattr(wmc_erc_config, 'get') else getattr(wmc_erc_config, 'clipping_method', "mask")

    response_mask = batch.batch["response_mask"]
    old_log_probs = batch.batch["old_log_probs"]
    advantages = batch.batch["advantages"]
    response_length = advantages.shape[1]
    attention_mask = batch.batch["attention_mask"]
    attention_mask_response = attention_mask[:, -response_length:]

    # 1. Turn boundaries
    turn_boundaries = compute_turn_boundaries(response_mask)

    # 2. Compute S_* and H_WM per turn
    s_star = compute_s_star(old_log_probs, entropys, response_mask, turn_boundaries)
    h_wm = compute_h_wm(entropys, response_mask, attention_mask_response, turn_boundaries)

    # Calculate batch statistics
    all_s = [s.item() for turns in s_star for s in turns]
    all_h = [h.item() for turns in h_wm for h in turns]

    if not all_s:
        return batch, {}

    batch_s_bar = np.mean(all_s)
    batch_s_std = np.std(all_s) + 1e-8
    batch_h_bar = np.mean(all_h) + 1e-8
    
    # Update global statistics
    momentum = wmc_erc_config.get("momentum", 0.9) if hasattr(wmc_erc_config, 'get') else getattr(wmc_erc_config, 'momentum', 0.9)
    if not running_stats.get("initialized", False):
        running_stats["s_bar"] = batch_s_bar
        running_stats["s_std"] = batch_s_std
        running_stats["h_bar"] = batch_h_bar
        running_stats["initialized"] = True
    else:
        running_stats["s_bar"] = (1 - momentum) * batch_s_bar + momentum * running_stats["s_bar"]
        running_stats["s_std"] = (1 - momentum) * batch_s_std + momentum * running_stats["s_std"]
        running_stats["h_bar"] = (1 - momentum) * batch_h_bar + momentum * running_stats["h_bar"]

    # Select statistics for masking
    if clipping_type == "global":
        use_s_bar = running_stats["s_bar"]
        use_s_std = running_stats["s_std"]
        use_h_bar = running_stats["h_bar"]
    else:
        use_s_bar = batch_s_bar
        use_s_std = batch_s_std
        use_h_bar = batch_h_bar

    # 4. Dynamic mask/clip
    mu_base = float(wmc_erc_config.get("mu_base", 1.0) if hasattr(wmc_erc_config, 'get') else getattr(wmc_erc_config, 'mu_base', 1.0))
    mu_exp = float(wmc_erc_config.get("mu_exp", 2.0) if hasattr(wmc_erc_config, 'get') else getattr(wmc_erc_config, 'mu_exp', 2.0))
    eta_wm = float(wmc_erc_config.get("eta_wm", 1.0) if hasattr(wmc_erc_config, 'get') else getattr(wmc_erc_config, 'eta_wm', 1.0))
    lambda_wm = float(wmc_erc_config.get("lambda_wm", 1.0) if hasattr(wmc_erc_config, 'get') else getattr(wmc_erc_config, 'lambda_wm', 1.0))
    
    mask = compute_dynamic_mask(
        s_star, h_wm, mu_base, mu_exp, eta_wm, lambda_wm,
        s_bar=use_s_bar,
        sigma=use_s_std,
        clipping_method=clipping_method
    )

    # 5. Apply mask/coeff to advantages
    batch_size = advantages.shape[0]
    for i in range(batch_size):
        for t, (start, end) in enumerate(turn_boundaries[i]):
            if t < len(mask[i]):
                advantages[i, start:end] *= mask[i][t]
    batch.batch["advantages"] = advantages

    # 6. Metrics
    all_m = [m for turns in mask for m in turns]
    
    num_collapsing_violated = 0
    num_exploration_violated = 0
    for i in range(len(s_star)):
        for t in range(len(s_star[i])):
            if mask[i][t] < 1.0:
                if s_star[i][t].item() > use_s_bar:
                    num_collapsing_violated += 1
                else:
                    num_exploration_violated += 1

    env_mask = attention_mask_response * (1.0 - response_mask)
    env_count = env_mask.sum()
    wm_nll = (-(old_log_probs * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0

    metrics = {
        "wmc_erc/batch_s_bar": float(batch_s_bar),
        "wmc_erc/batch_s_std": float(batch_s_std),
        "wmc_erc/batch_h_bar": float(batch_h_bar),
        "wmc_erc/running_s_bar": float(running_stats["s_bar"]),
        "wmc_erc/running_s_std": float(running_stats["s_std"]),
        "wmc_erc/running_h_bar": float(running_stats["h_bar"]),
        "wmc_erc/mask_ratio": float(np.mean(all_m)) if all_m else 1.0,
        "wmc_erc/num_violated_turns": sum(1 for m in all_m if m < 1.0),
        "wmc_erc/num_collapsing_violated": num_collapsing_violated,
        "wmc_erc/num_exploration_violated": num_exploration_violated,
        "wmc_erc/total_turns": len(all_m),
        "wmc_erc/wm_nll": wm_nll,
    }

    return batch, metrics
