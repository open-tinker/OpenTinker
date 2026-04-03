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
RWML: Reinforcement World Model Learning (arxiv:2602.05842).

Trains the LLM's world-modeling ability by rewarding accurate next-state
predictions using text-level embedding similarity with an external model.

For each action turn t in a multi-turn trajectory:
  - predicted_obs: model's argmax-decoded tokens at observation positions
  - actual_obs: ground-truth observation tokens from the environment
  - d(pred, actual) = 1 - cos(E(pred), E(actual))   [external embedding model]
  - r^WM = 1.0 if d < tau_d else 0.0                 [binary reward]

The rewards are added to per-turn turn_scores and flow through GRPO per-step
advantage computation.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import (
    compute_turn_boundaries,
)


class EmbeddingSimilarityReward:
    """Loads an external text embedding model and computes RWML rewards.

    Uses a HuggingFace sentence-transformers or compatible model to encode
    text strings into embeddings, then measures cosine similarity.
    """

    def __init__(self, model_name_or_path: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.device = device

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into L2-normalized embeddings.

        Args:
            texts: List of text strings.

        Returns:
            Tensor of shape (N, embed_dim), L2-normalized.
        """
        if not texts:
            return torch.empty(0)
        embeddings = self.model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        )
        return F.normalize(embeddings, p=2, dim=1)

    def compute_similarity(
        self, texts_a: List[str], texts_b: List[str]
    ) -> List[float]:
        """Compute pairwise cosine similarity between text pairs.

        Args:
            texts_a: First list of texts.
            texts_b: Second list of texts (same length).

        Returns:
            List of cosine similarity values in [-1, 1].
        """
        assert len(texts_a) == len(texts_b)
        if not texts_a:
            return []
        emb_a = self.encode(texts_a)
        emb_b = self.encode(texts_b)
        sims = (emb_a * emb_b).sum(dim=1)
        return sims.cpu().tolist()

    def compute_reward(
        self, predicted: List[str], actual: List[str], tau_d: float = 0.2
    ) -> Tuple[List[float], List[float]]:
        """Compute binary RWML rewards per the paper.

        r^WM = 1.0 if d(pred, actual) < tau_d else 0.0
        where d = 1 - cos_sim

        Args:
            predicted: Predicted observation texts.
            actual: Actual observation texts.
            tau_d: Distance threshold (default 0.2 per paper).

        Returns:
            (rewards, similarities): Lists of binary rewards and raw similarities.
        """
        similarities = self.compute_similarity(predicted, actual)
        rewards = [1.0 if (1.0 - sim) < tau_d else 0.0 for sim in similarities]
        return rewards, similarities


def decode_per_turn_texts(
    token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[str]]:
    """Decode per-turn observation texts from token IDs.

    For each action turn t (defined by turn_boundaries), observation tokens
    are at positions [end_t, start_{t+1}) in the response portion. This
    function decodes those token spans to text strings.

    Args:
        token_ids: (batch_size, response_length) token IDs (predicted or actual).
        response_mask: (batch_size, response_length) 1=action, 0=env/pad.
        attention_mask: (batch_size, full_seq_length) 1=real, 0=pad.
        tokenizer: Tokenizer for decoding.
        turn_boundaries: Per-sample list of (start, end) for action turns.

    Returns:
        List of lists of decoded observation text strings per turn per sample.
    """
    batch_size = token_ids.shape[0]
    seq_len = token_ids.shape[1]
    resp_len = response_mask.shape[1]
    attn_resp = attention_mask[:, -resp_len:]
    env_mask = attn_resp * (1.0 - response_mask.float())  # 1=env token, 0=action/pad

    all_texts = []
    for i in range(batch_size):
        boundaries = turn_boundaries[i]
        sample_texts = []
        for t, (start, end) in enumerate(boundaries):
            # Observation region: [end, next_turn_start) or [end, seq_len)
            if t + 1 < len(boundaries):
                env_end = boundaries[t + 1][0]
            else:
                env_end = seq_len

            region_mask = env_mask[i, end:env_end]
            region_ids = token_ids[i, end:env_end]

            # Extract valid (non-padding) observation token IDs
            valid_positions = region_mask.bool()
            if valid_positions.sum() == 0:
                sample_texts.append("")
                continue

            valid_ids = region_ids[valid_positions].cpu().tolist()
            text = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
            sample_texts.append(text)
        all_texts.append(sample_texts)

    return all_texts


def compute_rwml_turn_rewards(
    predicted_observations: List[List[str]],
    actual_observations: List[List[str]],
    similarity_reward: EmbeddingSimilarityReward,
    tau_d: float = 0.2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute per-turn RWML rewards for a batch.

    Flattens all per-turn pairs, computes embedding similarity rewards in one
    batch, then reshapes back to per-sample per-turn structure.

    Args:
        predicted_observations: Per-sample, per-turn predicted obs texts.
        actual_observations: Per-sample, per-turn actual obs texts.
        similarity_reward: EmbeddingSimilarityReward instance.
        tau_d: Distance threshold for binary reward.

    Returns:
        rwml_turn_rewards: np.ndarray(batch_size, dtype=object), each element
            is a list of per-turn reward floats.
        metrics: Dict with diagnostic metrics.
    """
    batch_size = len(predicted_observations)

    # Flatten all (predicted, actual) pairs with valid text
    flat_pred = []
    flat_actual = []
    indices = []  # (sample_idx, turn_idx)
    for i in range(batch_size):
        n_turns = min(len(predicted_observations[i]), len(actual_observations[i]))
        for t in range(n_turns):
            pred = predicted_observations[i][t]
            actual = actual_observations[i][t]
            if pred and actual:  # skip empty
                flat_pred.append(pred)
                flat_actual.append(actual)
                indices.append((i, t))

    # Initialize per-turn rewards with 0.0
    rwml_rewards = np.empty(batch_size, dtype=object)
    for i in range(batch_size):
        n_turns = len(actual_observations[i])
        rwml_rewards[i] = [0.0] * n_turns

    if not flat_pred:
        return rwml_rewards, {
            "rwml/mean_reward": 0.0,
            "rwml/mean_similarity": 0.0,
            "rwml/num_valid_pairs": 0,
            "rwml/total_turns": sum(len(a) for a in actual_observations),
        }

    # Batch compute rewards
    rewards, similarities = similarity_reward.compute_reward(
        flat_pred, flat_actual, tau_d
    )

    # Scatter back to per-sample per-turn structure
    for idx, (i, t) in enumerate(indices):
        rwml_rewards[i][t] = rewards[idx]

    metrics = {
        "rwml/mean_reward": float(np.mean(rewards)),
        "rwml/mean_similarity": float(np.mean(similarities)),
        "rwml/num_valid_pairs": len(flat_pred),
        "rwml/total_turns": sum(len(a) for a in actual_observations),
        "rwml/reward_rate": float(np.mean(rewards)) if rewards else 0.0,
    }
    return rwml_rewards, metrics
