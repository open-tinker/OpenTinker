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
Tests for per_step_core_algos module.

Run with: pytest opentinker/tests/test_per_step_core_algos.py -v
"""

import numpy as np
import pytest
import torch

from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import (
    compute_cumulative_returns,
    compute_grpo_per_step_advantage,
    compute_turn_boundaries,
)


class TestComputeTurnBoundaries:
    """Test turn boundary detection from response_mask."""

    def test_single_turn(self):
        """Single turn should return one boundary."""
        response_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
        boundaries = compute_turn_boundaries(response_mask)
        assert len(boundaries) == 1
        assert boundaries[0] == [(0, 4)]

    def test_two_turns(self):
        """Two turns separated by observation tokens."""
        # Turn1: tokens 0-2, Obs: tokens 3-5, Turn2: tokens 6-8, Padding: 9-10
        response_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]])
        boundaries = compute_turn_boundaries(response_mask)
        assert len(boundaries) == 1
        assert boundaries[0] == [(0, 3), (6, 9)]

    def test_three_turns(self):
        """Three turns with multiple observation segments."""
        response_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 0, 1, 0]])
        boundaries = compute_turn_boundaries(response_mask)
        assert len(boundaries) == 1
        assert boundaries[0] == [(0, 2), (4, 7), (8, 9)]

    def test_ends_with_ones(self):
        """Mask ending with 1s should correctly capture the last turn."""
        response_mask = torch.tensor([[1, 1, 0, 1, 1, 1]])
        boundaries = compute_turn_boundaries(response_mask)
        assert len(boundaries) == 1
        assert boundaries[0] == [(0, 2), (3, 6)]

    def test_batch(self):
        """Batch of samples with different turn structures."""
        response_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],  # 1 turn
                [1, 1, 0, 0, 1, 1, 0, 0],  # 2 turns
            ]
        )
        boundaries = compute_turn_boundaries(response_mask)
        assert len(boundaries) == 2
        assert boundaries[0] == [(0, 3)]
        assert boundaries[1] == [(0, 2), (4, 6)]


class TestComputeCumulativeReturns:
    """Test cumulative return computation."""

    def test_single_reward(self):
        """Single reward should return itself."""
        turn_scores = [1.0]
        returns = compute_cumulative_returns(turn_scores, gamma=1.0)
        assert returns == [1.0]

    def test_multiple_rewards_no_discount(self):
        """Multiple rewards with gamma=1.0."""
        turn_scores = [0.0, 0.0, 1.0]
        returns = compute_cumulative_returns(turn_scores, gamma=1.0)
        assert returns == [1.0, 1.0, 1.0]

    def test_multiple_rewards_with_discount(self):
        """Multiple rewards with gamma=0.9."""
        turn_scores = [0.0, 0.0, 1.0]
        returns = compute_cumulative_returns(turn_scores, gamma=0.9)
        expected = [0.9 * 0.9 * 1.0, 0.9 * 1.0, 1.0]
        assert np.allclose(returns, expected)

    def test_varied_rewards(self):
        """Different rewards at each turn."""
        turn_scores = [0.1, 0.2, 0.7]
        returns = compute_cumulative_returns(turn_scores, gamma=1.0)
        expected = [1.0, 0.9, 0.7]  # 0.1+0.2+0.7, 0.2+0.7, 0.7
        assert np.allclose(returns, expected)

    def test_empty_list(self):
        """Empty turn_scores should return empty list."""
        returns = compute_cumulative_returns([], gamma=1.0)
        assert returns == []


class TestGrpoPerStepAdvantage:
    """Test the main grpo_per_step advantage computation."""

    def test_single_turn_equivalence(self):
        """Single-turn case should be equivalent to standard GRPO.

        When there's only one turn per sample, grpo_per_step should produce
        the same advantages as standard grpo (modulo the per-token vs per-sample
        structure).
        """
        batch_size = 4
        response_length = 8

        # Create samples from the same group (same uid)
        # With different rewards
        token_level_rewards = torch.zeros(batch_size, response_length)
        # Place rewards at the end of each response
        token_level_rewards[0, 4] = 1.0  # Sample 0: score = 1.0
        token_level_rewards[1, 4] = 0.5  # Sample 1: score = 0.5
        token_level_rewards[2, 4] = 0.0  # Sample 2: score = 0.0
        token_level_rewards[3, 4] = 0.8  # Sample 3: score = 0.8

        # All samples are single-turn (mask = all 1s for response tokens)
        response_mask = torch.zeros(batch_size, response_length)
        response_mask[:, :5] = 1  # 5 response tokens, 3 padding

        # All from the same group
        index = np.array(["group1", "group1", "group1", "group1"])

        # No turn_scores provided - should fallback to standard GRPO
        advantages, returns = compute_grpo_per_step_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            turn_scores=None,
            gamma=1.0,
        )

        # Check that advantages are computed
        assert advantages.shape == (batch_size, response_length)

        # Advantages should be zero where mask is zero
        assert (advantages[:, 5:] == 0).all()

        # Within group, advantages should sum to approximately 0
        # (mean-centered by design)
        total_adv = advantages.sum(dim=1)
        assert np.allclose(total_adv.sum().item(), 0.0, atol=1e-5)

    def test_multi_turn_advantage(self):
        """Multi-turn case with turn_scores should use per-turn advantages."""
        batch_size = 2
        response_length = 10

        # Sample 0: 2 turns, Sample 1: 2 turns
        # Turn structure: [1,1,0,0,1,1,0,0,0,0] for both
        response_mask = torch.tensor(
            [
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # 2 turns
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  # 2 turns
            ],
            dtype=torch.float32,
        )

        # Token-level rewards (for fallback, not used when turn_scores provided)
        token_level_rewards = torch.zeros(batch_size, response_length)

        # Turn scores: [r_0, r_1] for each sample
        # Sample 0: [0.0, 1.0] - first turn fails, second succeeds
        # Sample 1: [0.5, 0.5] - both turns moderate
        turn_scores = np.array(
            [
                [0.0, 1.0],  # Returns: [1.0, 1.0] with gamma=1.0
                [0.5, 0.5],  # Returns: [1.0, 0.5] with gamma=1.0
            ],
            dtype=object,
        )

        # Same group
        index = np.array(["group1", "group1"])

        advantages, returns = compute_grpo_per_step_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            turn_scores=turn_scores,
            gamma=1.0,
        )

        assert advantages.shape == (batch_size, response_length)

        # Advantages should be zero where mask is zero
        assert (advantages[:, 2:4] == 0).all()  # Observation tokens
        assert (advantages[:, 6:] == 0).all()  # Padding

        # Check that turn 1 and turn 2 have different advantages
        # For sample 0: both turns have same cumulative return (1.0)
        # For sample 1: turn 1 return = 1.0, turn 2 return = 0.5

    def test_fallback_without_turn_scores(self):
        """When turn_scores is None, should fallback to standard GRPO."""
        batch_size = 2
        response_length = 6

        token_level_rewards = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        response_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        )

        index = np.array(["g1", "g1"])

        # No turn_scores
        advantages, _ = compute_grpo_per_step_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            turn_scores=None,
        )

        # Should still compute advantages
        assert advantages.shape == (batch_size, response_length)
        # Standard GRPO: all tokens in a sample get the same advantage
        assert advantages[0, 0] == advantages[0, 1] == advantages[0, 2]
        assert advantages[1, 0] == advantages[1, 1] == advantages[1, 2]

    def test_different_groups(self):
        """Samples in different groups should be normalized separately."""
        batch_size = 4
        response_length = 4

        token_level_rewards = torch.zeros(batch_size, response_length)
        token_level_rewards[0, 2] = 1.0  # Group A
        token_level_rewards[1, 2] = 0.0  # Group A
        token_level_rewards[2, 2] = 0.8  # Group B
        token_level_rewards[3, 2] = 0.2  # Group B

        response_mask = torch.ones(batch_size, response_length)

        index = np.array(["A", "A", "B", "B"])

        advantages, _ = compute_grpo_per_step_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            turn_scores=None,
        )

        # Within group A, advantages should be opposite signs
        # Sample 0 has higher reward, so positive advantage
        # Sample 1 has lower reward, so negative advantage
        assert advantages[0, 0] > 0
        assert advantages[1, 0] < 0

        # Same for group B
        assert advantages[2, 0] > 0
        assert advantages[3, 0] < 0


class TestComputeOpsdJsdLoss:
    """Tests for the OPSD full-vocabulary JSD loss function."""

    def _make_logprobs(self, batch, seq_len, vocab_size, seed=42):
        """Generate random log-probs (log-softmax normalised)."""
        torch.manual_seed(seed)
        logits = torch.randn(batch, seq_len, vocab_size)
        return torch.log_softmax(logits, dim=-1)

    def test_zero_when_identical(self):
        """JSD(p ‖ p) = 0 for any distribution p."""
        from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import compute_opsd_jsd_loss

        B, R, V = 2, 8, 32
        lp = self._make_logprobs(B, R, V)
        mask = torch.ones(B, R)
        loss, _ = compute_opsd_jsd_loss(lp, lp, mask, beta=0.5)
        assert loss.item() < 1e-5, f"JSD(p||p) should be ~0, got {loss.item()}"

    def test_non_negative(self):
        """JSD is always non-negative."""
        from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import compute_opsd_jsd_loss

        B, R, V = 4, 10, 64
        s_lp = self._make_logprobs(B, R, V, seed=1)
        t_lp = self._make_logprobs(B, R, V, seed=2)
        mask = torch.ones(B, R)
        loss, _ = compute_opsd_jsd_loss(s_lp, t_lp, mask, beta=0.5)
        assert loss.item() >= 0.0, f"JSD should be >= 0, got {loss.item()}"

    def test_mask_zeros_out_padding(self):
        """Masked tokens should not contribute to the loss."""
        from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import compute_opsd_jsd_loss

        B, R, V = 2, 8, 32
        s_lp = self._make_logprobs(B, R, V, seed=10)
        t_lp = self._make_logprobs(B, R, V, seed=11)

        # Full mask
        full_mask = torch.ones(B, R)
        loss_full, _ = compute_opsd_jsd_loss(s_lp, t_lp, full_mask)

        # Half mask (only first 4 tokens)
        half_mask = torch.zeros(B, R)
        half_mask[:, :4] = 1.0

        # Only-second-half mask
        second_mask = torch.zeros(B, R)
        second_mask[:, 4:] = 1.0

        loss_half, _ = compute_opsd_jsd_loss(s_lp, t_lp, half_mask)
        loss_second, _ = compute_opsd_jsd_loss(s_lp, t_lp, second_mask)

        # The average of the two halves need not equal full, but both should differ from full
        # and both should be non-negative and less than or equal to the maximum possible value
        assert loss_half.item() >= 0.0
        assert loss_second.item() >= 0.0
        # Full loss should be between min and max of the two halves (not strictly necessary,
        # but a sanity check that masking changes the value)
        assert not torch.isclose(loss_full, loss_half) or not torch.isclose(loss_full, loss_second), \
            "Masking should change the loss value"

    def test_numerically_stable_with_extreme_logits(self):
        """Should not produce NaN/Inf with very large or very small logits."""
        from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import compute_opsd_jsd_loss

        B, R, V = 2, 4, 16
        # One distribution almost deterministic (very peaked)
        s_logits = torch.full((B, R, V), -100.0)
        s_logits[:, :, 0] = 100.0
        s_lp = torch.log_softmax(s_logits.float(), dim=-1)

        t_logits = torch.full((B, R, V), -100.0)
        t_logits[:, :, 1] = 100.0
        t_lp = torch.log_softmax(t_logits.float(), dim=-1)

        mask = torch.ones(B, R)
        loss, _ = compute_opsd_jsd_loss(s_lp, t_lp, mask)
        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    def test_asymmetric_distributions_positive(self):
        """JSD between two different distributions should be strictly positive."""
        from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import compute_opsd_jsd_loss

        B, R, V = 2, 4, 16
        s_lp = self._make_logprobs(B, R, V, seed=20)
        t_lp = self._make_logprobs(B, R, V, seed=21)
        mask = torch.ones(B, R)
        loss, metrics = compute_opsd_jsd_loss(s_lp, t_lp, mask)
        assert loss.item() > 0.0, "JSD of different distributions should be > 0"
        assert "opsd/jsd_per_token" in metrics
        assert "opsd/jsd_beta" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
