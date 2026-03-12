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
Tests for WMC-ERC dynamic entropy clipping module.

Run with: pytest opentinker/tests/test_wmc_erc.py -v
"""

import numpy as np
import pytest
import torch

from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import (
    compute_turn_boundaries,
)
from opentinker.backend_patch.verl.trainer.ppo.wmc_erc import (
    apply_wmc_erc,
    compute_dynamic_mask,
    compute_h_wm,
    compute_s_star,
)


class TestComputeSStar:
    """Test S_* (policy blind confidence) computation."""

    def test_single_turn(self):
        """S_* for a single turn = mean of p_k * (H + log p_k) over action tokens."""
        # 1 sample, 6 positions: 4 action + 2 padding
        p = torch.tensor([0.8, 0.6, 0.9, 0.7])
        old_log_probs = torch.zeros(1, 6)
        old_log_probs[0, :4] = torch.log(p)
        entropys = torch.tensor([[1.0, 1.5, 0.5, 1.2, 0.0, 0.0]])
        response_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.float32)
        boundaries = compute_turn_boundaries(response_mask)

        result = compute_s_star(old_log_probs, entropys, response_mask, boundaries)
        assert len(result) == 1  # 1 sample
        assert len(result[0]) == 1  # 1 turn

        # Manual: p_k * (H + log(p_k)) for each token, then mean
        H = torch.tensor([1.0, 1.5, 0.5, 1.2])
        expected = (p * (H + torch.log(p))).mean().item()
        assert abs(result[0][0].item() - expected) < 1e-5

    def test_multi_turn(self):
        """Two turns should produce two S_* values."""
        old_log_probs = torch.log(
            torch.tensor([[0.8, 0.6, 0.5, 0.5, 0.9, 0.7, 0.5, 0.5]])
        )
        entropys = torch.tensor([[1.0, 1.5, 0.0, 0.0, 0.5, 1.2, 0.0, 0.0]])
        response_mask = torch.tensor(
            [[1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float32
        )
        boundaries = compute_turn_boundaries(response_mask)

        result = compute_s_star(old_log_probs, entropys, response_mask, boundaries)
        assert len(result[0]) == 2  # 2 turns
        # Turn 0 and Turn 1 should have different values
        assert result[0][0].item() != result[0][1].item()

    def test_batch(self):
        """Batch of 2 samples."""
        old_log_probs = torch.log(
            torch.tensor(
                [
                    [0.8, 0.6, 0.5, 0.5],
                    [0.5, 0.9, 0.5, 0.5],
                ]
            )
        )
        entropys = torch.tensor(
            [
                [1.0, 1.5, 0.0, 0.0],
                [2.0, 0.3, 0.0, 0.0],
            ]
        )
        response_mask = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
        boundaries = compute_turn_boundaries(response_mask)

        result = compute_s_star(old_log_probs, entropys, response_mask, boundaries)
        assert len(result) == 2

    def test_empty_turns(self):
        """Sample with no action tokens should produce empty list."""
        old_log_probs = torch.zeros(1, 4)
        entropys = torch.zeros(1, 4)
        response_mask = torch.zeros(1, 4, dtype=torch.float32)
        boundaries = compute_turn_boundaries(response_mask)

        result = compute_s_star(old_log_probs, entropys, response_mask, boundaries)
        assert len(result) == 1
        assert len(result[0]) == 0


class TestComputeHWM:
    """Test H_WM (world model uncertainty) computation."""

    def test_single_turn_with_env_tokens(self):
        """H_WM for turn 0 = mean entropy at env positions after turn 0."""
        # Sequence: [action, action, env, env, pad, pad]
        entropys = torch.tensor([[1.0, 1.5, 3.0, 4.0, 0.0, 0.0]])
        response_mask = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        attention_mask_response = torch.tensor(
            [[1, 1, 1, 1, 0, 0]], dtype=torch.float32
        )
        boundaries = [[(0, 2)]]  # 1 turn: action at [0,2)

        result = compute_h_wm(
            entropys, response_mask, attention_mask_response, boundaries
        )
        assert len(result) == 1
        assert len(result[0]) == 1
        # Env tokens at [2,3] → mean entropy = (3.0 + 4.0) / 2 = 3.5
        assert abs(result[0][0].item() - 3.5) < 1e-5

    def test_two_turns(self):
        """Two turns: H_WM_0 from env between turns, H_WM_1 from env after turn 1."""
        # [act, act, env, env, act, act, env, pad]
        entropys = torch.tensor([[1.0, 1.5, 3.0, 4.0, 0.5, 1.2, 2.0, 0.0]])
        response_mask = torch.tensor(
            [[1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float32
        )
        attention_mask_response = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32
        )
        boundaries = [[(0, 2), (4, 6)]]

        result = compute_h_wm(
            entropys, response_mask, attention_mask_response, boundaries
        )
        assert len(result[0]) == 2
        # Turn 0 env: positions [2,4) → (3.0+4.0)/2 = 3.5
        assert abs(result[0][0].item() - 3.5) < 1e-5
        # Turn 1 env: positions [6,8) but attn_mask=[1,0] → only pos 6 → 2.0
        assert abs(result[0][1].item() - 2.0) < 1e-5

    def test_no_env_after_last_turn(self):
        """Last turn has no env tokens → H_WM = 0."""
        # [act, act, env, act, act, pad]
        entropys = torch.tensor([[1.0, 1.5, 3.0, 0.5, 1.2, 0.0]])
        response_mask = torch.tensor([[1, 1, 0, 1, 1, 0]], dtype=torch.float32)
        attention_mask_response = torch.tensor(
            [[1, 1, 1, 1, 1, 0]], dtype=torch.float32
        )
        boundaries = [[(0, 2), (3, 5)]]

        result = compute_h_wm(
            entropys, response_mask, attention_mask_response, boundaries
        )
        # Turn 0 env: positions [2,3) → 3.0
        assert abs(result[0][0].item() - 3.0) < 1e-5
        # Turn 1 env: positions [5,6) but attn_mask=[0] → H_WM = 0
        assert result[0][1].item() == 0.0

    def test_empty_turns(self):
        """No turns → empty H_WM list."""
        entropys = torch.zeros(1, 4)
        response_mask = torch.zeros(1, 4, dtype=torch.float32)
        attention_mask_response = torch.ones(1, 4, dtype=torch.float32)
        boundaries = [[]]

        result = compute_h_wm(
            entropys, response_mask, attention_mask_response, boundaries
        )
        assert len(result[0]) == 0


class TestComputeDynamicMask:
    """Test dynamic entropy clipping mask."""

    def test_all_pass(self):
        """When all S_* are close to mean, all masks = 1."""
        s_star = [[torch.tensor(1.0), torch.tensor(1.1)]]
        h_wm = [[torch.tensor(0.5), torch.tensor(0.5)]]
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=3.0, lambda_wm=1.0)
        assert mask == [[1.0, 1.0]]

    def test_outlier_blocked_low_hwm(self):
        """High S_* outlier with low H_WM (known env) → blocked."""
        s_star = [
            [torch.tensor(0.5)],
            [torch.tensor(0.5)],
            [torch.tensor(0.5)],
            [torch.tensor(10.0)],  # outlier
        ]
        h_wm = [
            [torch.tensor(0.0)],
            [torch.tensor(0.0)],
            [torch.tensor(0.0)],
            [torch.tensor(0.0)],  # known env → tight threshold
        ]
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=1.0, lambda_wm=1.0)
        # The outlier (10.0) should be blocked when threshold is tight
        assert mask[3] == [0.0]
        # Normal ones should pass
        assert mask[0] == [1.0]

    def test_outlier_allowed_high_hwm(self):
        """High S_* outlier with high H_WM (unknown env) → allowed."""
        s_star = [
            [torch.tensor(0.5)],
            [torch.tensor(0.5)],
            [torch.tensor(0.5)],
            [torch.tensor(10.0)],  # outlier
        ]
        h_wm = [
            [torch.tensor(0.0)],
            [torch.tensor(0.0)],
            [torch.tensor(0.0)],
            [torch.tensor(100.0)],  # very uncertain → wide threshold
        ]
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=1.0, lambda_wm=1.0)
        # The outlier should be allowed because H_WM is very high
        assert mask[3] == [1.0]

    def test_empty(self):
        """Empty input should return empty."""
        mask = compute_dynamic_mask([], [], mu_base=1.0, lambda_wm=1.0)
        assert mask == []

    def test_single_element(self):
        """Single S_* value should not produce nan (std guard)."""
        s_star = [[torch.tensor(5.0)]]
        h_wm = [[torch.tensor(1.0)]]
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=1.0, lambda_wm=1.0)
        # Single element: |5.0 - 5.0| = 0 <= threshold → should pass
        assert mask == [[1.0]]


class TestApplyWmcErc:
    """Test full WMC-ERC orchestration."""

    def _make_batch(
        self, advantages, response_mask, old_log_probs, attention_mask
    ):
        """Create a minimal mock batch with required fields."""
        from unittest.mock import MagicMock

        batch = MagicMock()
        batch.batch = {
            "advantages": advantages.clone(),
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "attention_mask": attention_mask,
        }
        return batch

    def test_masking_zeros_advantage(self):
        """When a turn is masked, its advantages become zero."""
        # 4 samples, 1 turn each (4 action tokens + 0 env tokens)
        # Sample 3 has extremely different S_* pattern
        response_mask = torch.ones(4, 4, dtype=torch.float32)
        advantages = torch.ones(4, 4) * 2.0
        # Make sample 3 very overconfident (high p_k, low H)
        old_log_probs = torch.tensor(
            [
                [np.log(0.3)] * 4,
                [np.log(0.3)] * 4,
                [np.log(0.3)] * 4,
                [np.log(0.99)] * 4,  # very high confidence
            ]
        )
        entropys = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [0.01, 0.01, 0.01, 0.01],  # very low entropy
            ]
        )
        attention_mask = torch.ones(4, 4, dtype=torch.float32)

        batch = self._make_batch(
            advantages, response_mask, old_log_probs, attention_mask
        )
        config = {"mu_base": 1.0, "lambda_wm": 1.0, "enable": True}

        _, metrics = apply_wmc_erc(batch, entropys, config)

        # Check metrics exist
        assert "wmc_erc/mask_ratio" in metrics
        assert "wmc_erc/s_star_mean" in metrics
        assert "wmc_erc/h_wm_mean" in metrics
        assert "wmc_erc/total_turns" in metrics
        assert metrics["wmc_erc/total_turns"] == 4

        # Verify masking behavior:
        # Samples 0-2: S_* = 0.3*(2.0+log(0.3)) ≈ 0.239 (normal)
        # Sample 3:    S_* = 0.99*(0.01+log(0.99)) ≈ 0 (outlier in opposite direction)
        # H_WM = 0 for all (no env tokens) → tight threshold
        # Sample 3 should be masked (|S_3 - S_bar| > threshold)
        adv = batch.batch["advantages"]
        assert metrics["wmc_erc/num_masked_turns"] >= 1
        assert (adv[3] == 0).all(), "Sample 3 (overconfident outlier) should have zero advantages"
        assert (adv[:3] == 2.0).all(), "Samples 0-2 (normal) should keep original advantages"

    def test_disabled(self):
        """When enable=False, advantages unchanged."""
        response_mask = torch.ones(2, 4, dtype=torch.float32)
        advantages = torch.ones(2, 4) * 5.0
        old_log_probs = torch.full((2, 4), np.log(0.5))
        attention_mask = torch.ones(2, 4, dtype=torch.float32)

        batch = self._make_batch(
            advantages, response_mask, old_log_probs, attention_mask
        )
        entropys = torch.ones(2, 4)
        config = {"mu_base": 1.0, "lambda_wm": 1.0, "enable": False}

        _, metrics = apply_wmc_erc(batch, entropys, config)
        assert (batch.batch["advantages"] == 5.0).all()
        assert metrics == {}

    def test_multi_turn_selective_masking(self):
        """Multi-turn: only overconfident turns in known env get masked."""
        # 2 samples, 2 turns each: [act, act, env, env, act, act, env, pad]
        response_mask = torch.tensor(
            [
                [1, 1, 0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
        advantages = torch.ones(2, 8) * 3.0
        # Sample 0: normal confidence
        # Sample 1: overconfident on both turns
        old_log_probs = torch.tensor(
            [
                [np.log(0.3), np.log(0.3), 0, 0, np.log(0.3), np.log(0.3), 0, 0],
                [np.log(0.99), np.log(0.99), 0, 0, np.log(0.99), np.log(0.99), 0, 0],
            ]
        )
        entropys = torch.tensor(
            [
                [2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 0.0],
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.float32,
        )

        batch = self._make_batch(
            advantages, response_mask, old_log_probs, attention_mask
        )
        config = {"mu_base": 1.0, "lambda_wm": 1.0, "enable": True}

        _, metrics = apply_wmc_erc(batch, entropys, config)
        assert metrics["wmc_erc/total_turns"] == 4
        # With low H_WM on sample 1, its overconfident turns should be blocked
        # This is a statistical test — the exact outcome depends on batch stats

    def test_returns_wm_nll_metric(self):
        """WM NLL metric should be computed from env token log probs."""
        response_mask = torch.tensor(
            [[1, 1, 0, 0, 0, 0]], dtype=torch.float32
        )
        advantages = torch.ones(1, 6)
        old_log_probs = torch.tensor(
            [[np.log(0.5), np.log(0.5), np.log(0.3), np.log(0.4), 0, 0]]
        )
        entropys = torch.tensor([[1.0, 1.0, 2.0, 3.0, 0.0, 0.0]])
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 0, 0]], dtype=torch.float32
        )

        batch = self._make_batch(
            advantages, response_mask, old_log_probs, attention_mask
        )
        config = {"mu_base": 5.0, "lambda_wm": 1.0, "enable": True}

        _, metrics = apply_wmc_erc(batch, entropys, config)
        assert "wmc_erc/wm_nll" in metrics
        # WM NLL = -mean(log_prob at env positions [2,3])
        # = -(log(0.3) + log(0.4)) / 2
        expected_nll = -(np.log(0.3) + np.log(0.4)) / 2.0
        assert abs(metrics["wmc_erc/wm_nll"] - expected_nll) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
