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
Tests for WMC-ERC dynamic entropy clipping module using unittest.

Run with: python opentinker/tests/test_wmc_erc.py
"""

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock

from opentinker.backend_patch.verl.trainer.ppo.per_step_core_algos import (
    compute_turn_boundaries,
)
from opentinker.backend_patch.verl.trainer.ppo.wmc_erc import (
    apply_wmc_erc,
    compute_dynamic_mask,
    compute_h_wm,
    compute_s_star,
)


class TestWmcErc(unittest.TestCase):
    """Test WMC-ERC components and orchestration."""

    def test_single_turn_s_star(self):
        """S_* for a single turn = mean of p_k * (H + log p_k) over action tokens."""
        p = torch.tensor([0.8, 0.6, 0.9, 0.7])
        old_log_probs = torch.zeros(1, 6)
        old_log_probs[0, :4] = torch.log(p)
        entropys = torch.tensor([[1.0, 1.5, 0.5, 1.2, 0.0, 0.0]])
        response_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.float32)
        boundaries = compute_turn_boundaries(response_mask)

        result = compute_s_star(old_log_probs, entropys, response_mask, boundaries)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)

        H = torch.tensor([1.0, 1.5, 0.5, 1.2])
        expected = (p * (H + torch.log(p))).mean().item()
        self.assertAlmostEqual(result[0][0].item(), expected, places=5)

    def test_asymmetric_behavior(self):
        """Test that mu_base and mu_exp act differently using compute_dynamic_mask."""
        s_star = [[torch.tensor(15.0)], [torch.tensor(5.0)]] # Mean=10
        h_wm = [[torch.tensor(1.0)]] * 2
        s_bar = 10.0
        sigma = 5.0
        h_bar = 1.0
        
        # 1. mu_base=0.1 (block high), mu_exp=10.0 (allow low)
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=0.1, mu_exp=10.0, lambda_wm=0.0,
                                   s_bar=s_bar, sigma=sigma, h_bar=h_bar)
        self.assertEqual(mask[0], [0.0]) # High blocked
        self.assertEqual(mask[1], [1.0]) # Low allowed
        
        # 2. mu_base=10.0 (allow high), mu_exp=0.1 (block low)
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=10.0, mu_exp=0.1, lambda_wm=0.0,
                                   s_bar=s_bar, sigma=sigma, h_bar=h_bar)
        self.assertEqual(mask[0], [1.0]) # High allowed
        self.assertEqual(mask[1], [0.0]) # Low blocked

    def _make_batch(self, advantages, response_mask, old_log_probs, attention_mask):
        batch = MagicMock()
        batch.batch = {
            "advantages": advantages.clone(),
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "attention_mask": attention_mask,
        }
        return batch

    def test_clipping_type_batch(self):
        """Verify that 'batch' mode uses current batch statistics."""
        response_mask = torch.ones(4, 4, dtype=torch.float32)
        advantages = torch.ones(4, 4) * 2.0
        # Sample 3 is an outlier in this batch
        old_log_probs = torch.tensor([[np.log(0.3)]*4, [np.log(0.3)]*4, [np.log(0.3)]*4, [np.log(0.9)]*4])
        entropys = torch.tensor([[2.0]*4, [2.0]*4, [2.0]*4, [3.0]*4])
        attention_mask = torch.ones(4, 4, dtype=torch.float32)
        batch = self._make_batch(advantages, response_mask, old_log_probs, attention_mask)
        
        config = {"mu_base": 0.1, "mu_exp": 10.0, "lambda_wm": 1.0, "enable": True, "clipping_type": "batch"}
        running_stats = {"s_bar": 100.0, "s_std": 1.0, "h_bar": 1.0, "initialized": True} # Very different global stats
        
        _, metrics = apply_wmc_erc(batch, entropys, config, running_stats)
        
        # In 'batch' mode, Sample 3 should be blocked based on BATCH mean, not GLOBAL mean
        # If it used global mean (100), sample 3 (S*~2.6) would be an exploration outlier and allowed by mu_exp=10
        # But in batch mode (S_bar~0.8), sample 3 is a collapsing outlier and blocked by mu_base=0.1
        self.assertTrue((batch.batch["advantages"][3] == 0).all())

    def test_clipping_type_global(self):
        """Verify that 'global' mode uses running statistics."""
        response_mask = torch.ones(4, 4, dtype=torch.float32)
        advantages = torch.ones(4, 4) * 2.0
        old_log_probs = torch.tensor([[np.log(0.3)]*4]*4)
        entropys = torch.tensor([[2.0]*4]*4)
        # S* for all samples will be ~0.24
        
        attention_mask = torch.ones(4, 4, dtype=torch.float32)
        batch = self._make_batch(advantages, response_mask, old_log_probs, attention_mask)
        
        # Global s_bar is far away (10.0), so batch S* (0.24) looks like a huge exploration outlier
        config = {"mu_base": 10.0, "mu_exp": 0.1, "lambda_wm": 0.0, "enable": True, "clipping_type": "global"}
        running_stats = {"s_bar": 10.0, "s_std": 1.0, "h_bar": 1.0, "initialized": True}
        
        _, metrics = apply_wmc_erc(batch, entropys, config, running_stats)
        
        # Should be blocked because mu_exp=0.1 is very tight relative to global s_bar
        self.assertTrue((batch.batch["advantages"] == 0).all())


if __name__ == "__main__":
    unittest.main()
