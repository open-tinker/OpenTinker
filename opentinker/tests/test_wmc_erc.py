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
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=0.1, mu_exp=10.0, eta_wm=1.0, lambda_wm=0.0,
                                   s_bar=s_bar, sigma=sigma)
        self.assertEqual(mask[0], [0.0]) # High blocked
        self.assertEqual(mask[1], [1.0]) # Low allowed
        
        # 2. mu_base=10.0 (allow high), mu_exp=0.1 (block low)
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=10.0, mu_exp=0.1, eta_wm=1.0, lambda_wm=0.0,
                                   s_bar=s_bar, sigma=sigma)
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

    def test_clipping_method_clip(self):
        """Verify that 'clip' method scales advantages instead of zeroing them."""
        # Setup data such that we have a violation
        # S* = 15.0, s_bar = 10.0, sigma = 1.0, mu_base = 1.0, h_factor = 1.0
        # diff = 5.0, threshold = 1.0
        # m_t should be 1.0 / 5.0 = 0.2
        
        s_star = [[torch.tensor(15.0)]]
        h_wm = [[torch.tensor(0.0)]] # lambda_wm=0, eta_wm=1 -> h_factor=1
        s_bar = 10.0
        sigma = 1.0
        
        # 1. Test clip
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=1.0, mu_exp=1.0, eta_wm=1.0, lambda_wm=0.0,
                                   s_bar=s_bar, sigma=sigma, clipping_method="clip")
        self.assertAlmostEqual(mask[0][0], 0.2, places=5)
        
        # 2. Test mask (for comparison)
        mask = compute_dynamic_mask(s_star, h_wm, mu_base=1.0, mu_exp=1.0, eta_wm=1.0, lambda_wm=0.0,
                                   s_bar=s_bar, sigma=sigma, clipping_method="mask")
        self.assertEqual(mask[0][0], 0.0)

    def test_clip_positive_only(self):
        """Verify that clip_positive_only only affects positive advantages."""
        # Two turns to create variance so s_t != s_bar
        response_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
        attention_mask = torch.ones(1, 4, dtype=torch.float32)
        # Turn 1 adv = 10, Turn 2 adv = -10
        advantages = torch.tensor([[10.0, 0.0, -10.0, 0.0]])
        # Make turn 1 very confident (p=0.9), turn 2 very uncertain (p=0.1)
        old_log_probs = torch.tensor([[np.log(0.9), 0.0, np.log(0.1), 0.0]])
        entropys = torch.tensor([[0.1, 0.0, 2.0, 0.0]])
        
        batch = self._make_batch(advantages, response_mask, old_log_probs, attention_mask)
        
        # Force a mask (m_t = 0.0) by setting very tight mu
        config = {
            "mu_base": 0.0001, 
            "mu_exp": 0.0001, 
            "lambda_wm": 0.0, 
            "enable": True, 
            "clipping_type": "batch",
            "clip_positive_only": True
        }
        running_stats = {"initialized": False}
        
        _, metrics = apply_wmc_erc(batch, entropys, config, running_stats)
        
        # If masked, turn 1 (positive adv) should be 0.0, turn 2 (negative adv) should remain -10.0
        self.assertEqual(batch.batch["advantages"][0, 0].item(), 0.0)
        self.assertEqual(batch.batch["advantages"][0, 2].item(), -10.0)

    def test_inverse_sft_mask(self):
        """Verify that inverse_sft_mask correctly computes sft_weights on env tokens."""
        # Seq: [Action1, Env1, Action2, Env2] -> masks: response_mask=[1, 0, 1, 0], attention=[1, 1, 1, 1]
        response_mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
        attention_mask = torch.ones(1, 4, dtype=torch.float32)
        advantages = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
        old_log_probs = torch.tensor([[np.log(0.5)] * 4])
        entropys = torch.tensor([[1.0] * 4])
        
        batch = self._make_batch(advantages, response_mask, old_log_probs, attention_mask)
        
        # Force m_t = 0.0 for turn 0, m_t = 1.0 for turn 1
        # Turn 0 S* = 0.5 * (1.0 + log(0.5)) ~ 0.15
        # We can just let compute_dynamic_mask do its thing.
        # Actually, let's just test that the weights are assigned correctly based on whatever mask is generated.
        config = {
            "mu_base": 0.001, # Force masking
            "mu_exp": 0.001,
            "lambda_wm": 0.0,
            "enable": True,
            "clipping_type": "batch",
            "inverse_sft_mask": True
        }
        running_stats = {"initialized": False}
        
        _, metrics = apply_wmc_erc(batch, entropys, config, running_stats)
        
        self.assertIn("sft_weights", batch.batch)
        sft_weights = batch.batch["sft_weights"]
        
        # Check shapes
        self.assertEqual(sft_weights.shape, advantages.shape)
        
        # m_t is applied to advantages. Action1 (idx 0), Action2 (idx 2)
        # Env1 (idx 1), Env2 (idx 3)
        # If Action1 was masked (m_t=0), Env1 should have weight 1.0
        # If Action1 was not masked (m_t=1), Env1 should have weight 0.0
        
        adv_action1 = batch.batch["advantages"][0, 0].item()
        m_t_1 = adv_action1 / 2.0 # original advantage was 2.0
        weight_env1 = sft_weights[0, 1].item()
        self.assertAlmostEqual(weight_env1, 1.0 - m_t_1)
        
        adv_action2 = batch.batch["advantages"][0, 2].item()
        m_t_2 = adv_action2 / 2.0
        weight_env2 = sft_weights[0, 3].item()
        self.assertAlmostEqual(weight_env2, 1.0 - m_t_2)
        
        # Ensure action tokens have 0 sft weight
        self.assertEqual(sft_weights[0, 0].item(), 0.0)
        self.assertEqual(sft_weights[0, 2].item(), 0.0)

if __name__ == "__main__":
    unittest.main()
