#!/usr/bin/env python3
"""Geo3K Geometry Problem Game Implementation.

This module provides the Geo3KGame class for geometry problem solving
with vision-language models. It extends the math game pattern for
geometry-specific problems.
"""

import re
from typing import Any, Dict, Optional

from opentinker.environment.base_game import AbstractGame, StepResult


class Geo3KGame(AbstractGame):
    """Single-turn geometry problem game with vision support.

    This game handles geometry problems from the Geo3K dataset.
    Similar to MathGame but optimized for geometry problem format.

    The game expects:
    - Images are provided via the VL processor
    - Ground truth answer in reset()
    - Model provides answer with reasoning in <think></think> tags
    - Final answer should be in \\boxed{} format
    """

    # Reward constants
    REWARD_CORRECT = 1.0
    REWARD_INCORRECT = 0.0

    def __init__(self, max_retries: int = 0):
        """Initialize Geo3KGame.

        Args:
            max_retries: Max retry attempts (0 = single turn)
        """
        self.max_retries = max_retries
        self._init_game_state()

    def _init_game_state(self):
        """Initialize/reset game state variables."""
        self.ground_truth = None
        self.data_source = "hiyouga/geometry3k"  # Must match verl's reward scorer
        self.extra_info = {}
        self.attempt_count = 0
        self.game_over = False

    def reset(
        self,
        ground_truth: Optional[str] = None,
        data_source: str = "hiyouga/geometry3k",
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Reset the game with a new geometry problem.

        Args:
            ground_truth: The correct answer
            data_source: Data source identifier
            extra_info: Additional info for reward computation
            **kwargs: Ignored (for compatibility)

        Returns:
            Empty string (problem is in prompt with images)
        """
        self._init_game_state()
        self.ground_truth = ground_truth
        self.data_source = data_source
        self.extra_info = extra_info or {}

        # Problem and images are in the prompt from VL dataset
        return ""

    def step(self, action: str) -> StepResult:
        """Process the model's answer and compute reward.

        Args:
            action: Model's response with reasoning and answer

        Returns:
            StepResult with reward based on correctness
        """
        if self.game_over:
            return StepResult(
                observation="Game already over.",
                reward=0.0,
                done=True,
                info={"error": "game_over"},
            )

        self.attempt_count += 1
        self.game_over = True  # Single-turn

        # Compute reward
        reward = self._compute_reward(action)

        # Build info dict
        info = {
            "ground_truth": self.ground_truth,
            "data_source": self.data_source,
            "attempt": self.attempt_count,
            "solution_str": action,
        }
        info.update(self.extra_info)

        return StepResult(
            observation="",
            reward=reward,
            done=True,
            info=info,
        )

    def _compute_reward(self, solution_str: str) -> float:
        """Compute reward using verl's reward scoring.

        Args:
            solution_str: Model's solution string

        Returns:
            Reward score (0.0 or 1.0)
        """
        try:
            from verl.utils.reward_score import default_compute_score

            score = default_compute_score(
                data_source=self.data_source,
                solution_str=solution_str,
                ground_truth=self.ground_truth,
                extra_info=self.extra_info,
            )

            # Handle dict or scalar return
            if isinstance(score, dict):
                return float(score.get("score", 0.0))
            return float(score)

        except Exception as e:
            # Fallback to simple matching
            import logging

            logging.warning(f"Reward computation failed, using fallback: {e}")
            return self._simple_match(solution_str)

    def _simple_match(self, solution_str: str) -> float:
        """Simple fallback: extract and compare boxed answer.

        Args:
            solution_str: Model's solution string

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        if self.ground_truth is None:
            return 0.0

        # Extract answer from \boxed{} if present
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution_str)
        if boxed_match:
            solution_str = boxed_match.group(1)

        # Normalize and compare
        solution_normalized = solution_str.strip().lower()
        gt_normalized = str(self.ground_truth).strip().lower()

        return (
            self.REWARD_CORRECT
            if gt_normalized in solution_normalized
            else self.REWARD_INCORRECT
        )

    def get_system_prompt(self) -> str:
        """Return system prompt for geometry problems.

        Note: Returns None because the geo3k parquet data already contains
        instruction in the user prompt (e.g., "You FIRST think about...").
        Adding a system prompt would duplicate the instruction.
        """
        return None

    def get_initial_user_message(self) -> str:
        """Return initial user message placeholder."""
        return "Please solve the following geometry problem:"

    def get_state(self) -> Dict[str, Any]:
        """Return current game state."""
        return {
            "ground_truth": self.ground_truth,
            "data_source": self.data_source,
            "attempt_count": self.attempt_count,
            "game_over": self.game_over,
        }

    def get_interaction_name(self) -> str:
        """Return interaction name for Geo3K."""
        return "geo3k"

    def generate_initial_state(self) -> Dict[str, Any]:
        """For static dataset, not used."""
        return {}

    def get_user_message_with_state(self, **kwargs) -> str:
        """Generate user message - for static data, comes from dataset."""
        return self.get_initial_user_message()
