#!/usr/bin/env python3
"""Math Problem Game Implementation.

This module provides the MathGame class that implements AbstractGame interface
for single-turn math problem solving. Reward is computed directly in step().

This is a single-turn game:
- reset(): Loads the math problem with ground truth
- step(): Model provides answer, reward computed, game ends

Example:
    from math_game import MathGame

    game = MathGame()
    obs = game.reset(ground_truth="42")
    result = game.step("The answer is 42")
"""

import re
from typing import Any, Dict, Optional

from opentinker.environment.base_game import AbstractGame, StepResult


class MathGame(AbstractGame):
    """Single-turn math problem game with reward computed in step().

    This implementation:
    - Receives ground_truth in reset()
    - Model provides one answer in step()
    - Reward is computed based on answer correctness
    - Game always ends after one step (single-turn)

    Attributes:
        max_retries: Max allowed retry attempts for invalid format (default: 0 = no retries)
    """

    # Reward constants
    REWARD_CORRECT = 1.0
    REWARD_INCORRECT = 0.0
    REWARD_FORMAT_ERROR = 0.0  # Same as incorrect for single-turn

    def __init__(
        self,
        max_retries: int = 0,
        answer_pattern: Optional[str] = None,
    ):
        """Initialize MathGame.

        Args:
            max_retries: Max retry attempts for format errors (0 = single turn)
            answer_pattern: Regex pattern to extract answer (optional)
        """
        self.max_retries = max_retries
        self.answer_pattern = answer_pattern
        self._init_game_state()

    def _init_game_state(self):
        """Initialize/reset game state variables."""
        self.ground_truth = None
        self.data_source = "math"
        self.extra_info = {}
        self.attempt_count = 0
        self.game_over = False

    def reset(
        self,
        ground_truth: Optional[str] = None,
        data_source: str = "math",
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Reset the game with a new problem.

        Args:
            ground_truth: The correct answer (used for reward computation)
            data_source: Data source identifier for reward function
            extra_info: Additional info passed to reward function
            **kwargs: Ignored (for compatibility)

        Returns:
            Empty string (prompt already contains the problem)
        """
        self._init_game_state()
        self.ground_truth = ground_truth
        self.data_source = data_source
        self.extra_info = extra_info or {}

        # For single-turn, no additional observation needed
        # The problem is already in the prompt from data generator
        return ""

    def step(self, action: str) -> StepResult:
        """Process the model's answer and compute reward.

        This is single-turn: always returns done=True after computing reward.

        Args:
            action: The model's response containing the answer

        Returns:
            StepResult with reward based on answer correctness
        """
        if self.game_over:
            return StepResult(
                observation="Game already over.",
                reward=0.0,
                done=True,
                info={"error": "game_over"},
            )

        self.attempt_count += 1
        self.game_over = True  # Single-turn: always end after this

        # Compute reward using the reward scoring function
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
            observation="",  # No observation needed for single-turn
            reward=reward,
            done=True,
            info=info,
        )

    def _compute_reward(self, solution_str: str) -> float:
        """Compute reward by comparing solution to ground truth.

        Uses the default_compute_score from verl for consistency.
        """
        try:
            from verl.utils.reward_score import default_compute_score

            # breakpoint()
            score = default_compute_score(
                data_source=self.data_source,
                solution_str=solution_str,
                ground_truth=self.ground_truth,
                extra_info=self.extra_info,
            )

            # Handle both dict and scalar return values
            if isinstance(score, dict):
                scalar_score = float(score.get("score", 0.0))
            else:
                scalar_score = float(score)

            # Strict MATH scorer may return 0 if model did not output \boxed{}.
            # For paper benchmarks, allow a conservative fallback parser that checks
            # explicit final-answer patterns and last-line numeric/latex answers.
            if scalar_score <= 0.0:
                loose = self._loose_match(solution_str)
                if loose > 0.0:
                    return loose
            return scalar_score

        except Exception as e:
            # Fallback: simple string matching
            import logging

            logging.warning(f"Reward computation failed, using fallback: {e}")
            return self._simple_match(solution_str)

    def _extract_answer_candidate(self, solution_str: str) -> Optional[str]:
        """Extract likely final answer from model output for loose validation."""
        try:
            from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

            boxed = last_boxed_only_string(solution_str)
            if boxed is not None:
                return remove_boxed(boxed).strip()
        except Exception:
            pass

        answer_matches = re.findall(
            r"(?is)(?:final\s+answer|answer)\s*[:：]\s*([^\n\r]+)",
            solution_str,
        )
        if answer_matches:
            return answer_matches[-1].strip().strip("$").strip(" .。!！")

        lines = [ln.strip() for ln in solution_str.splitlines() if ln.strip()]
        if not lines:
            return None
        tail = lines[-1]
        tail = re.sub(r"^[-*`> ]+", "", tail)
        if "=" in tail and len(tail.split("=")[0].strip()) <= 10:
            tail = tail.split("=")[-1]
        tail = tail.strip().strip("$").strip(" .。!！")
        return tail or None

    def _loose_match(self, solution_str: str) -> float:
        """Loose equivalence check to mitigate answer-format mismatch."""
        if self.ground_truth is None:
            return 0.0
        candidate = self._extract_answer_candidate(solution_str)
        if not candidate:
            return 0.0
        try:
            from verl.utils.reward_score.math_reward import is_equiv

            return self.REWARD_CORRECT if is_equiv(candidate, str(self.ground_truth)) else self.REWARD_INCORRECT
        except Exception:
            return self.REWARD_CORRECT if str(self.ground_truth).strip() == candidate.strip() else self.REWARD_INCORRECT

    def _simple_match(self, solution_str: str) -> float:
        """Simple fallback: check if ground_truth appears in solution."""
        if self.ground_truth is None:
            return 0.0

        # Extract boxed answer if present
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
        """Return the system prompt for math problems."""
        return (
            "You are a helpful assistant that solves math problems step by step. "
            "Think through the problem carefully and provide your final answer. "
            "Put your final answer in \\boxed{} format."
        )

    def get_initial_user_message(self) -> str:
        """Return the initial user message.

        Note: For static dataset, the actual problem is in the prompt,
        so this is just a placeholder.
        """
        return "Please solve the following problem:"

    def get_state(self) -> Dict[str, Any]:
        """Return current game state."""
        return {
            "ground_truth": self.ground_truth,
            "data_source": self.data_source,
            "attempt_count": self.attempt_count,
            "game_over": self.game_over,
        }

    def get_interaction_name(self) -> str:
        """Return interaction name for math."""
        return "math"

    # =========================================================================
    # Data Generation Methods (for training with static dataset)
    # =========================================================================

    def generate_initial_state(self) -> Dict[str, Any]:
        """For static dataset, this is not used - data comes from generator."""
        return {}

    def get_user_message_with_state(self, **kwargs) -> str:
        """Generate user message - for static data, prompt comes from dataset."""
        return self.get_initial_user_message()
