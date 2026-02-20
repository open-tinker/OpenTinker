#!/usr/bin/env python3
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
"""Geo3K Multi-Turn Game with Answer Verification.

This module provides a multi-turn game for geometry problem solving.
The model can submit answers for verification and receive feedback
in verl-compatible format: "Current parsed answer={answer} reward={0.0|1.0}"

Features:
- Per-step penalty (-0.05) if answer doesn't improve
- Support for multiple verification attempts
- Final reward based on answer correctness
"""

from typing import Any, Dict, Optional

from opentinker.environment.base_game import AbstractGame, StepResult


class Geo3KToolGame(AbstractGame):
    """Multi-turn Geo3K game with answer verification.

    The model receives a geometry problem with image and can submit
    answers in \\boxed{} format for verification. The environment
    returns feedback with the parsed answer and reward value.

    **Feedback Format (matching verl's Geo3kTool):**
    - Returns: "Current parsed answer={answer} reward={0.0|1.0}"
    - Per-step reward: -0.05 penalty if answer doesn't improve

    Attributes:
        max_retries: Maximum number of verification attempts
        ground_truth: Correct answer for the problem
        best_reward: Tracks best reward achieved (for penalty logic)
    """

    # Reward constants
    REWARD_CORRECT = 1.0
    REWARD_INCORRECT = 0.0
    PENALTY_NO_IMPROVEMENT = -0.05

    def __init__(self, max_retries: int = 3):
        """Initialize Geo3KToolGame.

        Args:
            max_retries: Maximum number of verification attempts (default: 3)
        """
        self.max_retries = max_retries
        self._init_game_state()

    def _init_game_state(self):
        """Initialize/reset game state variables."""
        self.ground_truth = None
        self.data_source = "hiyouga/geometry3k"
        self.extra_info = {}
        self.attempt_count = 0
        self.game_over = False
        self.best_reward = 0.0  # Track best reward for penalty logic
        self.last_parsed_answer = None

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

        return ""

    def step(self, action: str) -> StepResult:
        """Process the model's answer and return verification feedback.

        Args:
            action: Model's response with reasoning and \\boxed{} answer

        Returns:
            StepResult with:
            - observation: "Current parsed answer={answer} reward={reward}"
            - reward: per-step penalty or final reward
            - done: True if correct or max attempts reached
            - info: Metadata including ground_truth and attempt count
        """
        if self.game_over:
            return StepResult(
                observation="Game already over.",
                reward=0.0,
                done=True,
                info={"error": "game_over"},
            )

        self.attempt_count += 1

        # Extract boxed answer
        parsed_answer = self._extract_boxed(action)
        self.last_parsed_answer = parsed_answer

        # Compute correctness reward (0.0 or 1.0)
        correctness_reward = self._compute_reward(action)

        # Per-step penalty logic (matching verl's Geo3kTool)
        if correctness_reward > self.best_reward:
            step_reward = 0.0  # No penalty if improved
        else:
            step_reward = self.PENALTY_NO_IMPROVEMENT  # -0.05 penalty

        self.best_reward = max(self.best_reward, correctness_reward)

        # Feedback matching verl format
        observation = (
            f"Current parsed answer={parsed_answer} reward={correctness_reward}"
        )

        # Check termination conditions
        is_correct = correctness_reward == self.REWARD_CORRECT
        max_attempts_reached = self.attempt_count >= self.max_retries
        done = is_correct or max_attempts_reached

        if done:
            self.game_over = True
            # Final reward is the correctness reward
            final_reward = correctness_reward
        else:
            final_reward = step_reward

        info = {
            "ground_truth": self.ground_truth,
            "data_source": self.data_source,
            "attempt_count": self.attempt_count,
            "parsed_answer": parsed_answer,
            "correctness_reward": correctness_reward,
            "is_correct": is_correct,
            "solution_str": action,
        }
        info.update(self.extra_info)

        return StepResult(
            observation=observation,
            reward=final_reward,
            done=done,
            info=info,
        )

    def _extract_boxed(self, text: str) -> str:
        """Extract answer from \\boxed{} format with nested brace support.

        Uses depth counting to correctly handle nested braces like \\frac{8}{3}.

        Args:
            text: Text containing \\boxed{answer}

        Returns:
            Extracted answer or "(no boxed answer)" if not found
        """
        # Find the last occurrence of \boxed{
        start_pos = text.rfind(r"\boxed{")
        if start_pos == -1:
            return "(no boxed answer)"

        # Extract content after \boxed{
        content = text[start_pos + len(r"\boxed{") :]

        # Use depth counting to find matching closing brace
        depth = 0
        end_pos = -1
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    end_pos = i
                    break
                depth -= 1

        if end_pos != -1:
            return content[:end_pos].strip()

        return "(no boxed answer)"

    def _compute_reward(self, solution_str: str) -> float:
        """Compute reward by comparing solution to ground truth.

        Uses verl's reward scoring for consistency.

        Args:
            solution_str: Model's solution string

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        try:
            from verl.utils.reward_score import geo3k

            score = geo3k.compute_score(
                solution_str,
                self.ground_truth,
                use_boxed=True,
                format_score=0.0,
            )
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

        parsed = self._extract_boxed(solution_str)
        if parsed == "(no boxed answer)":
            return 0.0

        # Normalize and compare
        parsed_normalized = parsed.strip().lower()
        gt_normalized = str(self.ground_truth).strip().lower()

        return (
            self.REWARD_CORRECT
            if parsed_normalized == gt_normalized
            else self.REWARD_INCORRECT
        )

    def get_system_prompt(self) -> str:
        """Return system prompt for multi-turn geometry problems."""
        return (
            "You are a geometry expert. You are given a geometry problem with an image. "
            "Solve it step by step. After reasoning, submit your answer in \\boxed{} format. "
            "If your answer is incorrect, you will receive feedback showing the parsed answer and reward. "
            "You can then refine your thinking and submit again."
        )

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
            "best_reward": self.best_reward,
            "last_parsed_answer": self.last_parsed_answer,
        }

    def get_interaction_name(self) -> str:
        """Return interaction name for Geo3K multi-turn."""
        return "geo3k_tool"

    def generate_initial_state(self) -> Dict[str, Any]:
        """For static dataset, not used."""
        return {}

    def get_user_message_with_state(self, **kwargs) -> str:
        """Generate user message - for static data, comes from dataset."""
        return self.get_initial_user_message()
