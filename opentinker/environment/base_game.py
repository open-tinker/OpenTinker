#!/usr/bin/env python3
"""Base Game Abstraction for LLM Training Environments.

This module provides the abstract base class that users implement to define
their own game/environment logic. The framework handles HTTP serving,
data generation, and training integration automatically.

SIMPLIFIED DESIGN: Users only need to implement ONE class that handles both
game logic AND data generation. The framework provides everything else.

Example:
    class MyGame(AbstractGame):
        def reset(self, **kwargs) -> str:
            return "Game started!"

        def step(self, action: str) -> StepResult:
            return StepResult("You did: " + action, reward=1.0, done=False)

        def get_system_prompt(self) -> str:
            return "You are playing a game..."

        def generate_initial_state(self) -> Dict[str, Any]:
            return {"level": random.randint(1, 10)}
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import random


@dataclass
class StepResult:
    """Standardized result from a game step.

    Attributes:
        observation: Text observation for the LLM
        reward: Numerical reward signal
        done: Whether the episode has ended
        info: Additional information dictionary
    """

    observation: str
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class AbstractGame(ABC):
    """Abstract base class for game/environment implementations.

    Users implement this interface to define their game logic AND data generation.
    The framework handles HTTP serving, training datasets, and environment wrappers.

    REQUIRED methods (game logic):
        - reset(**kwargs) -> str: Reset game, return initial observation
        - step(action) -> StepResult: Execute action, return result
        - get_system_prompt() -> str: Return system prompt for LLM
        - get_initial_user_message() -> str: Return first user message

    OPTIONAL methods (data generation - override for custom behavior):
        - generate_initial_state() -> Dict: Generate random initial state for training
        - render_state_for_prompt(**state) -> str: Render state as text for prompt

    Example:
        class GomokuGame(AbstractGame):
            def reset(self, initial_moves=None, **kwargs):
                self._apply_moves(initial_moves)
                return self._render_board()

            def generate_initial_state(self):
                return {"initial_moves": self._random_moves()}
    """

    # Agent loop name for server rollout (used when building get_config()).
    # Override to "android_agent" etc. for task-specific agent loops.
    agent_loop_name: str = "generic_agent"

    # =========================================================================
    # REQUIRED: Game Logic Methods
    # =========================================================================

    @abstractmethod
    def reset(self, **kwargs) -> str:
        """Reset the game to initial state.

        Args:
            **kwargs: Game-specific initialization parameters
                     (these come from generate_initial_state() during training)

        Returns:
            Initial observation string for the LLM
        """
        pass

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Execute an action and return the result.

        Args:
            action: Raw action string from the LLM

        Returns:
            StepResult with observation, reward, done flag, and info
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this game.

        Should explain the game rules and expected response format.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_initial_user_message(self) -> str:
        """Return the initial user message to start the game.

        Called after reset() to build the initial prompt.

        Returns:
            Initial user message string
        """
        pass

    # =========================================================================
    # OPTIONAL: Data Generation Methods (override for custom training data)
    # =========================================================================

    def generate_initial_state(self) -> Dict[str, Any]:
        """Generate random initial state for training data.

        Override this to customize how training samples are generated.
        The returned dict is passed to reset() as kwargs.

        Returns:
            Dict of kwargs for reset()

        Example:
            def generate_initial_state(self):
                return {"initial_moves": self._random_moves(), "difficulty": 3}
        """
        return {}  # Default: empty initial state

    def render_state_for_prompt(self, **state) -> str:
        """Render initial state as text for the training prompt.

        Override to customize how initial state appears in prompts.
        By default, calls reset() and uses its output.

        Args:
            **state: The state dict from generate_initial_state()

        Returns:
            Text representation for the prompt
        """
        return self.reset(**state)

    def get_user_message_with_state(self, **state) -> str:
        """Generate user message including the initial state.

        Override for custom formatting. Default combines initial_user_message
        with rendered state.

        Args:
            **state: The state dict from generate_initial_state()

        Returns:
            User message string
        """
        base_message = self.get_initial_user_message()
        state_text = self.render_state_for_prompt(**state)
        return f"{base_message}\n\n{state_text}"

    # =========================================================================
    # OPTIONAL: Utility Methods
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Return current game state for debugging/visualization.

        Returns:
            Dictionary with game state information
        """
        return {}

    def get_interaction_name(self) -> str:
        """Return the interaction name for routing.

        Returns:
            Interaction name (default: class name in lowercase)
        """
        return self.__class__.__name__.lower().replace("game", "")

    def parse_action(self, raw_action: str) -> Optional[Any]:
        """Parse raw LLM output to extract the action.

        Override to implement custom parsing logic.

        Args:
            raw_action: Raw string from LLM output

        Returns:
            Parsed action, or None if parsing fails
        """
        return raw_action


class GameDataGenerator:
    """Data generator that works with any AbstractGame.

    This class wraps an AbstractGame instance and generates training
    samples using the game's generate_initial_state() method.

    Users don't need to create this themselves - it's used internally
    by GameEnvironment.
    """

    def __init__(
        self,
        game_class: type,
        game_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize GameDataGenerator.

        Args:
            game_class: The AbstractGame subclass to use
            game_kwargs: Arguments passed to game constructor
            seed: Optional seed for reproducible generation
        """
        self.game_class = game_class
        self.game_kwargs = game_kwargs or {}
        self.seed = seed

        # Create a game instance for generating samples
        self._game = game_class(**self.game_kwargs)

    def generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a single training sample.

        Args:
            index: Sample index

        Returns:
            Dict with prompt, env_kwargs, data_source
        """
        if self.seed is not None:
            random.seed(self.seed + index)

        # Generate initial state using game's method
        initial_state = self._game.generate_initial_state()

        # Build messages
        messages = [
            {"role": "system", "content": self._game.get_system_prompt()},
            {
                "role": "user",
                "content": self._game.get_user_message_with_state(**initial_state),
            },
        ]

        return {
            "prompt": messages,
            "env_kwargs": initial_state,
            "data_source": self._game.get_interaction_name(),
        }

    def get_interaction_name(self) -> str:
        """Return interaction name from game."""
        return self._game.get_interaction_name()
