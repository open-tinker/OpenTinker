"""ALFWorld environment module for OpenTinker.

This module provides ALFWorld text-based environment integration
for LLM RL training.

Usage:
    from opentinker.environment.alfworld import ALFWorldGame

    game = ALFWorldGame()
    obs = game.reset()
    result = game.step("go to desk 1")
"""

from opentinker.environment.alfworld.alfworld_game import ALFWorldGame

__all__ = ["ALFWorldGame"]
