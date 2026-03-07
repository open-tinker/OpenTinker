"""ScienceWorld environment module for OpenTinker.

This module provides ScienceWorld text-based environment integration
for LLM RL training.

Usage:
    from opentinker.environment.sciworld import SciWorldGame

    game = SciWorldGame()
    obs = game.reset()
    result = game.step("pick up thermometer")
"""

from opentinker.environment.sciworld.sciworld_game import SciWorldGame

__all__ = ["SciWorldGame"]
