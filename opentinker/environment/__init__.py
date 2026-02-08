"""OpenTinker Environment Module.

This module provides the environment framework for LLM training, including:
- BaseEnvironment: Abstract base class for all environments
- GameEnvironment: For multi-turn game environments (Gomoku, etc.)
- StaticDataEnvironment: For single-turn static datasets (Math, etc.)
"""

# Base classes
from opentinker.environment.environment import BaseEnvironment, RewardFunctionSpec
from opentinker.environment.base_game import AbstractGame, StepResult, GameDataGenerator
from opentinker.environment.base_game_environment import (
    GameEnvironment,
    InteractionSpec,
)
from opentinker.environment.base_data_generator import (
    AbstractGameDataGenerator,
    DynamicGameDataset,
    collate_fn,
)

# Static data support
from opentinker.environment.static_data_generator import StaticDatasetGenerator
# from opentinker.environment.static_data_environment import StaticDataEnvironment

# Server utilities
from opentinker.environment.base_game_server import (
    BaseGameStats,
    GameStats,
    create_game_server,
    run_game_server,
)

from opentinker.environment.inference_pipeline import (
    InferencePipeline,
    InferenceResult,
    RemoteEnvironmentClient,
    run_inference,
    load_samples,
    generate_samples,
)

from opentinker.environment.swegym import SWEGymGame

__all__ = [
    # Base
    "BaseEnvironment",
    "RewardFunctionSpec",
    # Game
    "AbstractGame",
    "StepResult",
    "GameDataGenerator",
    "GameEnvironment",
    "InteractionSpec",
    # Data
    "AbstractGameDataGenerator",
    "DynamicGameDataset",
    "collate_fn",
    # Static
    "StaticDatasetGenerator",
    # Inference
    "InferencePipeline",
    "InferenceResult",
    "RemoteEnvironmentClient",
    "run_inference",
    "load_samples",
    "generate_samples",
    # "StaticDataEnvironment",
    # Server
    "BaseGameStats",
    "GameStats",
    "create_game_server",
    "run_game_server",
    # SWE-Gym
    "SWEGymGame",
]
