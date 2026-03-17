"""OpenTinker Environment Module.

This module provides the environment framework for LLM training, including:
- BaseEnvironment: Abstract base class for all environments
- GameEnvironment: For multi-turn game environments (Gomoku, etc.)
- StaticDataEnvironment: For single-turn static datasets (Math, etc.)
- Data generators and utilities
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

# Lazy import for InferencePipeline to avoid heavy dependencies (like vllm) 
# when only the game server is needed.
def __getattr__(name):
    if name in [
        "InferencePipeline",
        "InferenceResult",
        "RemoteEnvironmentClient",
        "run_inference",
        "load_samples",
        "generate_samples",
    ]:
        from opentinker.environment.inference_pipeline import (
            InferencePipeline,
            InferenceResult,
            RemoteEnvironmentClient,
            run_inference,
            load_samples,
            generate_samples,
        )
        globals()["InferencePipeline"] = InferencePipeline
        globals()["InferenceResult"] = InferenceResult
        globals()["RemoteEnvironmentClient"] = RemoteEnvironmentClient
        globals()["run_inference"] = run_inference
        globals()["load_samples"] = load_samples
        globals()["generate_samples"] = generate_samples
        return globals()[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")

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
]
