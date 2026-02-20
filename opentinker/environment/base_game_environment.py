#!/usr/bin/env python3
"""Generic Game Environment for LLM Training.

This module provides a universal GameEnvironment class that works with
any AbstractGame implementation. Users don't need to create custom
environment wrapper classes anymore.

Usage:
    from base_game_environment import GameEnvironment
    from my_game import MyGame

    env = GameEnvironment(
        game_class=MyGame,
        config=config,
    )
    train_loader, val_loader = env.get_dataloader()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type
import os
import tempfile

import yaml
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from opentinker.environment.environment import BaseEnvironment, RewardFunctionSpec
from opentinker.environment.base_data_generator import DynamicGameDataset, collate_fn
from opentinker.environment.base_game import AbstractGame, GameDataGenerator


@dataclass
class InteractionSpec:
    """Specification for interaction configuration."""

    name: str
    class_path: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_config_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class_name": self.class_path,
            "config": self.config,
        }


class GameEnvironment(BaseEnvironment):
    """Universal environment for any AbstractGame implementation.

    This eliminates the need for users to create per-game environment wrappers.
    Just pass your game class and it handles everything automatically.

    Usage:
        from base_game_environment import GameEnvironment
        from my_game import MyGame

        config = OmegaConf.create({
            "tokenizer_path": "meta-llama/Llama-2-7b-chat-hf",
            "max_prompt_tokens": 1024,
            "batch_size": 8,
            "interaction": {
                "config": {
                    "env_endpoint": "http://localhost:8081",
                    "max_steps": 100,
                }
            }
        })

        env = GameEnvironment(
            game_class=MyGame,
            config=config,
            game_kwargs={"board_size": 15},  # Passed to game constructor
        )
    """

    def __init__(
        self,
        game_class: Type[AbstractGame],
        config: Dict[str, Any],
        game_kwargs: Optional[Dict[str, Any]] = None,
        reward_function: Optional[RewardFunctionSpec] = None,
        job_id: Optional[str] = None,
    ):
        """Initialize GameEnvironment.

        Args:
            game_class: The AbstractGame subclass to use
            config: Environment configuration (OmegaConf)
            game_kwargs: Arguments passed to game constructor
            reward_function: Optional external reward function
            job_id: Job ID for statistics isolation. If provided, will be
                   automatically injected into interaction config.
        """
        self.game_class = game_class
        self.config = config
        self.game_kwargs = game_kwargs or {}
        self.reward_function = reward_function

        # Get a game instance to extract metadata
        self._game_instance = game_class(**self.game_kwargs)
        self.interaction_name = self._game_instance.get_interaction_name()

        # Extract config (support both flat and nested config)
        if hasattr(config, "interaction") and hasattr(config.interaction, "config"):
            interaction_config = config.interaction.config
            self.env_endpoint = interaction_config.get(
                "env_endpoint", "http://localhost:8081"
            )
            self.max_steps = interaction_config.get("max_steps", 100)
            self.observation_template = interaction_config.get(
                "observation_template", "{observation}"
            )
            # Use provided job_id or fallback to config or default
            self.job_id = job_id or interaction_config.get("job_id", "default")

            # IMPORTANT: Preserve the full interaction.config dict (e.g., env_shards, env_endpoints, etc.).
            # Previous code only forwarded a subset, which broke sharded env routing.
            self._interaction_config_dict = (
                OmegaConf.to_container(interaction_config, resolve=True) or {}
            )
            if not isinstance(self._interaction_config_dict, dict):
                self._interaction_config_dict = {}
            # Ensure job_id is injected/overridden for stats isolation.
            self._interaction_config_dict["job_id"] = self.job_id
            # Ensure env_endpoint is present (after possible templating).
            self._interaction_config_dict.setdefault("env_endpoint", self.env_endpoint)
        else:
            self.env_endpoint = config.get("env_endpoint", "http://localhost:8081")
            self.max_steps = config.get("max_steps", 100)
            self.observation_template = config.get(
                "observation_template", "{observation}"
            )
            self.job_id = job_id or config.get("job_id", "default")
            self._interaction_config_dict = {
                "env_endpoint": self.env_endpoint,
                "max_steps": self.max_steps,
                "observation_template": self.observation_template,
                "job_id": self.job_id,
            }

        # Create interaction spec
        self.interaction_specs = [
            InteractionSpec(
                name=self.interaction_name,
                class_path="opentinker.environment.gym_environment_interaction.GymEnvironmentInteraction",
                config=self._interaction_config_dict,
            )
        ]

        self.train_dataloader = None
        self.val_dataloader = None
        self._interaction_config_path = None

        self._setup_dataloader()
        self._setup_interaction_config()

    def _setup_dataloader(self):
        """Setup training and validation dataloaders."""
        print(f"Loading tokenizer from {self.config.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Dataset config
        dataset_config = OmegaConf.create(
            {
                "max_prompt_length": self.config.max_prompt_tokens,
                "truncation": "right",
                "return_raw_chat": True,
            }
        )

        # Calculate virtual size
        num_steps = getattr(self.config, "num_steps", None)
        virtual_size = (
            num_steps * self.config.batch_size
            if num_steps
            else self.config.batch_size * 100
        )

        # Create data generator using the game class
        train_generator = GameDataGenerator(
            game_class=self.game_class,
            game_kwargs=self.game_kwargs,
        )

        print(f"Creating training dataset (virtual_size={virtual_size})")
        train_dataset = DynamicGameDataset(
            data_generator=train_generator,
            tokenizer=tokenizer,
            config=dataset_config,
            virtual_size=virtual_size,
        )

        # Create training dataloader
        num_workers = getattr(self.config, "num_workers", 0)
        print(f"Creating training DataLoader (batch_size={self.config.batch_size})")
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
        print(f"Training dataloader: {len(self.train_dataloader)} batches")

        # Create validation dataset with fixed seed
        val_samples = getattr(self.config, "val_batch_size", 50)
        val_generator = GameDataGenerator(
            game_class=self.game_class,
            game_kwargs=self.game_kwargs,
            seed=42,
        )

        val_dataset = DynamicGameDataset(
            data_generator=val_generator,
            tokenizer=tokenizer,
            config=dataset_config,
            virtual_size=val_samples,
            seed=42,
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=getattr(self.config, "val_batch_size", val_samples),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        print(f"Validation dataloader: {len(self.val_dataloader)} batches")

    def _setup_interaction_config(self):
        """Generate interaction config YAML file and store content for cross-node transmission."""
        if not self.interaction_specs:
            return

        interaction_list = [spec.to_config_dict() for spec in self.interaction_specs]
        config_dict = {"interaction": interaction_list}

        # Store the config content as YAML string for transmission to remote nodes
        self._interaction_config_content = yaml.dump(
            config_dict, default_flow_style=False
        )

        # Also create local temp file for backward compatibility
        fd, path = tempfile.mkstemp(
            suffix=".yaml", prefix=f"{self.interaction_name}_interaction_config_"
        )
        with os.fdopen(fd, "w") as f:
            f.write(self._interaction_config_content)

        self._interaction_config_path = path
        print(f"Generated interaction config: {path}")

    def get_dataloader(self):
        """Return training and validation dataloaders."""
        return self.train_dataloader, self.val_dataloader

    def get_interaction_config_path(self) -> Optional[str]:
        """Return path to interaction config file."""
        return self._interaction_config_path

    def get_interaction_config_content(self) -> Optional[str]:
        """Return interaction config content as YAML string for cross-node transmission."""
        return getattr(self, "_interaction_config_content", None)

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dictionary for server.

        Includes both path (for local use) and content (for cross-node distribution).
        """
        config = {}

        if self._interaction_config_path:
            default_agent_loop = getattr(
                self.game_class, "agent_loop_name", "generic_agent"
            )
            config["actor_rollout_ref"] = {
                "rollout": {
                    "multi_turn": {
                        "interaction_config_path": self._interaction_config_path,
                        # Include content for cross-node transmission
                        "interaction_config_content": self._interaction_config_content,
                    },
                    "agent": {"default_agent_loop": default_agent_loop},
                },
            }

        if self.reward_function:
            config["custom_reward_function"] = self.reward_function.to_config_dict()

        return config

    def setup(self, client):
        """Setup environment on the server."""
        if self.reward_function and self.reward_function.type == "code":
            print(
                f"Uploading reward function: {self.reward_function.code_function.__name__}"
            )
            client.upload_reward_function(
                function_name=self.reward_function.code_function.__name__,
                source_code=self.reward_function.code_source,
            )

        config = self.get_config()
        print(f"Environment config: {config}")
        return config

    def cleanup(self):
        """Clean up temporary files."""
        if self._interaction_config_path and os.path.exists(
            self._interaction_config_path
        ):
            os.remove(self._interaction_config_path)
            print(f"Removed: {self._interaction_config_path}")


def create_game_environment(
    game_class: Type[AbstractGame], config: Dict[str, Any], **game_kwargs
) -> GameEnvironment:
    """Convenience function to create a GameEnvironment.

    Args:
        game_class: The AbstractGame subclass
        config: Environment configuration
        **game_kwargs: Passed to game constructor

    Returns:
        GameEnvironment instance
    """
    return GameEnvironment(
        game_class=game_class,
        config=config,
        game_kwargs=game_kwargs,
    )
