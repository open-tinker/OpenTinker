#!/usr/bin/env python3
"""Vision-Language Game Environment for OpenTinker.

This module provides a base GameEnvironment class for vision-language models
that uses AutoProcessor instead of AutoTokenizer.
"""

import logging
from typing import Optional, List

from omegaconf import OmegaConf
from transformers import AutoProcessor
from torchdata.stateful_dataloader import StatefulDataLoader

from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.base_data_generator_vl import (
    DynamicGameDatasetVL,
    collate_fn_vl,
)
from opentinker.environment.static_data_generator_vl import StaticDatasetGeneratorVL
from verl.trainer.main_ppo import create_rl_sampler

logger = logging.getLogger(__name__)


class VLGameEnvironment(GameEnvironment):
    """Base GameEnvironment for vision-language models.

    This class extends GameEnvironment to support vision-language models
    by using AutoProcessor for multimodal data processing.

    Key differences from GameEnvironment:
        - Uses AutoProcessor instead of AutoTokenizer
        - Uses DynamicGameDatasetVL for multimodal data
        - Uses StaticDatasetGeneratorVL for image loading

    Args:
        game_class: Game class to instantiate
        config: Configuration object
        data_paths: Training data paths
        val_data_paths: Validation data paths (optional)
        game_kwargs: Additional kwargs for game initialization
        job_id: Job identifier
        image_key: Key for image field in data (default: "images")
    """

    def __init__(
        self,
        game_class,
        config,
        data_paths,
        val_data_paths: Optional[List[str]] = None,
        game_kwargs: Optional[dict] = None,
        job_id: Optional[str] = None,
        image_key: str = "images",
    ):
        # Store VL-specific parameters
        self.data_paths = (
            [data_paths] if isinstance(data_paths, str) else list(data_paths)
        )
        self.val_data_paths = (
            [val_data_paths]
            if isinstance(val_data_paths, str)
            else (list(val_data_paths) if val_data_paths else None)
        )
        self.image_key = image_key

        # Initialize parent
        super().__init__(
            game_class=game_class,
            config=config,
            game_kwargs=game_kwargs or {},
            job_id=job_id,
        )

    def _setup_dataloader(self):
        """Setup dataloaders with vision-language support."""
        # Use AutoProcessor for VL models
        processor_path = getattr(
            self.config, "processor_path", self.config.tokenizer_path
        )
        processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=True
        )

        # Ensure padding side is left for RL
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token

        logger.info(f"Loaded AutoProcessor from {processor_path}")

        # Dataset configuration
        dataset_config = OmegaConf.create(
            {
                "max_prompt_length": self.config.max_prompt_tokens,
                "truncation": "right",
                "return_raw_chat": True,
            }
        )

        # Create game instance for system prompt
        game_instance = self.game_class()

        # Training data generator
        train_generator = StaticDatasetGeneratorVL(
            data_paths=self.data_paths,
            interaction_name=self.interaction_name,
            prompt_key="prompt",
            ground_truth_key="ground_truth",
            image_key=self.image_key,
            shuffle=True,
            system_prompt=game_instance.get_system_prompt(),
        )

        # Pre-filter overlong samples to prevent "max_tokens must be at least 1" vLLM error
        # This is similar to verl's RLHFDataset.maybe_filter_out_long_prompts()
        max_prompt_tokens = self.config.max_prompt_tokens
        filter_stats = train_generator.filter_overlong_samples(
            processor=processor,
            max_prompt_length=max_prompt_tokens,
        )
        print(
            f"[VLGameEnvironment] Training data: {filter_stats['filtered_count']}/{filter_stats['original_count']} samples after filtering"
        )

        # Calculate virtual size for training
        batch_size = self.config.batch_size
        num_steps = getattr(self.config, "num_steps", None)
        virtual_size = (
            num_steps * batch_size
            if num_steps
            else len(train_generator) * getattr(self.config, "num_epochs", 1)
        )

        # Create VL dataset
        train_dataset = DynamicGameDatasetVL(
            data_generator=train_generator,
            tokenizer=None,  # Will use processor's tokenizer
            processor=processor,
            config=dataset_config,
            virtual_size=virtual_size,
        )

        # Create sampler
        sampler_config = OmegaConf.create(
            {
                "shuffle": True,
                "seed": 42,
                "sampler": None,
            }
        )
        train_sampler = create_rl_sampler(sampler_config, train_dataset)

        # Create training dataloader
        self.train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=getattr(self.config, "num_workers", 0),
            collate_fn=collate_fn_vl,
            drop_last=True,
        )
        logger.info(f"Training dataloader: {len(self.train_dataloader)} batches")

        # Validation data generator
        if self.val_data_paths:
            val_generator = StaticDatasetGeneratorVL(
                data_paths=self.val_data_paths,
                interaction_name=self.interaction_name,
                prompt_key="prompt",
                ground_truth_key="ground_truth",
                image_key=self.image_key,
                shuffle=False,
                seed=42,
                system_prompt=game_instance.get_system_prompt(),
            )

            # Pre-filter validation data too
            val_filter_stats = val_generator.filter_overlong_samples(
                processor=processor,
                max_prompt_length=max_prompt_tokens,
            )
            print(
                f"[VLGameEnvironment] Validation data: {val_filter_stats['filtered_count']}/{val_filter_stats['original_count']} samples after filtering"
            )

            val_batch_size = getattr(
                self.config, "val_batch_size", min(64, len(val_generator))
            )

            # Create VL validation dataset
            val_dataset = DynamicGameDatasetVL(
                data_generator=val_generator,
                tokenizer=None,
                processor=processor,
                config=dataset_config,
                virtual_size=val_batch_size,
                seed=42,
            )

            self.val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=getattr(self.config, "num_workers", 0),
                collate_fn=collate_fn_vl,
                drop_last=False,
            )
            logger.info(
                f"Validation dataloader: {val_batch_size} fixed samples in {len(self.val_dataloader)} batch(es)"
            )
