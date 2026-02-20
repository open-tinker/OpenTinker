#!/usr/bin/env python3
"""Geo3K Game Environment using VL components."""

from opentinker.environment.vl_game_environment import VLGameEnvironment
from opentinker.environment.geo3k.geo3k_game import Geo3KGame


class Geo3KGameEnvironment(VLGameEnvironment):
    """GameEnvironment for Geo3K geometry problems with vision-language models.

    This environment uses:
    - VLGameEnvironment for multimodal data processing
    - StaticDatasetGeneratorVL for loading Geo3K parquet data with images
    - Geo3KGame for geometry problem logic

    Args:
        config: Configuration object
        data_paths: Training data paths (parquet files)
        val_data_paths: Validation data paths (optional)
        job_id: Job identifier

    Example:
        env = Geo3KGameEnvironment(
            config=config,
            data_paths=["~/data/geo3k/train.parquet"],
            val_data_paths=["~/data/geo3k/test.parquet"],
            job_id="geo3k_training_001",
        )
    """

    def __init__(self, config, data_paths, val_data_paths=None, job_id=None):
        # Initialize with Geo3K game and VL environment
        super().__init__(
            game_class=Geo3KGame,
            config=config,
            data_paths=data_paths,
            val_data_paths=val_data_paths,
            game_kwargs={},
            job_id=job_id,
            image_key="images",  # Geo3K uses "images" field
        )
