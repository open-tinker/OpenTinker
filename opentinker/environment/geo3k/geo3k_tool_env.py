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
"""Geo3K Multi-Turn VL Environment.

This environment supports multi-turn geometry problem solving with vision-language models.
It uses Geo3KToolGame for verification-based interactions.
"""

from opentinker.environment.vl_game_environment import VLGameEnvironment
from opentinker.environment.geo3k.geo3k_tool_game import Geo3KToolGame


class Geo3KToolEnvironment(VLGameEnvironment):
    """Multi-turn VL environment for Geo3K geometry problems.

    This environment uses:
    - Geo3KToolGame for multi-turn verification logic
    - StaticDatasetGeneratorVL for image handling
    - GymEnvironmentInteraction for HTTP communication

    The model can submit answers multiple times and receive feedback
    in verl-compatible format: "Current parsed answer={answer} reward={0.0|1.0}"

    Args:
        config: Configuration object
        data_paths: Training data paths (parquet files)
        val_data_paths: Validation data paths (optional)
        job_id: Job identifier
        max_retries: Max verification attempts per problem (default: 3)

    Example:
        env = Geo3KToolEnvironment(
            config=config,
            data_paths=["~/data/geo3k_multiturn/train.parquet"],
            val_data_paths=["~/data/geo3k_multiturn/test.parquet"],
            job_id="geo3k_tool_training_001",
        )
    """

    def __init__(
        self,
        config,
        data_paths,
        val_data_paths=None,
        job_id=None,
        max_retries: int = 3,
    ):
        # Initialize with multi-turn Geo3K game and VL environment
        super().__init__(
            game_class=Geo3KToolGame,
            config=config,
            data_paths=data_paths,
            val_data_paths=val_data_paths,
            game_kwargs={"max_retries": max_retries},
            job_id=job_id,
            image_key="images",  # Geo3K uses "images" field
        )
