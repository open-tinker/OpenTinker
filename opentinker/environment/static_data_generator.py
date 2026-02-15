#!/usr/bin/env python3
"""Static Dataset Generator for LLM Training.

This module provides a data generator that wraps static datasets (parquet/jsonl)
with producer-consumer sampling, compatible with the DynamicGameDataset interface.

This enables using static math/reasoning datasets in the same framework as
dynamic game environments like Gomoku.

Example:
    from static_data_generator import StaticDatasetGenerator
    from base_data_generator import DynamicGameDataset

    generator = StaticDatasetGenerator(
        data_paths=['data/math/train.parquet'],
        interaction_name='math',
    )
    dataset = DynamicGameDataset(generator, tokenizer, config, virtual_size=10000)
"""

import logging
import os
import random
import threading
from typing import Any, Dict, List, Optional, Union

import datasets
from omegaconf import ListConfig

from opentinker.environment.base_data_generator import AbstractGameDataGenerator

logger = logging.getLogger(__name__)


class StaticDatasetGenerator(AbstractGameDataGenerator):
    """Data generator that wraps static datasets with producer-consumer sampling.

    This generator:
    - Loads static data from parquet/jsonl files once at initialization
    - Provides producer-consumer sampling (cycling through data with shuffling)
    - Returns samples in the same format as GameDataGenerator
    - Supports epoch-based shuffling for better training

    Args:
        data_paths: Path(s) to data files (parquet or jsonl)
        interaction_name: Name for the interaction (used in interaction_kwargs)
        prompt_key: Key in the data dict containing the prompt/messages
        ground_truth_key: Key for the ground truth/answer (optional)
        data_source_key: Key for data source identifier (optional)
        shuffle: Whether to shuffle data at epoch boundaries
        seed: Random seed for reproducibility
        cache_dir: Directory for caching downloaded files

    Example:
        generator = StaticDatasetGenerator(
            data_paths=['data/math/train.parquet'],
            interaction_name='math',
            prompt_key='prompt',
            ground_truth_key='answer',
        )

        sample = generator.generate_sample(0)
        # Returns:
        # {
        #     "prompt": [{"role": "user", "content": "What is 2+2?"}],
        #     "env_kwargs": {"ground_truth": "4"},
        #     "data_source": "math",
        # }
    """

    def __init__(
        self,
        data_paths: Union[str, List[str]],
        interaction_name: str = "math",
        prompt_key: str = "prompt",
        ground_truth_key: Optional[str] = "ground_truth",
        data_source_key: Optional[str] = "data_source",
        extra_keys: Optional[List[str]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        solution_key: Optional[str] = None,
    ):
        # Normalize data_paths to list
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        elif isinstance(data_paths, (list, ListConfig)):
            data_paths = list(data_paths)

        self.data_paths = data_paths
        self._interaction_name = interaction_name
        self.prompt_key = prompt_key
        self.ground_truth_key = ground_truth_key
        self.data_source_key = data_source_key
        self.extra_keys = extra_keys or []
        self.shuffle = shuffle
        self.seed = seed
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/verl/static_data")
        # solution_key: field name for full solution text (for self-distillation)
        self.solution_key = solution_key

        self.system_prompt = system_prompt

        # Thread-safe state
        self._lock = threading.Lock()
        self._index = 0
        self._epoch = 0

        # Load data
        self._samples: List[Dict[str, Any]] = []
        self._indices: List[int] = []
        self._load_data()

        logger.info(
            f"StaticDatasetGenerator initialized: {len(self._samples)} samples from "
            f"{len(data_paths)} file(s), shuffle={shuffle}"
        )

    def _load_data(self):
        """Load data from all data paths."""
        all_samples = []

        for data_path in self.data_paths:
            if not os.path.exists(data_path):
                # Try to download if it's a remote path
                from verl.utils.fs import copy_to_local

                data_path = copy_to_local(src=data_path, cache_dir=self.cache_dir)

            # Determine file type and load
            if data_path.endswith(".parquet"):
                dataset = datasets.load_dataset("parquet", data_files=data_path)[
                    "train"
                ]
            elif data_path.endswith(".jsonl") or data_path.endswith(".json"):
                dataset = datasets.load_dataset("json", data_files=data_path)["train"]
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            # Convert to list of dicts
            for i in range(len(dataset)):
                sample = dict(dataset[i])
                all_samples.append(sample)

            logger.info(f"Loaded {len(dataset)} samples from {data_path}")

        self._samples = all_samples
        self._indices = list(range(len(all_samples)))

        # Initial shuffle if enabled
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self._indices)

    def generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a single training sample using producer-consumer pattern.

        This method cycles through the dataset, shuffling at epoch boundaries.
        The `index` parameter is ignored in favor of internal state tracking
        to ensure proper epoch-based sampling.

        Args:
            index: Sample index (ignored, uses internal counter)

        Returns:
            Dict with prompt, env_kwargs, and data_source
        """
        actual_idx = self._indices[index % len(self._samples)]
        sample = self._samples[actual_idx]

        # Extract prompt (should be list of message dicts)
        prompt = sample.get(self.prompt_key, None)
        if prompt is None:
            # Fallback: try common column names for prompt/question text
            for alt_key in ("Problem", "problem", "question", "Question", "input"):
                if alt_key in sample:
                    prompt = sample[alt_key]
                    break
            else:
                prompt = []
        if not isinstance(prompt, list):
            # If prompt is a string, wrap it in a message format
            prompt = [{"role": "user", "content": str(prompt)}]

        if self.system_prompt:
            # Check if first message is already a system message
            if not prompt or prompt[0].get("role") != "system":
                prompt = [{"role": "system", "content": self.system_prompt}] + prompt

        # Build env_kwargs with ground truth and any extra keys
        env_kwargs = {}
        if self.ground_truth_key and self.ground_truth_key in sample:
            env_kwargs["ground_truth"] = sample[self.ground_truth_key]
        elif (
            self.ground_truth_key
            and "reward_model" in sample
            and self.ground_truth_key in sample["reward_model"]
        ):
            env_kwargs["ground_truth"] = sample["reward_model"][self.ground_truth_key]
        elif (
            self.ground_truth_key
            and "extra_info" in sample
            and self.ground_truth_key in sample["extra_info"]
        ):
            env_kwargs["ground_truth"] = sample["extra_info"][self.ground_truth_key]
        elif self.ground_truth_key:
            # Fallback: try common column names for answer/ground truth
            for alt_key in ("Answer", "answer", "expected_answer", "target"):
                if alt_key in sample:
                    env_kwargs["ground_truth"] = sample[alt_key]
                    break

        # Ensure ground_truth is always a string (some datasets store it as int)
        if "ground_truth" in env_kwargs and not isinstance(env_kwargs["ground_truth"], str):
            env_kwargs["ground_truth"] = str(env_kwargs["ground_truth"])

        for key in self.extra_keys:
            if key in sample:
                env_kwargs[key] = sample[key]

        # Add extra_info to env_kwargs if present in sample
        # This ensures extra_info is passed through to the game's reset() method
        if "extra_info" in sample:
            env_kwargs["extra_info"] = sample["extra_info"]

        # Determine data source
        if self.data_source_key and self.data_source_key in sample:
            data_source = sample[self.data_source_key]
        else:
            data_source = self._interaction_name

        # Add data_source to env_kwargs so it gets passed to game.reset()
        # This is critical for correct reward function routing
        env_kwargs["data_source"] = data_source

        # Extract solution text for self-distillation if configured
        solution_text = None
        if self.solution_key:
            if self.solution_key in sample:
                solution_text = str(sample[self.solution_key])
            elif "extra_info" in sample and self.solution_key in sample.get("extra_info", {}):
                solution_text = str(sample["extra_info"][self.solution_key])
            elif "reward_model" in sample and self.solution_key in sample.get("reward_model", {}):
                solution_text = str(sample["reward_model"][self.solution_key])

        # data_source, solution_str, ground_truth, extra_info
        result = {
            "prompt": prompt,
            "env_kwargs": env_kwargs,
            "data_source": data_source,
        }
        if solution_text is not None:
            result["solution_text"] = solution_text
        return result

    def get_interaction_name(self) -> str:
        """Return the interaction name for this data generator."""
        return self._interaction_name

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._samples)

    @property
    def current_epoch(self) -> int:
        """Return the current epoch number."""
        return self._epoch

    @property
    def current_index(self) -> int:
        """Return the current sample index within the epoch."""
        return self._index

    def reset(self, seed: Optional[int] = None):
        """Reset the generator state.

        Args:
            seed: Optional new seed for shuffling
        """
        with self._lock:
            self._index = 0
            self._epoch = 0
            if seed is not None:
                self.seed = seed
            if self.shuffle:
                if self.seed is not None:
                    random.seed(self.seed)
                random.shuffle(self._indices)
        logger.info("StaticDatasetGenerator reset")
