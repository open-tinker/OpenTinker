#!/usr/bin/env python3
"""Base Data Generator for LLM Training Environments.

This module provides abstract base class and utilities for generating
training data for game environments. Users implement generate_sample()
to define their game-specific data generation logic.

Example:
    class MyGameDataGenerator(AbstractGameDataGenerator):
        def generate_sample(self, index: int) -> Dict[str, Any]:
            return {
                "prompt": [{"role": "user", "content": "Play!"}],
                "env_kwargs": {"level": 1},
            }

    generator = MyGameDataGenerator()
    dataset = DynamicGameDataset(generator, tokenizer, config)
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


class AbstractGameDataGenerator(ABC):
    """Abstract base class for game data generators.

    Users implement this interface to define how training samples
    are generated for their specific game/environment.

    Required method:
        - generate_sample(index): Generate a single training sample

    Optional methods:
        - get_interaction_name(): Return the interaction name for routing
    """

    @abstractmethod
    def generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a single training sample.

        Args:
            index: Sample index (can be used for deterministic generation)

        Returns:
            Dict with keys:
                - prompt: List of message dicts (role, content)
                - env_kwargs: Optional dict passed to environment /reset
                - data_source: Optional string identifying data source

        Example:
            return {
                "prompt": [
                    {"role": "system", "content": "You are playing..."},
                    {"role": "user", "content": "Game state: ..."},
                ],
                "env_kwargs": {"initial_moves": [[0, 0], [1, 1]]},
                "data_source": "my_game",
            }
        """
        pass

    def get_interaction_name(self) -> str:
        """Return the interaction name for this data generator.

        This is used in interaction_kwargs to route to the correct
        interaction handler on the server.

        Returns:
            Interaction name string (default: "game")
        """
        return "game"


class DynamicGameDataset(Dataset):
    """Dynamic dataset that generates samples on-the-fly.

    This dataset wraps an AbstractGameDataGenerator to provide
    tokenized samples compatible with the training framework.
    Each __getitem__ call generates a new sample, providing
    infinite variety during training.

    Features:
        - Generates data on-the-fly (no pre-generation needed)
        - Compatible with standard DataLoader
        - Supports tokenization and padding
        - Returns raw_prompt for agent_loop

    Usage:
        generator = MyGameDataGenerator()
        dataset = DynamicGameDataset(
            data_generator=generator,
            tokenizer=tokenizer,
            config=OmegaConf.create({"max_prompt_length": 1024}),
            virtual_size=1000,
        )
    """

    def __init__(
        self,
        data_generator: AbstractGameDataGenerator,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        virtual_size: int = 1000,
        seed: Optional[int] = None,
    ):
        """Initialize DynamicGameDataset.

        Args:
            data_generator: AbstractGameDataGenerator instance
            tokenizer: Tokenizer for encoding prompts
            config: Configuration with max_prompt_length, truncation, etc.
            processor: Optional multimodal processor
            virtual_size: Virtual dataset size (for dataloader)
            seed: Optional seed for reproducible generation
        """
        self.data_generator = data_generator
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.virtual_size = virtual_size
        self.seed = seed

        # Config options
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "right")
        self.return_raw_chat = config.get("return_raw_chat", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        # For compatibility with RLHFDataset interface
        self.serialize_dataset = False
        self.original_data_files = []
        self.data_files = []

        seed_info = f", seed={seed}" if seed is not None else ""
        logger.info(
            f"DynamicGameDataset initialized: virtual_size={virtual_size}{seed_info}"
        )

    def __len__(self) -> int:
        return self.virtual_size

    def _build_raw_prompt(self, messages: List[Dict[str, Any]]) -> str:
        if self.processor is not None:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
        return (
            "\n\n".join(f"[{msg['role'].upper()}]\n{msg['content']}" for msg in messages)
            + "\n\n[ASSISTANT]\n"
        )

    def _encode_with_overflow(
        self, text: str, max_length: int
    ) -> tuple[list[int], bool]:
        if not text:
            return [], False
        safe_max_length = max_length
        if getattr(self.tokenizer, "model_max_length", None):
            try:
                safe_max_length = min(max_length, int(self.tokenizer.model_max_length))
            except (TypeError, ValueError):
                safe_max_length = max_length
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=safe_max_length,
            return_overflowing_tokens=True,
            return_attention_mask=False,
        )
        input_ids = encoded.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        overflow = bool(encoded.get("overflowing_tokens"))
        return list(input_ids), overflow

    def _truncate_messages_to_max_tokens(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not self.max_prompt_length:
            return messages

        raw_prompt = self._build_raw_prompt(messages)
        prompt_ids, overflow = self._encode_with_overflow(
            raw_prompt, self.max_prompt_length
        )
        if not overflow and len(prompt_ids) <= self.max_prompt_length:
            return messages

        truncated = [dict(m) for m in messages]
        while truncated:
            last_idx = len(truncated) - 1
            prefix_messages = [dict(m) for m in truncated]
            prefix_messages[last_idx]["content"] = ""
            prefix_prompt = self._build_raw_prompt(prefix_messages)
            prefix_ids, prefix_overflow = self._encode_with_overflow(
                prefix_prompt, self.max_prompt_length
            )

            if not prefix_overflow and len(prefix_ids) <= self.max_prompt_length:
                remaining = self.max_prompt_length - len(prefix_ids)
                last_content = truncated[last_idx].get("content", "")
                if remaining <= 0:
                    truncated[last_idx]["content"] = ""
                else:
                    last_ids, _ = self._encode_with_overflow(last_content, remaining)
                    truncated[last_idx]["content"] = self.tokenizer.decode(last_ids)
                return truncated

            if len(truncated) > 1:
                truncated = truncated[1:]
                continue

            remaining = max(self.max_prompt_length - len(prefix_ids), 0)
            last_content = truncated[0].get("content", "")
            if remaining > 0:
                last_ids, _ = self._encode_with_overflow(last_content, remaining)
                truncated[0]["content"] = self.tokenizer.decode(last_ids)
            else:
                truncated[0]["content"] = ""
            return truncated

        return messages

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Get a dynamically generated sample."""
        import random

        # Set seed for reproducible generation if configured
        if self.seed is not None:
            random.seed(self.seed + item)

        # Generate sample from data generator
        # breakpoint()
        sample = self.data_generator.generate_sample(item)
        messages = self._truncate_messages_to_max_tokens(sample["prompt"])

        # Apply chat template
        if self.processor is not None:
            raw_prompt = self._build_raw_prompt(messages)
            try:
                model_inputs = self.processor(
                    text=[raw_prompt],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_prompt_length,
                )
            except TypeError:
                model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        elif hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            raw_prompt = self._build_raw_prompt(messages)
            model_inputs = self.tokenizer(
                raw_prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_prompt_length,
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        else:
            # Fallback: simple concatenation
            raw_prompt = self._build_raw_prompt(messages)
            model_inputs = self.tokenizer(
                raw_prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_prompt_length,
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        # Postprocess (padding, truncation)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)

        # Build result dict
        row_dict = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "data_source": sample.get("data_source", "game"),
        }

        # Raw prompt IDs
        raw_prompt_ids, _ = self._encode_with_overflow(
            raw_prompt, self.max_prompt_length
        )
        row_dict["raw_prompt_ids"] = raw_prompt_ids

        # Raw chat for agent_loop
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # Index and interaction kwargs
        row_dict["index"] = item

        # Build interaction_kwargs from env_kwargs
        interaction_name = self.data_generator.get_interaction_name()
        env_kwargs = sample.get("env_kwargs", {})
        row_dict["interaction_kwargs"] = {
            "name": interaction_name,
            "env_kwargs": env_kwargs,
        }
        row_dict["tools_kwargs"] = sample.get("tools_kwargs", {})
        row_dict["extra_info"] = {"interaction_kwargs": row_dict["interaction_kwargs"]}

        # Add reward_model field for NaiveRewardManager compatibility
        # NaiveRewardManager expects ground_truth at non_tensor_batch["reward_model"]["ground_truth"]
        row_dict["reward_model"] = {
            "ground_truth": env_kwargs.get("ground_truth", ""),
        }

        return row_dict

    def resume_dataset_state(self):
        """Resume dataset state (no-op for dynamic dataset)."""
        logger.info("DynamicGameDataset: resume_dataset_state called (no-op)")

    def __getstate__(self):
        """Serialize state."""
        return self.__dict__.copy()


def collate_fn(data_list: List[Dict]) -> Dict:
    """Collate function for DynamicGameDataset batches."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}
