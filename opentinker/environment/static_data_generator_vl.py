#!/usr/bin/env python3
"""Vision-Language Static Data Generator for OpenTinker.

This module extends StaticDatasetGenerator to support vision-language models
by loading and processing images from parquet files.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from opentinker.environment.static_data_generator import StaticDatasetGenerator

if TYPE_CHECKING:
    from transformers import ProcessorMixin

logger = logging.getLogger(__name__)


class StaticDatasetGeneratorVL(StaticDatasetGenerator):
    """Static dataset generator with vision-language support.

    This generator extends StaticDatasetGenerator to handle image data
    from parquet files. Images are typically stored as lists of PIL images
    or image paths in the dataset.

    Args:
        data_paths: List of parquet file paths
        interaction_name: Name of the interaction handler
        prompt_key: Key for prompt field in data (default: "prompt")
        ground_truth_key: Key for ground truth answer (default: "ground_truth")
        image_key: Key for image field in data (default: "images")
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling
        system_prompt: Optional system prompt to prepend

    Example:
        generator = StaticDatasetGeneratorVL(
            data_paths=["~/data/geo3k/train.parquet"],
            interaction_name="game",
            image_key="images",
        )
    """

    def __init__(
        self,
        data_paths: List[str],
        interaction_name: str = "game",
        prompt_key: str = "prompt",
        ground_truth_key: str = "ground_truth",
        image_key: str = "images",
        shuffle: bool = False,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(
            data_paths=data_paths,
            interaction_name=interaction_name,
            prompt_key=prompt_key,
            ground_truth_key=ground_truth_key,
            shuffle=shuffle,
            seed=seed,
            system_prompt=system_prompt,
        )
        self.image_key = image_key
        logger.info(
            f"StaticDatasetGeneratorVL initialized with image_key='{image_key}'"
        )

    def filter_overlong_samples(
        self,
        processor: "ProcessorMixin",
        max_prompt_length: int,
        apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Filter out samples whose tokenized length (including image tokens) exceeds max_prompt_length.

        This is similar to verl's RLHFDataset.maybe_filter_out_long_prompts() but adapted
        for our generator structure. This prevents the "max_tokens must be at least 1" vLLM error
        that occurs when prompt tokens exceed the context budget.

        Args:
            processor: HuggingFace processor (with tokenizer and image processor)
            max_prompt_length: Maximum allowed prompt length in tokens
            apply_chat_template_kwargs: Optional kwargs for apply_chat_template

        Returns:
            Dict with filtering statistics:
                - original_count: Number of samples before filtering
                - filtered_count: Number of samples after filtering
                - removed_count: Number of samples removed
                - removed_ratio: Ratio of samples removed
        """
        apply_kwargs = apply_chat_template_kwargs or {}
        original_count = len(self._samples)

        def compute_sample_length(idx: int) -> int:
            """Compute tokenized length of a sample including image tokens."""
            try:
                row = self._samples[idx]

                # Build messages (same as generate_sample)
                prompt = row.get(self.prompt_key, [])
                messages = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                if isinstance(prompt, list):
                    messages.extend(prompt)
                elif isinstance(prompt, str):
                    messages.append({"role": "user", "content": prompt})

                # Get raw prompt text
                raw_prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                )

                # Get images
                images = None
                if self.image_key in row and row[self.image_key]:
                    images = row[self.image_key]
                    if not isinstance(images, list):
                        images = [images]

                # Tokenize with processor (this includes image token expansion)
                model_inputs = processor(
                    text=[raw_prompt],
                    images=images if images else None,
                    return_tensors="pt",
                )

                return len(model_inputs["input_ids"][0])

            except Exception as e:
                logger.warning(f"Error computing sample length for idx {idx}: {e}")
                traceback.print_exc()
                # Return a value that will cause this sample to be filtered
                return max_prompt_length + 1

        # Filter samples
        valid_indices = []
        removed_lengths = []

        print(f"\n{'='*60}")
        print("[VL Pre-Filter] Starting overlong sample filtering...")
        print(f"[VL Pre-Filter] max_prompt_length = {max_prompt_length}")
        print(f"[VL Pre-Filter] Processing {original_count} samples...")

        for idx in range(original_count):
            length = compute_sample_length(idx)
            if length <= max_prompt_length:
                valid_indices.append(idx)
            else:
                removed_lengths.append(length)
                if len(removed_lengths) <= 5:  # Log first 5 removed samples
                    print(
                        f"[VL Pre-Filter]   Removed sample {idx}: length={length} > {max_prompt_length}"
                    )

        # Update internal samples list
        self._samples = [self._samples[i] for i in valid_indices]
        self._indices = list(range(len(self._samples)))

        # Compute stats
        filtered_count = len(self._samples)
        removed_count = original_count - filtered_count
        removed_ratio = removed_count / original_count if original_count > 0 else 0.0

        # Print detailed statistics
        print("[VL Pre-Filter] Filtering complete!")
        print(f"[VL Pre-Filter]   Original samples: {original_count}")
        print(
            f"[VL Pre-Filter]   Removed samples:  {removed_count} ({removed_ratio*100:.1f}%)"
        )
        print(f"[VL Pre-Filter]   Remaining samples: {filtered_count}")

        if removed_lengths:
            avg_removed_len = sum(removed_lengths) / len(removed_lengths)
            max_removed_len = max(removed_lengths)
            print(f"[VL Pre-Filter]   Avg removed length: {avg_removed_len:.0f} tokens")
            print(f"[VL Pre-Filter]   Max removed length: {max_removed_len} tokens")

        if removed_ratio > 0.2:
            print(
                f"[VL Pre-Filter] ⚠️  WARNING: High filtering ratio ({removed_ratio*100:.1f}%)!"
            )
            print("[VL Pre-Filter]    Consider increasing max_prompt_length in config.")

        print(f"{'='*60}\n")

        stats = {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "removed_count": removed_count,
            "removed_ratio": removed_ratio,
        }

        return stats

    def generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a sample with vision-language data.

        Args:
            index: Sample index

        Returns:
            Dict with keys:
                - prompt: List of message dicts
                - env_kwargs: Dict with ground_truth
                - images: List of images (if present)
                - data_source: Data source identifier
        """
        # Get base sample from parent class
        sample = super().generate_sample(index)

        # Add images if present in the data
        actual_idx = self._indices[index % len(self._samples)]
        row = self._samples[actual_idx]
        if self.image_key in row:
            images = row[self.image_key]
            # Ensure images is a list
            if not isinstance(images, list):
                images = [images] if images is not None else []
            sample["images"] = images
        else:
            sample["images"] = []

        return sample
