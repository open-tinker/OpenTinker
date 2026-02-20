#!/usr/bin/env python3
"""Vision-Language Dynamic Dataset for OpenTinker.

This module extends DynamicGameDataset to support vision-language models
by using AutoProcessor to handle both text and images.
"""

import logging
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from opentinker.environment.base_data_generator import (
    AbstractGameDataGenerator,
    DynamicGameDataset,
    collate_fn,
)
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.vision_utils import process_image

logger = logging.getLogger(__name__)


class DynamicGameDatasetVL(DynamicGameDataset):
    """Dynamic dataset with vision-language support.

    This dataset extends DynamicGameDataset to handle multimodal data
    using AutoProcessor. It processes both text and images from the
    data generator.

    Args:
        data_generator: AbstractGameDataGenerator instance
        tokenizer: Tokenizer for encoding prompts (can be None if processor is provided)
        config: Configuration with max_prompt_length, truncation, etc.
        processor: AutoProcessor for multimodal processing
        virtual_size: Virtual dataset size (for dataloader)
        seed: Optional seed for reproducible generation

    Example:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        dataset = DynamicGameDatasetVL(
            data_generator=generator,
            tokenizer=None,
            processor=processor,
            config=config,
            virtual_size=1000,
        )
    """

    def __init__(
        self,
        data_generator: AbstractGameDataGenerator,
        tokenizer: Optional[PreTrainedTokenizer],
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        virtual_size: int = 1000,
        seed: Optional[int] = None,
    ):
        # For VL models, processor is required
        if processor is None and tokenizer is None:
            raise ValueError("Either processor or tokenizer must be provided")

        # Store processor and initialize parent
        super().__init__(
            data_generator=data_generator,
            tokenizer=tokenizer
            or processor.tokenizer,  # Use processor's tokenizer if no separate tokenizer
            config=config,
            processor=processor,
            virtual_size=virtual_size,
            seed=seed,
        )

        logger.info(
            f"DynamicGameDatasetVL initialized with processor: {processor is not None}"
        )

    def _convert_image_tags(self, messages: list) -> list:
        """Convert <image> tags in message content to structured format for VL processors.

        Qwen2.5-VL and similar VL models expect images to be represented as
        {"type": "image"} in the content list, not as "<image>" string placeholders.

        This follows the pattern from verl's rl_dataset.py _build_messages() method.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Converted messages with structured content
        """
        import re
        import copy

        converted_messages = []
        for message in messages:
            message = copy.deepcopy(message)
            content = message.get("content", "")

            # If content is already a list, it's already structured
            if isinstance(content, list):
                converted_messages.append(message)
                continue

            # If content is a string, convert <image>/<video> tags
            if isinstance(content, str) and (
                "<image>" in content or "<video>" in content
            ):
                content_list = []
                segments = re.split(r"(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]

                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

            converted_messages.append(message)

        return converted_messages

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Get a dynamically generated sample with vision-language data."""
        import random

        # Set seed for reproducible generation if configured
        if self.seed is not None:
            random.seed(self.seed + item)

        # Generate sample from data generator
        sample = self.data_generator.generate_sample(item)
        messages = sample["prompt"]
        images = sample.get("images", [])

        # Convert <image> tags to structured format for Qwen2.5-VL processor
        # This follows the pattern from verl's rl_dataset.py _build_messages()
        if images:
            messages = self._convert_image_tags(messages)

        # Apply chat template and process with processor
        if self.processor is not None:
            # Use processor for multimodal processing
            raw_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )

            # Process text and images together
            # Images can be empty list for text-only samples
            model_inputs = self.processor(
                text=[raw_prompt],
                images=images if images else None,
                return_tensors="pt",
                padding=False,  # We'll handle padding ourselves
            )

            input_ids = model_inputs.pop("input_ids")
            original_input_length = input_ids.shape[-1]  # Save before postprocess
            attention_mask = model_inputs.pop("attention_mask")

            # Store other processor outputs (pixel_values, image_grid_thw, etc.)
            extra_model_inputs = model_inputs
        else:
            # Fallback to text-only processing
            if (
                hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template
            ):
                raw_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                )
                model_inputs = self.tokenizer(
                    raw_prompt, return_tensors="pt", add_special_tokens=False
                )
            else:
                # Simple concatenation fallback
                raw_prompt = (
                    "\n\n".join(
                        f"[{msg['role'].upper()}]\n{msg['content']}" for msg in messages
                    )
                    + "\n\n[ASSISTANT]\n"
                )
                model_inputs = self.tokenizer(
                    raw_prompt, return_tensors="pt", add_special_tokens=True
                )

            input_ids = model_inputs.pop("input_ids")
            original_input_length = input_ids.shape[-1]  # Save before postprocess
            attention_mask = model_inputs.pop("attention_mask")
            extra_model_inputs = {}

        # Postprocess (padding, truncation) for text
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Check if original prompt was too long (for VL models, images add many tokens)
        # If so, skip this sample and try another one to avoid agent_loop postprocess errors
        if original_input_length > self.max_prompt_length:
            # Log warning only occasionally to avoid spam
            if item % 100 == 0:
                logger.warning(
                    f"Sample {item}: prompt length {original_input_length} exceeds max {self.max_prompt_length}, "
                    f"resampling with random index"
                )
            # Resample with a random index to reduce duplicate probability
            # Use a simple hash to get a different but deterministic index for the same item
            new_item = (item * 31 + 17) % len(self)
            if new_item == item:
                new_item = (item + 1) % len(self)
            return self.__getitem__(new_item)

        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)

        # Special handling for Qwen2-VL models (mrope - multimodal rotary position embedding)
        # Qwen2-VL requires 3D vision position IDs + 1D text position IDs = 4D total
        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            try:
                from verl.models.transformers.qwen2_vl import get_rope_index

                # Get image metadata from extra_model_inputs
                image_grid_thw = extra_model_inputs.get("image_grid_thw")
                video_grid_thw = extra_model_inputs.get("video_grid_thw")
                second_per_grid_ts = extra_model_inputs.get("second_per_grid_ts")

                # Calculate vision position ids (3D for mrope)
                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask[0],
                ).unsqueeze(0)  # (1, 3, seq_len)

                # Calculate text position ids
                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                text_position_ids = text_position_ids.unsqueeze(0)  # (1, 1, seq_len)

                # Concatenate: (1, 4, seq_len) - first is text position, next 3 are vision positions
                position_ids = torch.cat(
                    (text_position_ids, vision_position_ids), dim=1
                )

            except ImportError:
                # If verl's qwen2_vl module is not available, fall back to standard position_ids
                import logging

                logging.warning(
                    "Could not import verl.models.transformers.qwen2_vl.get_rope_index, "
                    "falling back to standard position_ids. VL model may not work correctly."
                )

        # Build multi_modal_data for agent loop (contains processed images for vLLM)
        # CRITICAL: This follows verl's pattern - use "image" key (not "images") for vLLM
        multi_modal_data = {}
        if images:
            # Process images for vLLM compatibility
            # Note: process_image handles PIL Image objects from the dataset
            processed_images = []
            for img in images:
                try:
                    # process_image expects PIL Image or bytes, returns processed image
                    processed_img = process_image(img, image_patch_size=14)
                    processed_images.append(processed_img)
                except Exception as e:
                    logger.warning(f"Failed to process image: {e}, using original")
                    processed_images.append(img)  # Fallback to original
            multi_modal_data["image"] = processed_images

        # Build result dict
        row_dict = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "data_source": sample.get("data_source", "game"),
        }

        # Add extra model inputs (image tensors, etc.)
        for key, val in extra_model_inputs.items():
            if val is not None:
                # Remove batch dimension if present
                row_dict[key] = (
                    val[0] if len(val.shape) > 0 and val.shape[0] == 1 else val
                )

        # Raw prompt IDs
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            else:
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
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
        row_dict["reward_model"] = {
            "ground_truth": env_kwargs.get("ground_truth", ""),
            "style": "rule",  # For geo3k reward function
        }

        # CRITICAL: Add multi_modal_data for agent loop to pass to vLLM
        # This contains processed images in the format expected by vLLM ("image" key)
        row_dict["multi_modal_data"] = multi_modal_data

        return row_dict


def collate_fn_vl(data_list: list) -> dict:
    """Collate function for vision-language datasets with variable-sized tensors.

    This collate function handles the case where image tensors (pixel_values)
    can have different sizes due to different image resolutions. It attempts
    to stack tensors where possible, but converts unstackable tensors to
    numpy object arrays for DataProto compatibility.

    DataProto only accepts torch.Tensor and np.ndarray, not lists. When tensors
    cannot be stacked (different shapes), they are converted to np.ndarray with
    dtype=object.

    Args:
        data_list: List of sample dicts from dataset

    Returns:
        Batched dict with stacked tensors and numpy arrays (no lists)
    """
    import torch
    import numpy as np
    from collections import defaultdict

    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    # Try to stack tensors, convert to numpy object array if sizes don't match
    result = {}
    for key, val in tensors.items():
        try:
            # Check if all tensors have the same shape
            shapes = [v.shape for v in val]
            if len(set(shapes)) == 1:
                # All same shape, can stack into a single tensor
                result[key] = torch.stack(val, dim=0)
            else:
                # Different shapes - convert to numpy object array for DataProto compatibility
                # This handles variable-sized tensors like pixel_values in VL models
                result[key] = np.array(val, dtype=object)
        except RuntimeError:
            # Stack failed, convert to numpy object array
            result[key] = np.array(val, dtype=object)

    # Convert non-tensors to numpy arrays
    for key, val in non_tensors.items():
        result[key] = np.fromiter(val, dtype=object, count=len(val))

    return result


# Export VL-specific collate function
__all__ = ["DynamicGameDatasetVL", "collate_fn_vl", "collate_fn"]
