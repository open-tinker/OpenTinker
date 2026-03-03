#!/usr/bin/env python3
"""Utilities for On-Policy Self-Distillation (OPSD) teacher context construction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

# Keep template as an internal constant (no extra public config knobs).
DEFAULT_TEACHER_PROMPT_TEMPLATE = (
    "Problem:\n{question}\n\n"
    "Here is a reference solution:\n{cot}\n\n"
    "After understanding the reference solution, please try to solve this problem\n"
    "using your own approach below:\n"
    "Answer:"
)


@dataclass
class OpsdTeacherBatchResult:
    """Result of building teacher-context ref batch for OPSD."""

    ref_batch: DataProto
    teacher_prompt_token_lens: list[int]
    missing_cot_count: int


def extract_cot(extra_info: Any) -> str:
    """Extract privileged CoT from sample extra_info.

    This implementation intentionally uses a fixed field path:
    `extra_info.answer`.
    """
    if isinstance(extra_info, dict):
        cot = extra_info.get("answer")
        if isinstance(cot, str) and cot.strip():
            return cot
    raise ValueError("Missing or empty CoT: expected non-empty extra_info.answer")


def _extract_question(extra_info: Any) -> str | None:
    """Extract question text from extra_info if available."""
    if not isinstance(extra_info, dict):
        return None

    question = extra_info.get("question")
    if isinstance(question, str) and question.strip():
        return question

    interaction_kwargs = extra_info.get("interaction_kwargs")
    if isinstance(interaction_kwargs, dict):
        query = interaction_kwargs.get("query")
        if isinstance(query, str) and query.strip():
            return query

    return None


def _truncate_and_left_pad(
    token_ids: torch.Tensor,
    target_len: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate to target_len (from right side) and left-pad."""
    if token_ids.ndim != 1:
        raise ValueError(f"Expected 1-D token tensor, got shape={tuple(token_ids.shape)}")

    if token_ids.numel() > target_len:
        token_ids = token_ids[:target_len]

    attn = torch.ones_like(token_ids, dtype=torch.long)

    if token_ids.numel() < target_len:
        pad_len = target_len - token_ids.numel()
        pad_ids = torch.full((pad_len,), int(pad_token_id), dtype=token_ids.dtype)
        pad_attn = torch.zeros((pad_len,), dtype=torch.long)
        token_ids = torch.cat([pad_ids, token_ids], dim=0)
        attn = torch.cat([pad_attn, attn], dim=0)

    return token_ids, attn


def _encode_teacher_prompt(
    tokenizer: Any,
    question: str,
    cot: str,
    prompt_template: str,
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode and pad/truncate teacher prompt to fixed prompt_len."""
    teacher_content = prompt_template.format(question=question, cot=cot)

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": teacher_content}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt_text = teacher_content

    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = encoded["input_ids"][0].to(dtype=torch.long)
    prompt_ids, prompt_attn = _truncate_and_left_pad(
        token_ids=prompt_ids,
        target_len=prompt_len,
        pad_token_id=int(tokenizer.pad_token_id),
    )
    prompt_token_len = int(prompt_attn.sum().item())
    return prompt_ids, prompt_attn, prompt_token_len


def _iter_extra_infos(extra_infos: Iterable[Any]) -> list[Any]:
    """Normalize non-tensor extra_info array/list to plain Python list."""
    if isinstance(extra_infos, np.ndarray):
        return extra_infos.tolist()
    return list(extra_infos)


def build_teacher_ref_batch(
    batch: DataProto,
    tokenizer: Any,
    prompt_template: str = DEFAULT_TEACHER_PROMPT_TEMPLATE,
) -> OpsdTeacherBatchResult:
    """Build teacher-context batch for reference log-prob computation.

    The returned batch keeps the same responses and response segment as the
    student batch, and only swaps the prompt prefix to a privileged teacher prompt.
    """
    required_batch_keys = {"input_ids", "attention_mask", "position_ids", "responses"}
    missing_batch_keys = required_batch_keys - set(batch.batch.keys())
    if missing_batch_keys:
        raise ValueError(f"Missing required tensor keys in batch: {sorted(missing_batch_keys)}")

    if "extra_info" not in batch.non_tensor_batch:
        raise ValueError("Missing non_tensor_batch['extra_info'] required for OPSD")

    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    position_ids = batch.batch["position_ids"]
    responses = batch.batch["responses"]

    if input_ids.ndim != 2 or attention_mask.ndim != 2 or responses.ndim != 2:
        raise ValueError(
            "Expected input_ids/attention_mask/responses to be 2-D tensors, "
            f"got input_ids={tuple(input_ids.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}, "
            f"responses={tuple(responses.shape)}"
        )

    batch_size = input_ids.shape[0]
    response_len = responses.shape[1]
    total_len = input_ids.shape[1]
    prompt_len = total_len - response_len
    if prompt_len <= 0:
        raise ValueError(f"Invalid prompt_len={prompt_len} from total_len={total_len}, response_len={response_len}")

    extra_infos = _iter_extra_infos(batch.non_tensor_batch["extra_info"])
    if len(extra_infos) != batch_size:
        raise ValueError(
            f"extra_info length mismatch: got {len(extra_infos)}, expected batch size {batch_size}"
        )

    teacher_prompt_token_lens: list[int] = []
    missing_cot_count = 0
    new_input_ids = []
    new_attention_mask = []

    for i in range(batch_size):
        extra_info = extra_infos[i]

        try:
            cot = extract_cot(extra_info)
        except ValueError:
            missing_cot_count += 1
            raise

        question = _extract_question(extra_info)
        if not question:
            valid_prompt_tokens = input_ids[i, :prompt_len][attention_mask[i, :prompt_len].bool()]
            question = tokenizer.decode(valid_prompt_tokens.tolist(), skip_special_tokens=True)

        prompt_ids, prompt_attn, prompt_token_len = _encode_teacher_prompt(
            tokenizer=tokenizer,
            question=question,
            cot=cot,
            prompt_template=prompt_template,
            prompt_len=prompt_len,
        )
        teacher_prompt_token_lens.append(prompt_token_len)

        response_ids = input_ids[i, prompt_len:]
        response_attn = attention_mask[i, prompt_len:].to(dtype=torch.long)

        full_ids = torch.cat([prompt_ids.to(dtype=input_ids.dtype), response_ids], dim=0)
        full_attn = torch.cat([prompt_attn.to(dtype=attention_mask.dtype), response_attn], dim=0)

        new_input_ids.append(full_ids)
        new_attention_mask.append(full_attn)

    new_input_ids_t = torch.stack(new_input_ids, dim=0)
    new_attention_mask_t = torch.stack(new_attention_mask, dim=0)

    if position_ids.ndim == 2:
        new_position_ids_t = compute_position_id_with_mask(new_attention_mask_t)
    else:
        # Keep existing position IDs for non-standard/multimodal layouts.
        new_position_ids_t = position_ids

    ref_batch = DataProto.from_dict(
        tensors={key: tensor.clone() for key, tensor in batch.batch.items()},
        non_tensors=copy.deepcopy(batch.non_tensor_batch),
        meta_info=copy.deepcopy(batch.meta_info),
    )
    ref_batch.batch["input_ids"] = new_input_ids_t
    ref_batch.batch["attention_mask"] = new_attention_mask_t
    ref_batch.batch["position_ids"] = new_position_ids_t

    return OpsdTeacherBatchResult(
        ref_batch=ref_batch,
        teacher_prompt_token_lens=teacher_prompt_token_lens,
        missing_cot_count=missing_cot_count,
    )
