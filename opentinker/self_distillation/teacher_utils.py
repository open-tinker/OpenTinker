"""Teacher input construction for On-Policy Self-Distillation.

The teacher policy p_T(.|x, y*) conditions on both the problem x and the
ground-truth solution y*. This module constructs teacher inputs by inserting
the privileged solution into the prompt, so the same model can produce
teacher log-probs when given these augmented inputs.

Teacher prompt format:
  [system message]
  [user message + "\n\nReference solution: {solution}"]
  [assistant generation prompt]
  [student-generated response tokens]
"""

import logging
from copy import deepcopy
from typing import List, Optional

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

logger = logging.getLogger(__name__)


DEFAULT_SOLUTION_TEMPLATE = "\n\nReference solution: {solution}\n\n After understanding the reference solution, please try to solve this problem using your own approach below:"


def _build_teacher_messages(
    student_messages: list[dict],
    solution_text: str,
    solution_template: str = DEFAULT_SOLUTION_TEMPLATE,
) -> list[dict]:
    """Insert privileged solution into the prompt messages.

    Appends the solution to the last user message content.

    Args:
        student_messages: Original chat messages [{"role": ..., "content": ...}, ...]
        solution_text: Ground-truth solution text
        solution_template: Template with {solution} placeholder

    Returns:
        Modified messages with solution inserted
    """
    if not student_messages:
        return student_messages

    messages = deepcopy(student_messages)

    # Find the last user message and append solution
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            suffix = solution_template.format(solution=solution_text)
            messages[i]["content"] = messages[i]["content"] + suffix
            break

    return messages


def construct_teacher_batch(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    solution_texts: List[str],
    max_teacher_prompt_length: int,
    solution_template: str = DEFAULT_SOLUTION_TEMPLATE,
) -> DataProto:
    """Construct a teacher-conditioned batch for self-distillation.

    Takes the student batch (with generated responses) and constructs a new
    DataProto where prompts include the ground-truth solution as privileged
    context. The responses remain the same (student-generated).

    Args:
        batch: Student batch with keys: input_ids, attention_mask, position_ids,
               responses, response_mask. non_tensor_batch should have raw_prompt.
        tokenizer: Tokenizer for encoding teacher prompts
        solution_texts: List of solution strings, one per sample
        max_teacher_prompt_length: Max length for teacher prompt tokens
        solution_template: Template for inserting solution into prompt

    Returns:
        DataProto with teacher-conditioned inputs, same responses
    """
    batch_size = len(solution_texts)
    responses = batch.batch["responses"]  # (batch_size, response_len)
    response_length = responses.size(-1)

    # Get raw prompts from non_tensor_batch
    raw_prompts = batch.non_tensor_batch.get("raw_prompt", None)

    if raw_prompts is not None:
        # Build teacher prompts from raw chat messages
        teacher_prompt_ids_list = []
        for i in range(batch_size):
            student_messages = raw_prompts[i]
            sol_text = str(solution_texts[i])

            teacher_messages = _build_teacher_messages(
                student_messages, sol_text, solution_template
            )

            # Tokenize teacher prompt
            teacher_prompt_str = tokenizer.apply_chat_template(
                teacher_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            teacher_prompt_tokens = tokenizer.encode(
                teacher_prompt_str, add_special_tokens=False
            )

            # Truncate if needed
            if len(teacher_prompt_tokens) > max_teacher_prompt_length:
                teacher_prompt_tokens = teacher_prompt_tokens[-max_teacher_prompt_length:]

            teacher_prompt_ids_list.append(
                torch.tensor(teacher_prompt_tokens, dtype=torch.long)
            )
    else:
        # Fallback: insert solution tokens directly before response
        # Extract student prompt tokens from input_ids
        teacher_prompt_ids_list = _construct_teacher_prompts_from_tokens(
            batch, tokenizer, solution_texts, max_teacher_prompt_length,
            solution_template, response_length,
        )

    # Build teacher input_ids: [teacher_prompt + response]
    # Need to pad to the same length (left padding)
    teacher_input_ids_list = []
    for i in range(batch_size):
        teacher_input = torch.cat([teacher_prompt_ids_list[i], responses[i]])
        teacher_input_ids_list.append(teacher_input)

    # Pad to max length
    max_len = max(t.size(0) for t in teacher_input_ids_list)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    teacher_input_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=torch.long
    )
    teacher_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i in range(batch_size):
        seq_len = teacher_input_ids_list[i].size(0)
        # Left-pad: place content at the end
        teacher_input_ids[i, max_len - seq_len:] = teacher_input_ids_list[i]
        teacher_attention_mask[i, max_len - seq_len:] = 1

    # Compute position_ids
    teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

    # Build teacher DataProto
    teacher_batch = DataProto.from_dict(
        {
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
            "responses": responses,
        }
    )

    return teacher_batch


def _construct_teacher_prompts_from_tokens(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    solution_texts: List[str],
    max_teacher_prompt_length: int,
    solution_template: str,
    response_length: int,
) -> list[torch.Tensor]:
    """Fallback: construct teacher prompts by inserting solution tokens.

    Used when raw_prompt is not available. Extracts prompt tokens from
    input_ids and inserts tokenized solution before the generation marker.

    Strategy: find the last occurrence of the generation prompt tokens
    (e.g., <|im_start|>assistant\n) in the prompt, and insert the solution
    text just before it.
    """
    input_ids = batch.batch["input_ids"]  # (batch, prompt_len + response_len)
    attention_mask = batch.batch["attention_mask"]

    # Extract prompt tokens (everything before response)
    prompt_ids = input_ids[:, :-response_length]  # (batch, prompt_len)
    prompt_mask = attention_mask[:, :-response_length]

    # Tokenize the generation prompt marker to find insertion point
    gen_prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": "X"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    # Extract just the assistant marker part (after "X")
    user_end_idx = gen_prompt_str.rfind("X") + 1
    assistant_marker = gen_prompt_str[user_end_idx:]
    assistant_marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)

    teacher_prompt_ids_list = []
    for i in range(len(solution_texts)):
        # Get actual (non-padded) prompt tokens
        mask = prompt_mask[i]
        valid_start = mask.nonzero(as_tuple=True)[0]
        if len(valid_start) == 0:
            teacher_prompt_ids_list.append(prompt_ids[i])
            continue

        start_idx = valid_start[0].item()
        actual_prompt = prompt_ids[i, start_idx:]  # remove left padding

        # Tokenize solution suffix
        sol_text = str(solution_texts[i])
        suffix = solution_template.format(solution=sol_text)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long)

        # Find assistant marker in prompt and insert solution before it
        marker_len = len(assistant_marker_ids)
        insertion_point = None
        if marker_len > 0:
            marker_tensor = torch.tensor(assistant_marker_ids, dtype=torch.long)
            for j in range(len(actual_prompt) - marker_len, -1, -1):
                if torch.equal(actual_prompt[j:j + marker_len], marker_tensor):
                    insertion_point = j
                    break

        if insertion_point is not None:
            teacher_prompt = torch.cat([
                actual_prompt[:insertion_point],
                suffix_tensor,
                actual_prompt[insertion_point:],
            ])
        else:
            # Fallback: append solution before the last token of prompt
            teacher_prompt = torch.cat([
                actual_prompt[:-1],
                suffix_tensor,
                actual_prompt[-1:],
            ])

        # Truncate if needed
        if len(teacher_prompt) > max_teacher_prompt_length:
            teacher_prompt = teacher_prompt[-max_teacher_prompt_length:]

        teacher_prompt_ids_list.append(teacher_prompt)

    return teacher_prompt_ids_list
