# Copyright 2026 OpenTinker
#
# Licensed under the Apache License, Version 2.0.
"""Tests for OPSD utilities."""

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from opentinker.server.opsd_utils import build_teacher_ref_batch, extract_cot


class DummyTokenizer:
    """Minimal tokenizer stub for unit tests."""

    pad_token_id = 0
    chat_template = "dummy"

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert not tokenize
        text = "\n".join(item["content"] for item in messages)
        if add_generation_prompt:
            text += "\n<assistant>"
        return text

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        del return_tensors, add_special_tokens
        token_ids = [((ord(ch) % 200) + 1) for ch in text]
        if not token_ids:
            token_ids = [1]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(str(x) for x in token_ids)


def _make_batch() -> DataProto:
    # prompt_len=6, response_len=4, total_len=10
    input_ids = torch.tensor(
        [
            [0, 0, 11, 12, 13, 14, 21, 22, 23, 0],
            [0, 0, 31, 32, 33, 34, 41, 42, 43, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )
    responses = torch.tensor(
        [
            [21, 22, 23, 0],
            [41, 42, 43, 0],
        ],
        dtype=torch.long,
    )
    position_ids = compute_position_id_with_mask(attention_mask)
    extra_info = np.array(
        [
            {"answer": "cot sample 1", "question": "What is 1+1?"},
            {"answer": "cot sample 2", "question": "What is 2+2?"},
        ],
        dtype=object,
    )
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
        },
        non_tensors={"extra_info": extra_info},
        meta_info={},
    )


def test_extract_cot_success():
    assert extract_cot({"answer": "abc"}) == "abc"


def test_extract_cot_missing_raises():
    with pytest.raises(ValueError):
        extract_cot({"question": "x"})


def test_build_teacher_ref_batch_keeps_response_segment():
    batch = _make_batch()
    tokenizer = DummyTokenizer()

    result = build_teacher_ref_batch(batch=batch, tokenizer=tokenizer)
    ref_batch = result.ref_batch

    assert len(result.teacher_prompt_token_lens) == len(batch)
    assert result.missing_cot_count == 0
    assert ref_batch.batch["input_ids"].shape == batch.batch["input_ids"].shape
    assert ref_batch.batch["attention_mask"].shape == batch.batch["attention_mask"].shape
    assert ref_batch.batch["position_ids"].shape == batch.batch["position_ids"].shape

    response_len = batch.batch["responses"].shape[1]
    assert torch.equal(
        ref_batch.batch["input_ids"][:, -response_len:],
        batch.batch["input_ids"][:, -response_len:],
    )
    assert torch.equal(
        ref_batch.batch["attention_mask"][:, -response_len:],
        batch.batch["attention_mask"][:, -response_len:],
    )

    # At least one prompt token should differ after injecting teacher context.
    assert not torch.equal(
        ref_batch.batch["input_ids"][:, :-response_len],
        batch.batch["input_ids"][:, :-response_len],
    )
