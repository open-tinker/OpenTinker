# Copyright 2026 OpenTinker
#
# Licensed under the Apache License, Version 2.0.
"""Utilities for building RLTF-SD auxiliary training batches."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def _to_pair_list(raw_pairs: Any) -> list[dict[str, Any]]:
    if raw_pairs is None:
        return []
    if isinstance(raw_pairs, dict):
        return [raw_pairs]
    if isinstance(raw_pairs, (list, tuple)):
        return [p for p in raw_pairs if isinstance(p, dict)]
    if isinstance(raw_pairs, np.ndarray):
        flat = raw_pairs.reshape(-1).tolist()
        return [p for p in flat if isinstance(p, dict)]
    return []


def _left_pad(ids: list[int], max_len: int, pad_id: int) -> tuple[list[int], list[int]]:
    ids = ids[-max_len:]
    pad_len = max_len - len(ids)
    return [pad_id] * pad_len + ids, [0] * pad_len + [1] * len(ids)


def _right_pad(ids: list[int], max_len: int, pad_id: int) -> tuple[list[int], list[int]]:
    ids = ids[:max_len]
    pad_len = max_len - len(ids)
    return ids + [pad_id] * pad_len, [1] * len(ids) + [0] * pad_len


def build_rltf_sd_training_batch(
    *,
    batch: DataProto,
    tokenizer,
    prompt_length: int,
    response_length: int,
    sd_coef: float,
    gamma: float,
    max_pairs_per_episode: int,
    temperature: float,
    loss_type: str = "awr",
    kl_enable: bool = False,
    kl_teacher_mode: str = "fixed",
    kl_distill_mode: str = "topk_reverse_kl_tail",
    kl_topk: int = 50,
    kl_beta: float = 0.5,
    kl_coef: float = 1.0,
    kl_vocab_chunk_size: int = 4096,
    kl_token_chunk_size: int = 0,
    kl_teacher_logits_cpu_offload: bool = True,
) -> tuple[Optional[DataProto], dict[str, float]]:
    """Parse `rltf_sd_pair` metadata and build an auxiliary SD batch."""
    loss_type = str(loss_type).lower()
    if loss_type not in {"awr", "kl"}:
        raise ValueError(
            f"Unsupported algorithm.rltf_sd.loss_type={loss_type!r}. "
            "Expected one of {'awr', 'kl'}."
        )
    use_kl_loss = loss_type == "kl" and bool(kl_enable)

    stats = {
        "pair_count": 0.0,
        "b0_mean": 0.0,
        "adv_mean": 0.0,
    }

    pair_field = batch.non_tensor_batch.get("rltf_sd_pair")
    if pair_field is None:
        return None, stats

    uid_field = batch.non_tensor_batch.get("uid")

    collected: list[dict[str, Any]] = []
    for sample_idx in range(len(batch)):
        sample_pairs = _to_pair_list(pair_field[sample_idx])
        kept = 0
        for pair in sample_pairs:
            if max_pairs_per_episode > 0 and kept >= max_pairs_per_episode:
                break

            x_t_ids = pair.get("x_t_token_ids")
            x_t_prime_ids = pair.get("x_t_prime_token_ids")
            y1_ids = pair.get("y1_token_ids")
            r0 = pair.get("r0")
            r1 = pair.get("r1")
            if x_t_ids is None or y1_ids is None or r0 is None or r1 is None:
                continue
            if use_kl_loss and x_t_prime_ids is None:
                continue

            x_t_ids = [int(tok) for tok in x_t_ids]
            x_t_prime_ids = (
                [int(tok) for tok in x_t_prime_ids]
                if x_t_prime_ids is not None
                else None
            )
            y1_ids = [int(tok) for tok in y1_ids]
            if not x_t_ids or not y1_ids:
                continue
            if use_kl_loss and not x_t_prime_ids:
                continue

            turn_id = int(pair.get("turn_id", kept + 1))
            if "group_id" in pair and pair["group_id"] is not None:
                group_id = str(pair["group_id"])
            else:
                uid = str(uid_field[sample_idx]) if uid_field is not None else str(sample_idx)
                group_id = f"{uid}::turn_{turn_id}"

            collected.append(
                {
                    "group_id": group_id,
                    "turn_id": turn_id,
                    "r0": float(r0),
                    "r1": float(r1),
                    "x_t_token_ids": x_t_ids,
                    "x_t_prime_token_ids": x_t_prime_ids,
                    "y1_token_ids": y1_ids,
                }
            )
            kept += 1

    if not collected:
        return None, stats

    # First-turn baseline per group: b(0) = mean(r0)
    group_r0: dict[str, list[float]] = defaultdict(list)
    for item in collected:
        group_r0[item["group_id"]].append(float(item["r0"]))
    group_b0 = {gid: float(np.mean(vals)) for gid, vals in group_r0.items()}

    rows: list[dict[str, Any]] = []
    b0_values = []
    adv_values = []
    for item in collected:
        b0 = group_b0[item["group_id"]]
        turn_discount = float(gamma) ** max(0, int(item["turn_id"]) - 1)
        adv = (float(item["r1"]) - b0) * turn_discount

        b0_values.append(b0)
        adv_values.append(adv)
        rows.append(
            {
                "x_t_token_ids": item["x_t_token_ids"],
                "x_t_prime_token_ids": item["x_t_prime_token_ids"],
                "y1_token_ids": item["y1_token_ids"],
                "adv": adv,
            }
        )

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    pad_id = int(pad_token_id)

    prompts = []
    responses = []
    response_masks = []
    attention_masks = []
    advantages = []
    teacher_attention_masks = []
    teacher_input_ids = []

    for row in rows:
        prompt_ids, prompt_mask = _left_pad(row["x_t_token_ids"], prompt_length, pad_id)
        response_ids, response_mask = _right_pad(
            row["y1_token_ids"], response_length, pad_id
        )
        valid_tokens = int(sum(response_mask))
        if valid_tokens <= 0:
            continue

        prompts.append(prompt_ids)
        responses.append(response_ids)
        response_masks.append(response_mask)
        attention_masks.append(prompt_mask + response_mask)

        if use_kl_loss:
            x_t_prime_ids = row["x_t_prime_token_ids"]
            if x_t_prime_ids is None:
                continue
            teacher_prompt_ids, teacher_prompt_mask = _left_pad(
                x_t_prime_ids, prompt_length, pad_id
            )
            teacher_attention_masks.append(teacher_prompt_mask + response_mask)
            teacher_input_ids.append(teacher_prompt_ids + response_ids)

        adv_row = [0.0] * response_length
        for idx in range(valid_tokens):
            adv_row[idx] = float(row["adv"])
        advantages.append(adv_row)

    if not prompts:
        return None, stats

    prompts_t = torch.tensor(prompts, dtype=torch.long)
    responses_t = torch.tensor(responses, dtype=torch.long)
    response_mask_t = torch.tensor(response_masks, dtype=torch.long)
    attention_mask_t = torch.tensor(attention_masks, dtype=torch.long)
    input_ids_t = torch.cat([prompts_t, responses_t], dim=1)
    position_ids_t = compute_position_id_with_mask(attention_mask_t)
    if use_kl_loss:
        teacher_input_ids_t = torch.tensor(teacher_input_ids, dtype=torch.long)
        teacher_attention_mask_t = torch.tensor(teacher_attention_masks, dtype=torch.long)
        teacher_position_ids_t = compute_position_id_with_mask(teacher_attention_mask_t)

        sd_batch = DataProto.from_dict(
            tensors={
                "prompts": prompts_t,
                "responses": responses_t,
                "response_mask": response_mask_t,
                "input_ids": input_ids_t,
                "attention_mask": attention_mask_t,
                "position_ids": position_ids_t,
                "teacher_input_ids": teacher_input_ids_t,
                "teacher_attention_mask": teacher_attention_mask_t,
                "teacher_position_ids": teacher_position_ids_t,
            },
            non_tensors={},
            meta_info={
                "temperature": float(temperature),
                "rltf_sd_loss_type": "kl",
                "rltf_sd_kl_enable": 1.0,
                "rltf_sd_kl_teacher_mode": str(kl_teacher_mode).lower(),
                "rltf_sd_kl_distill_mode": str(kl_distill_mode).lower(),
                "rltf_sd_kl_topk": int(kl_topk),
                "rltf_sd_kl_beta": float(kl_beta),
                "rltf_sd_kl_coef": float(kl_coef),
                "rltf_sd_kl_vocab_chunk_size": int(kl_vocab_chunk_size),
                "rltf_sd_kl_token_chunk_size": int(kl_token_chunk_size),
                "rltf_sd_kl_teacher_logits_cpu_offload": bool(
                    kl_teacher_logits_cpu_offload
                ),
                "global_token_num": attention_mask_t.sum(dim=-1).tolist(),
            },
        )
    else:
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        sd_batch = DataProto.from_dict(
            tensors={
                "prompts": prompts_t,
                "responses": responses_t,
                "response_mask": response_mask_t,
                "input_ids": input_ids_t,
                "attention_mask": attention_mask_t,
                "position_ids": position_ids_t,
                "rltf_sd_advantages": advantages_t,
            },
            non_tensors={},
            meta_info={
                "temperature": float(temperature),
                "rltf_sd_loss_type": "awr",
                "rltf_sd_coef": float(sd_coef),
                "global_token_num": attention_mask_t.sum(dim=-1).tolist(),
            },
        )

    stats["pair_count"] = float(len(prompts))
    stats["b0_mean"] = float(np.mean(b0_values)) if b0_values else 0.0
    stats["adv_mean"] = float(np.mean(adv_values)) if adv_values else 0.0
    return sd_batch, stats
