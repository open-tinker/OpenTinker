"""Shared OPSD mode validation helpers.

This module centralizes OPSD teacher/loss mode defaults and validation so
client/server/worker code paths stay consistent.
"""

from collections.abc import Mapping
from typing import Any, Tuple

import torch

VALID_OPSD_TEACHER_SOURCES = frozenset({"online", "initial_frozen"})
VALID_OPSD_LOSS_MODES = frozenset({"full_jsd", "sampled_kl"})

DEFAULT_OPSD_TEACHER_SOURCE = "online"
DEFAULT_OPSD_LOSS_MODE = "full_jsd"


def _get_cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def validate_opsd_modes(algorithm_cfg: Any) -> Tuple[str, str]:
    teacher_source = str(
        _get_cfg_value(
            algorithm_cfg,
            "opsd_teacher_source",
            DEFAULT_OPSD_TEACHER_SOURCE,
        )
    )
    loss_mode = str(
        _get_cfg_value(
            algorithm_cfg,
            "opsd_loss_mode",
            DEFAULT_OPSD_LOSS_MODE,
        )
    )

    if teacher_source not in VALID_OPSD_TEACHER_SOURCES:
        allowed = ", ".join(sorted(VALID_OPSD_TEACHER_SOURCES))
        raise ValueError(
            f"Invalid algorithm.opsd_teacher_source={teacher_source!r}. "
            f"Expected one of: {allowed}."
        )

    if loss_mode not in VALID_OPSD_LOSS_MODES:
        allowed = ", ".join(sorted(VALID_OPSD_LOSS_MODES))
        raise ValueError(
            f"Invalid algorithm.opsd_loss_mode={loss_mode!r}. "
            f"Expected one of: {allowed}."
        )

    return teacher_source, loss_mode


def should_use_initial_frozen_teacher(algorithm_cfg: Any) -> bool:
    teacher_source, _ = validate_opsd_modes(algorithm_cfg)
    use_opsd = bool(
        _get_cfg_value(algorithm_cfg, "use_opsd_jsd_in_advantage", False)
        or _get_cfg_value(algorithm_cfg, "use_opsd", False)
    )
    return use_opsd and teacher_source == "initial_frozen"


def compute_sampled_kl_penalty(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token sampled KL term used in OPSD sampled_kl mode."""
    return (student_log_probs - teacher_log_probs) * response_mask
