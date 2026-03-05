# Copyright 2026 OpenTinker
#
# Licensed under the Apache License, Version 2.0.
"""Lightweight OPSD config helpers used by the HTTP training server."""

from __future__ import annotations


def resolve_opsd_runtime_flags(config):
    """Resolve OPSD/full-vocab-JSD mode flags from config."""
    opsd_cfg = config.algorithm.get("opsd", {}) if hasattr(config, "algorithm") else {}
    opsd_enabled = bool(opsd_cfg.get("enable", False))
    opsd_teacher_mode = str(opsd_cfg.get("teacher_mode", "fixed")).lower()
    opsd_full_vocab_cfg = opsd_cfg.get("full_vocab_jsd", {})
    opsd_full_vocab_jsd_enabled = bool(opsd_full_vocab_cfg.get("enable", False))
    if opsd_teacher_mode not in {"fixed", "shared"}:
        raise ValueError(
            f"Invalid algorithm.opsd.teacher_mode={opsd_teacher_mode!r}. "
            "Expected one of: 'fixed', 'shared'."
        )
    opsd_shared_teacher = opsd_enabled and opsd_teacher_mode == "shared"
    return {
        "opsd_enabled": opsd_enabled,
        "opsd_teacher_mode": opsd_teacher_mode,
        "opsd_full_vocab_jsd_enabled": opsd_full_vocab_jsd_enabled,
        "opsd_shared_teacher": opsd_shared_teacher,
    }


def validate_opsd_full_vocab_jsd_config(
    config,
    *,
    opsd_enabled: bool,
    opsd_full_vocab_jsd_enabled: bool,
    **_,
):
    """Validate full-vocab JSD compatibility and fail fast on unsupported mixes."""
    if not (opsd_enabled and opsd_full_vocab_jsd_enabled):
        return

    conflicts = []
    if bool(config.algorithm.get("use_kl_in_reward", False)):
        conflicts.append("algorithm.use_kl_in_reward")
    if bool(config.algorithm.get("use_kl_in_advantage", False)):
        conflicts.append("algorithm.use_kl_in_advantage")

    if conflicts:
        raise ValueError(
            "algorithm.opsd.full_vocab_jsd.enable=true is incompatible with: "
            + ", ".join(conflicts)
        )


def should_create_ref_policy_worker(
    config,
    *,
    opsd_enabled: bool,
    opsd_shared_teacher: bool,
    opsd_full_vocab_jsd_enabled: bool,
    **_,
) -> bool:
    """Return whether a standalone RefPolicy worker is required."""
    if not hasattr(config, "algorithm"):
        return False
    if opsd_enabled and opsd_full_vocab_jsd_enabled:
        return False
    return bool(
        config.algorithm.use_kl_in_reward
        or (config.actor_rollout_ref.actor.use_kl_loss and not opsd_shared_teacher)
        or (
            config.algorithm.get("use_kl_in_advantage", False)
            and not opsd_shared_teacher
        )
        or (
            opsd_enabled
            and not opsd_shared_teacher
            and not opsd_full_vocab_jsd_enabled
        )
    )
