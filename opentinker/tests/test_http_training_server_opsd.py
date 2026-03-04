# Copyright 2026 OpenTinker
#
# Licensed under the Apache License, Version 2.0.
"""Tests for OPSD full-vocab-JSD config gating in HTTP training server."""

import pytest
from omegaconf import OmegaConf

from opentinker.server.opsd_config_utils import (
    resolve_opsd_runtime_flags,
    should_create_ref_policy_worker,
    validate_opsd_full_vocab_jsd_config,
)


def _base_cfg():
    return OmegaConf.create(
        {
            "algorithm": {
                "use_kl_in_reward": False,
                "use_kl_in_advantage": False,
                "opsd": {
                    "enable": True,
                    "teacher_mode": "fixed",
                    "full_vocab_jsd": {"enable": True},
                },
            },
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": False,
                    "strategy": "fsdp",
                }
            },
        }
    )


def test_full_vocab_jsd_no_conflict_skips_ref_policy_worker():
    cfg = _base_cfg()
    flags = resolve_opsd_runtime_flags(cfg)
    validate_opsd_full_vocab_jsd_config(cfg, **flags)
    assert should_create_ref_policy_worker(cfg, **flags) is False


@pytest.mark.parametrize(
    "override, expected_field",
    [
        ({"algorithm": {"use_kl_in_reward": True}}, "algorithm.use_kl_in_reward"),
        (
            {"algorithm": {"use_kl_in_advantage": True}},
            "algorithm.use_kl_in_advantage",
        ),
    ],
)
def test_full_vocab_jsd_conflicts_raise(override, expected_field):
    cfg = OmegaConf.merge(_base_cfg(), OmegaConf.create(override))
    flags = resolve_opsd_runtime_flags(cfg)
    with pytest.raises(ValueError, match=expected_field):
        validate_opsd_full_vocab_jsd_config(cfg, **flags)


def test_full_vocab_jsd_allows_actor_use_kl_loss():
    cfg = OmegaConf.merge(
        _base_cfg(),
        OmegaConf.create({"actor_rollout_ref": {"actor": {"use_kl_loss": True}}}),
    )
    flags = resolve_opsd_runtime_flags(cfg)
    validate_opsd_full_vocab_jsd_config(cfg, **flags)
    assert should_create_ref_policy_worker(cfg, **flags) is False
