from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentinker.server.opsd_config import (
    compute_sampled_kl_penalty,
    should_use_initial_frozen_teacher,
    validate_opsd_modes,
)


def test_validate_opsd_modes_defaults():
    teacher_source, loss_mode = validate_opsd_modes({})
    assert teacher_source == "online"
    assert loss_mode == "full_jsd"


def test_validate_opsd_modes_valid_values():
    teacher_source, loss_mode = validate_opsd_modes(
        {
            "opsd_teacher_source": "initial_frozen",
            "opsd_loss_mode": "sampled_kl",
        }
    )
    assert teacher_source == "initial_frozen"
    assert loss_mode == "sampled_kl"


def test_validate_opsd_modes_invalid_teacher_source():
    with pytest.raises(ValueError, match="opsd_teacher_source"):
        validate_opsd_modes({"opsd_teacher_source": "frozen"})


def test_validate_opsd_modes_invalid_loss_mode():
    with pytest.raises(ValueError, match="opsd_loss_mode"):
        validate_opsd_modes({"opsd_loss_mode": "kl"})


def test_should_use_initial_frozen_teacher():
    cfg = {
        "use_opsd_jsd_in_advantage": True,
        "opsd_teacher_source": "initial_frozen",
    }
    assert should_use_initial_frozen_teacher(cfg) is True

    cfg = {
        "use_opsd_jsd_in_advantage": False,
        "opsd_teacher_source": "initial_frozen",
    }
    assert should_use_initial_frozen_teacher(cfg) is False

    cfg = {
        "use_opsd_jsd_in_advantage": True,
        "opsd_teacher_source": "online",
    }
    assert should_use_initial_frozen_teacher(cfg) is False


def test_compute_sampled_kl_penalty_matches_definition():
    student = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]], dtype=torch.float32)
    teacher = torch.tensor([[-0.1, -0.4, 0.2], [0.2, 0.1, -0.2]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32)
    expected = (student - teacher) * mask
    actual = compute_sampled_kl_penalty(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
    )
    assert torch.allclose(actual, expected)
