"""Unit tests for distillation loss functions."""

import sys
import os

# Add the repo root to path so verl can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "verl"))

import torch
import pytest


def test_forward_kl_zero_when_equal():
    """Forward KL should be 0 when teacher == student."""
    from opentinker.distillation.distillation_loss import compute_forward_kl_loss

    log_probs = torch.log(torch.tensor([[0.3, 0.5, 0.2], [0.4, 0.4, 0.2]]))
    mask = torch.ones(2, 3)
    loss = compute_forward_kl_loss(
        student_log_probs=log_probs,
        teacher_log_probs=log_probs.clone(),
        response_mask=mask,
        loss_agg_mode="token-mean",
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"Expected ~0, got {loss.item()}"


def test_forward_kl_non_negative():
    """Forward KL should always be >= 0."""
    from opentinker.distillation.distillation_loss import compute_forward_kl_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2]]))
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    mask = torch.ones(1, 3)
    loss = compute_forward_kl_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        loss_agg_mode="token-mean",
    )
    assert loss.item() >= 0, f"KL should be non-negative, got {loss.item()}"


def test_forward_kl_gradient_flows_through_student():
    """Gradient should flow through student but not teacher."""
    from opentinker.distillation.distillation_loss import compute_forward_kl_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2]])).requires_grad_(True)
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2]])).requires_grad_(True)
    mask = torch.ones(1, 3)

    loss = compute_forward_kl_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        loss_agg_mode="token-mean",
    )
    loss.backward()

    assert student.grad is not None, "Student should have gradients"
    assert teacher.grad is None or torch.all(teacher.grad == 0), "Teacher should have no gradients (detached)"


def test_forward_kl_mask_excludes_padding():
    """Masked (padding) tokens should not contribute to loss."""
    from opentinker.distillation.distillation_loss import compute_forward_kl_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2, 0.1]]))
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2, 0.9]]))

    # Only first 2 tokens are valid
    mask_2 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss_2 = compute_forward_kl_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask_2,
        loss_agg_mode="token-mean",
    )

    # All 4 tokens are valid
    mask_4 = torch.ones(1, 4)
    loss_4 = compute_forward_kl_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask_4,
        loss_agg_mode="token-mean",
    )

    # Losses should differ because the last 2 tokens have different distributions
    assert not torch.isclose(loss_2, loss_4, atol=1e-6), (
        f"Masking should change the loss: masked={loss_2.item()}, unmasked={loss_4.item()}"
    )


def test_reverse_kl_zero_when_equal():
    """Reverse KL should be 0 when teacher == student."""
    from opentinker.distillation.distillation_loss import compute_reverse_kl_loss

    log_probs = torch.log(torch.tensor([[0.3, 0.5, 0.2]]))
    mask = torch.ones(1, 3)
    loss = compute_reverse_kl_loss(
        student_log_probs=log_probs,
        teacher_log_probs=log_probs.clone(),
        response_mask=mask,
        loss_agg_mode="token-mean",
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"Expected ~0, got {loss.item()}"


def test_reverse_kl_non_negative():
    """Reverse KL should always be >= 0."""
    from opentinker.distillation.distillation_loss import compute_reverse_kl_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2]]))
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    mask = torch.ones(1, 3)
    loss = compute_reverse_kl_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        loss_agg_mode="token-mean",
    )
    assert loss.item() >= 0, f"KL should be non-negative, got {loss.item()}"


def test_loss_agg_mode_seq_mean():
    """Test seq-mean-token-sum aggregation mode."""
    from opentinker.distillation.distillation_loss import compute_forward_kl_loss

    # Two different sequences
    student = torch.log(torch.tensor([[0.3, 0.5], [0.4, 0.4]]))
    teacher = torch.log(torch.tensor([[0.5, 0.3], [0.6, 0.2]]))
    mask = torch.ones(2, 2)

    loss = compute_forward_kl_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        loss_agg_mode="seq-mean-token-sum",
    )
    assert loss.item() >= 0, f"KL should be non-negative, got {loss.item()}"
    assert torch.isfinite(loss), "Loss should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
