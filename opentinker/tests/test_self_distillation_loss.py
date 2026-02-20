"""Unit tests for self-distillation (OPSD) loss functions."""

import sys
import os

# Add the repo root to path so verl can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "verl"))

import torch
import pytest


# ===== JSD Loss Tests =====


def test_jsd_zero_when_equal():
    """JSD should be 0 when teacher == student."""
    from opentinker.self_distillation.loss import compute_jsd_loss

    log_probs = torch.log(torch.tensor([[0.3, 0.5, 0.2], [0.4, 0.4, 0.2]]))
    mask = torch.ones(2, 3)
    loss = compute_jsd_loss(
        student_log_probs=log_probs,
        teacher_log_probs=log_probs.clone(),
        response_mask=mask,
        loss_agg_mode="token-mean",
        beta=0.5,
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"Expected ~0, got {loss.item()}"


def test_jsd_non_negative():
    """JSD should always be >= 0."""
    from opentinker.self_distillation.loss import compute_jsd_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2]]))
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    mask = torch.ones(1, 3)
    loss = compute_jsd_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        loss_agg_mode="token-mean",
        beta=0.5,
    )
    assert loss.item() >= 0, f"JSD should be non-negative, got {loss.item()}"


def test_jsd_bounded():
    """JSD should be bounded above (JSD <= log(2) for beta=0.5)."""
    from opentinker.self_distillation.loss import compute_jsd_loss
    import math

    # Extreme case: very different distributions
    student = torch.log(torch.tensor([[0.99, 0.01]]))
    teacher = torch.log(torch.tensor([[0.01, 0.99]]))
    mask = torch.ones(1, 2)
    loss = compute_jsd_loss(
        student_log_probs=student,
        teacher_log_probs=teacher,
        response_mask=mask,
        loss_agg_mode="token-mean",
        beta=0.5,
    )
    assert loss.item() <= math.log(2) + 1e-4, (
        f"JSD should be <= ln(2) for beta=0.5, got {loss.item()}"
    )


def test_jsd_symmetric_when_beta_half():
    """JSD with beta=0.5 should be symmetric: JSD(p||q) == JSD(q||p)."""
    from opentinker.self_distillation.loss import compute_jsd_loss

    p = torch.log(torch.tensor([[0.3, 0.5, 0.2]]))
    q = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    mask = torch.ones(1, 3)

    loss_pq = compute_jsd_loss(
        student_log_probs=p, teacher_log_probs=q, response_mask=mask,
        loss_agg_mode="token-mean", beta=0.5,
    )
    loss_qp = compute_jsd_loss(
        student_log_probs=q, teacher_log_probs=p, response_mask=mask,
        loss_agg_mode="token-mean", beta=0.5,
    )
    assert torch.isclose(loss_pq, loss_qp, atol=1e-5), (
        f"JSD(p||q)={loss_pq.item()} != JSD(q||p)={loss_qp.item()}"
    )


def test_jsd_gradient_flows_through_student():
    """Gradient should flow through student but not teacher."""
    from opentinker.self_distillation.loss import compute_jsd_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2]])).requires_grad_(True)
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2]])).requires_grad_(True)
    mask = torch.ones(1, 3)

    loss = compute_jsd_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean",
    )
    loss.backward()

    assert student.grad is not None, "Student should have gradients"
    assert teacher.grad is None or torch.all(teacher.grad == 0), "Teacher should have no gradients"


def test_jsd_mask_excludes_padding():
    """Masked tokens should not contribute to JSD loss."""
    from opentinker.self_distillation.loss import compute_jsd_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2, 0.1]]))
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2, 0.9]]))

    mask_2 = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    loss_2 = compute_jsd_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask_2, loss_agg_mode="token-mean",
    )

    mask_4 = torch.ones(1, 4)
    loss_4 = compute_jsd_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask_4, loss_agg_mode="token-mean",
    )

    assert not torch.isclose(loss_2, loss_4, atol=1e-6), (
        f"Masking should change the loss: masked={loss_2.item()}, unmasked={loss_4.item()}"
    )


def test_jsd_returns_metrics():
    """JSD loss should return metrics when requested."""
    from opentinker.self_distillation.loss import compute_jsd_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2]]))
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    mask = torch.ones(1, 3)

    loss, metrics = compute_jsd_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean", return_metrics=True,
    )

    expected_keys = {
        "jsd_per_token", "kl_teacher_m_per_token", "kl_student_m_per_token",
        "student_log_prob_mean", "teacher_log_prob_mean",
        "student_entropy", "teacher_entropy", "advantage_mean",
    }
    assert set(metrics.keys()) == expected_keys, f"Missing keys: {expected_keys - set(metrics.keys())}"
    for k, v in metrics.items():
        assert torch.isfinite(v), f"Metric {k} is not finite: {v}"


def test_jsd_beta_parameter():
    """Different beta values should give different losses for asymmetric distributions."""
    from opentinker.self_distillation.loss import compute_jsd_loss

    # Use clearly asymmetric distributions (different entropies)
    student = torch.log(torch.tensor([[0.9, 0.05, 0.05]]))  # peaked
    teacher = torch.log(torch.tensor([[0.33, 0.34, 0.33]]))  # uniform-ish
    mask = torch.ones(1, 3)

    loss_01 = compute_jsd_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean", beta=0.1,
    )
    loss_09 = compute_jsd_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean", beta=0.9,
    )

    assert not torch.isclose(loss_01, loss_09, atol=1e-4), (
        f"Extreme beta values should give different losses: "
        f"beta=0.1 -> {loss_01.item()}, beta=0.9 -> {loss_09.item()}"
    )


# ===== Sampled Token Loss Tests =====


def test_sampled_token_zero_when_equal():
    """Sampled token loss should be 0 when teacher == student (A_n = 0)."""
    from opentinker.self_distillation.loss import compute_sampled_token_loss

    log_probs = torch.log(torch.tensor([[0.3, 0.5, 0.2]])).requires_grad_(True)
    mask = torch.ones(1, 3)
    loss = compute_sampled_token_loss(
        student_log_probs=log_probs,
        teacher_log_probs=log_probs.detach().clone(),
        response_mask=mask,
        loss_agg_mode="token-mean",
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5), f"Expected ~0, got {loss.item()}"


def test_sampled_token_gradient_flows_through_student():
    """Gradient should flow through student log probs."""
    from opentinker.self_distillation.loss import compute_sampled_token_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2]])).requires_grad_(True)
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2]])).requires_grad_(True)
    mask = torch.ones(1, 3)

    loss = compute_sampled_token_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean",
    )
    loss.backward()

    assert student.grad is not None, "Student should have gradients"
    assert teacher.grad is None or torch.all(teacher.grad == 0), "Teacher should have no gradients"


def test_sampled_token_positive_advantage_encourages_action():
    """When teacher gives higher probability (positive advantage), loss should push student higher."""
    from opentinker.self_distillation.loss import compute_sampled_token_loss

    # Teacher likes this token more than student
    student = torch.tensor([[-2.0]]).requires_grad_(True)
    teacher = torch.tensor([[-0.5]])  # higher prob
    mask = torch.ones(1, 1)

    loss = compute_sampled_token_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean",
    )
    loss.backward()

    # Gradient should be negative (pushing student log prob higher = lower loss)
    assert student.grad.item() < 0, (
        f"Positive advantage should produce negative gradient, got {student.grad.item()}"
    )


def test_sampled_token_clip_advantage():
    """Clipping should limit the advantage magnitude."""
    from opentinker.self_distillation.loss import compute_sampled_token_loss

    student = torch.tensor([[-5.0, -0.1]]).requires_grad_(True)
    teacher = torch.tensor([[-0.1, -5.0]])  # Very different
    mask = torch.ones(1, 2)

    loss_clipped = compute_sampled_token_loss(
        student_log_probs=student.detach().requires_grad_(True),
        teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean",
        clip_advantage=1.0,
    )

    loss_unclipped = compute_sampled_token_loss(
        student_log_probs=student.detach().requires_grad_(True),
        teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean",
        clip_advantage=0.0,
    )

    # Clipped loss should have smaller magnitude
    assert abs(loss_clipped.item()) <= abs(loss_unclipped.item()) + 1e-6, (
        f"Clipped loss {loss_clipped.item()} should be <= unclipped {loss_unclipped.item()}"
    )


def test_sampled_token_returns_metrics():
    """Sampled token loss should return metrics when requested."""
    from opentinker.self_distillation.loss import compute_sampled_token_loss

    student = torch.log(torch.tensor([[0.3, 0.5, 0.2]])).requires_grad_(True)
    teacher = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    mask = torch.ones(1, 3)

    loss, metrics = compute_sampled_token_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean", return_metrics=True,
    )

    expected_keys = {
        "advantage_mean", "advantage_std", "advantage_positive_frac",
        "student_log_prob_mean", "teacher_log_prob_mean",
        "student_entropy", "teacher_entropy",
    }
    assert set(metrics.keys()) == expected_keys, f"Missing keys: {expected_keys - set(metrics.keys())}"


# ===== Teacher Quality Filtering Tests =====


def test_quality_mask_positive_advantage_only():
    """positive_advantage_only should mask tokens where student > teacher."""
    from opentinker.self_distillation.loss import compute_teacher_quality_mask

    # Token 0: teacher > student (keep), Token 1: student > teacher (filter)
    student = torch.tensor([[-2.0, -0.5, -1.0]])
    teacher = torch.tensor([[-0.5, -2.0, -1.0]])
    mask = torch.ones(1, 3)

    filtered, metrics = compute_teacher_quality_mask(
        student, teacher, mask, positive_advantage_only=True,
    )
    # Token 0: adv = -0.5 - (-2.0) = 1.5 > 0 → keep
    # Token 1: adv = -2.0 - (-0.5) = -1.5 < 0 → filter
    # Token 2: adv = 0 → filter (not strictly > 0)
    assert filtered[0, 0].item() == 1.0
    assert filtered[0, 1].item() == 0.0
    assert filtered[0, 2].item() == 0.0
    assert "filter/positive_adv_ratio" in metrics


def test_quality_mask_teacher_min_log_prob():
    """teacher_min_log_prob should mask tokens where teacher is too uncertain."""
    from opentinker.self_distillation.loss import compute_teacher_quality_mask

    student = torch.tensor([[-1.0, -1.0, -1.0]])
    teacher = torch.tensor([[-0.5, -3.0, -6.0]])
    mask = torch.ones(1, 3)

    filtered, metrics = compute_teacher_quality_mask(
        student, teacher, mask, teacher_min_log_prob=-2.0,
    )
    # Token 0: -0.5 >= -2.0 → keep
    # Token 1: -3.0 < -2.0 → filter
    # Token 2: -6.0 < -2.0 → filter
    assert filtered[0, 0].item() == 1.0
    assert filtered[0, 1].item() == 0.0
    assert filtered[0, 2].item() == 0.0
    assert "filter/teacher_conf_ratio" in metrics


def test_quality_mask_sequence_ppl():
    """sequence_ppl_max should mask entire sequences with high teacher perplexity."""
    from opentinker.self_distillation.loss import compute_teacher_quality_mask

    # Batch of 2 sequences
    # Seq 0: teacher log probs ~ -1.0 → ppl ~ e^1 ≈ 2.7
    # Seq 1: teacher log probs ~ -5.0 → ppl ~ e^5 ≈ 148.4
    student = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    teacher = torch.tensor([[-1.0, -1.0], [-5.0, -5.0]])
    mask = torch.ones(2, 2)

    filtered, metrics = compute_teacher_quality_mask(
        student, teacher, mask, sequence_ppl_max=10.0,
    )
    # Seq 0: ppl ≈ 2.7 <= 10.0 → keep
    assert filtered[0, 0].item() == 1.0
    assert filtered[0, 1].item() == 1.0
    # Seq 1: ppl ≈ 148.4 > 10.0 → filter entire sequence
    assert filtered[1, 0].item() == 0.0
    assert filtered[1, 1].item() == 0.0
    assert "filter/seq_keep_ratio" in metrics
    assert metrics["filter/seq_keep_ratio"].item() == 0.5


def test_quality_mask_combined_filters():
    """Multiple filters should compose (intersection)."""
    from opentinker.self_distillation.loss import compute_teacher_quality_mask

    student = torch.tensor([[-2.0, -0.5]])
    teacher = torch.tensor([[-0.5, -2.0]])
    mask = torch.ones(1, 2)

    filtered, metrics = compute_teacher_quality_mask(
        student, teacher, mask,
        positive_advantage_only=True,
        teacher_min_log_prob=-1.0,
    )
    # Token 0: adv > 0 (keep) AND teacher=-0.5 >= -1.0 (keep) → keep
    # Token 1: adv < 0 (filter) → filter regardless of confidence
    assert filtered[0, 0].item() == 1.0
    assert filtered[0, 1].item() == 0.0


def test_quality_mask_no_filters():
    """No filters should return original mask unchanged."""
    from opentinker.self_distillation.loss import compute_teacher_quality_mask

    student = torch.tensor([[-1.0, -2.0]])
    teacher = torch.tensor([[-2.0, -1.0]])
    mask = torch.tensor([[1.0, 0.0]])

    filtered, metrics = compute_teacher_quality_mask(student, teacher, mask)
    assert torch.equal(filtered, mask)
    assert metrics["filter/token_keep_ratio"].item() == 1.0


def test_sampled_token_with_positive_advantage_filter():
    """Sampled token loss with positive_advantage_only should only use positive-advantage tokens."""
    from opentinker.self_distillation.loss import compute_sampled_token_loss

    student = torch.tensor([[-2.0, -0.5]]).requires_grad_(True)
    teacher = torch.tensor([[-0.5, -2.0]])  # token 0: adv>0, token 1: adv<0
    mask = torch.ones(1, 2)

    loss, metrics = compute_sampled_token_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean",
        return_metrics=True, positive_advantage_only=True,
    )

    assert "filter/positive_adv_ratio" in metrics
    assert "filter/token_keep_ratio" in metrics
    # Loss should be positive since only positive-advantage tokens are used
    # A_n > 0, log p_S < 0, so -A_n * log p_S > 0
    assert loss.item() > 0, f"Loss with positive-advantage filter should be > 0, got {loss.item()}"


def test_jsd_with_sequence_ppl_filter():
    """JSD loss with sequence_ppl_max should skip high-perplexity sequences."""
    from opentinker.self_distillation.loss import compute_jsd_loss

    # 2 sequences: one low ppl, one high ppl
    student = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    teacher = torch.tensor([[-1.0, -1.0], [-8.0, -8.0]])
    mask = torch.ones(2, 2)

    loss_filtered, metrics = compute_jsd_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean",
        return_metrics=True, sequence_ppl_max=10.0,
    )

    loss_unfiltered = compute_jsd_loss(
        student_log_probs=student, teacher_log_probs=teacher,
        response_mask=mask, loss_agg_mode="token-mean",
    )

    # They should differ since seq 1 is filtered
    assert not torch.isclose(loss_filtered, loss_unfiltered, atol=1e-6), (
        f"Filtered={loss_filtered.item()} should differ from unfiltered={loss_unfiltered.item()}"
    )
    assert "filter/seq_keep_ratio" in metrics


# ===== Teacher Utils Tests =====


def test_build_teacher_messages():
    """Solution should be appended to the last user message."""
    from opentinker.self_distillation.teacher_utils import _build_teacher_messages

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    teacher_msgs = _build_teacher_messages(messages, "4")

    assert len(teacher_msgs) == 2
    assert "Reference solution: 4" in teacher_msgs[1]["content"]
    assert teacher_msgs[1]["content"].startswith("What is 2+2?")
    # Original should be unchanged
    assert "Reference" not in messages[1]["content"]


def test_build_teacher_messages_custom_template():
    """Custom solution template should be used."""
    from opentinker.self_distillation.teacher_utils import _build_teacher_messages

    messages = [
        {"role": "user", "content": "Solve: x+1=3"},
    ]

    teacher_msgs = _build_teacher_messages(
        messages, "x=2",
        solution_template="\n[ANSWER: {solution}]",
    )

    assert "[ANSWER: x=2]" in teacher_msgs[0]["content"]


def test_build_teacher_messages_preserves_system():
    """System message should be preserved unchanged."""
    from opentinker.self_distillation.teacher_utils import _build_teacher_messages

    messages = [
        {"role": "system", "content": "System prompt here."},
        {"role": "user", "content": "Question?"},
    ]

    teacher_msgs = _build_teacher_messages(messages, "answer")

    assert teacher_msgs[0]["content"] == "System prompt here."
    assert "answer" in teacher_msgs[1]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
