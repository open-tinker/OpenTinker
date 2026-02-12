"""
On-policy distillation loss functions.

Implements forward KL divergence loss for knowledge distillation,
where a student model learns to match a teacher model's output distribution.
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_forward_kl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Compute forward KL divergence loss: KL(teacher || student).

    Forward KL = Σ p_teacher * (log p_teacher - log p_student)

    This loss encourages the student to cover all modes of the teacher's
    distribution (mean-seeking behavior).

    Args:
        student_log_probs: (batch, response_len) — student's log p(token | prefix)
        teacher_log_probs: (batch, response_len) — teacher's log p(token | prefix)
        response_mask: (batch, response_len) — 1 for valid response tokens, 0 for padding
        loss_agg_mode: aggregation mode, e.g. "token-mean" or "seq-mean-token-sum"
        return_metrics: if True, return additional metrics dict

    Returns:
        If return_metrics=False: Scalar loss tensor.
        If return_metrics=True: (loss, metrics_dict) tuple.
    """
    # Detach teacher to ensure no gradient flows through teacher
    teacher_log_probs = teacher_log_probs.detach()

    # p_teacher * (log p_teacher - log p_student)
    teacher_probs = teacher_log_probs.exp()
    kl = teacher_probs * (teacher_log_probs - student_log_probs)

    # Clamp to avoid numerical issues (KL should be non-negative)
    kl = kl.clamp(min=0.0)

    loss = agg_loss(loss_mat=kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    if return_metrics:
        # Compute additional metrics
        valid_tokens = response_mask.sum()
        metrics = {
            # Per-token KL (mean over valid tokens)
            "kl_per_token": (kl * response_mask).sum() / valid_tokens.clamp(min=1),
            # Student log prob stats (on valid tokens)
            "student_log_prob_mean": (student_log_probs * response_mask).sum() / valid_tokens.clamp(min=1),
            # Teacher log prob stats
            "teacher_log_prob_mean": (teacher_log_probs * response_mask).sum() / valid_tokens.clamp(min=1),
            # Student entropy estimate: -E[log p] ≈ -mean(log_prob)
            "student_entropy": -(student_log_probs * response_mask).sum() / valid_tokens.clamp(min=1),
            # Teacher entropy estimate
            "teacher_entropy": -(teacher_log_probs * response_mask).sum() / valid_tokens.clamp(min=1),
            # Log prob difference (teacher - student)
            "log_prob_diff": ((teacher_log_probs - student_log_probs) * response_mask).sum() / valid_tokens.clamp(min=1),
        }
        return loss, metrics

    return loss


def compute_reverse_kl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Compute reverse KL divergence loss: KL(student || teacher).

    Reverse KL = Σ p_student * (log p_student - log p_teacher)

    This loss encourages the student to be mode-seeking (concentrate on
    high-probability regions of the teacher).

    Args:
        student_log_probs: (batch, response_len) — student's log p(token | prefix)
        teacher_log_probs: (batch, response_len) — teacher's log p(token | prefix)
        response_mask: (batch, response_len) — 1 for valid response tokens, 0 for padding
        loss_agg_mode: aggregation mode
        return_metrics: if True, return additional metrics dict

    Returns:
        If return_metrics=False: Scalar loss tensor.
        If return_metrics=True: (loss, metrics_dict) tuple.
    """
    teacher_log_probs = teacher_log_probs.detach()

    student_probs = student_log_probs.exp()
    kl = student_probs * (student_log_probs - teacher_log_probs)
    kl = kl.clamp(min=0.0)

    loss = agg_loss(loss_mat=kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    if return_metrics:
        valid_tokens = response_mask.sum()
        metrics = {
            "kl_per_token": (kl * response_mask).sum() / valid_tokens.clamp(min=1),
            "student_log_prob_mean": (student_log_probs * response_mask).sum() / valid_tokens.clamp(min=1),
            "teacher_log_prob_mean": (teacher_log_probs * response_mask).sum() / valid_tokens.clamp(min=1),
            "student_entropy": -(student_log_probs * response_mask).sum() / valid_tokens.clamp(min=1),
            "teacher_entropy": -(teacher_log_probs * response_mask).sum() / valid_tokens.clamp(min=1),
            "log_prob_diff": ((teacher_log_probs - student_log_probs) * response_mask).sum() / valid_tokens.clamp(min=1),
        }
        return loss, metrics

    return loss
