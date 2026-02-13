"""On-Policy Self-Distillation loss functions.

Implements the loss functions from Self-Distilled Reasoner (arXiv:2601.18734):

1. Generalized Jensen-Shannon Divergence (JSD):
   JSD_beta(p_T || p_S) = beta * KL(p_T || m) + (1-beta) * KL(p_S || m)
   where m = beta * p_T + (1-beta) * p_S

2. Sampled-token advantage-weighted loss:
   L = -(1/|y|) * sum_n A_n * log p_S(y_n | x, y_<n)
   where A_n = log p_T(y_n | x, y*, y_<n) - log p_S(y_n | x, y_<n)

Both losses:
- Use the SAME model with different conditioning (student vs teacher)
- Gradient flows only through the student (teacher is detached)
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_jsd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    beta: float = 0.5,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Compute Generalized Jensen-Shannon Divergence loss.

    JSD_beta(p_T || p_S) = beta * KL(p_T || m) + (1-beta) * KL(p_S || m)
    where m = beta * p_T + (1-beta) * p_S

    This is a symmetric, bounded divergence (0 <= JSD <= log 2).
    The gradient flows only through the student distribution.

    Args:
        student_log_probs: (batch, response_len) student log p(token | prefix)
        teacher_log_probs: (batch, response_len) teacher log p(token | prefix, solution)
        response_mask: (batch, response_len) 1 for valid tokens, 0 for padding
        loss_agg_mode: aggregation mode, e.g. "token-mean"
        beta: interpolation parameter (0.5 = symmetric JSD)
        return_metrics: if True, return additional metrics dict

    Returns:
        loss tensor, optionally (loss, metrics_dict) tuple
    """
    # Detach teacher: gradient flows only through student
    teacher_log_probs = teacher_log_probs.detach()

    # Convert to probabilities
    p_teacher = teacher_log_probs.exp()
    p_student = student_log_probs.exp()

    # Mixture distribution: m = beta * p_T + (1-beta) * p_S
    m = beta * p_teacher + (1 - beta) * p_student
    log_m = m.log()

    # JSD = beta * KL(p_T || m) + (1-beta) * KL(p_S || m)
    # KL(p || q) = sum p * (log p - log q)
    kl_teacher_m = p_teacher * (teacher_log_probs - log_m)
    kl_student_m = p_student * (student_log_probs - log_m)

    # Clamp for numerical stability
    kl_teacher_m = kl_teacher_m.clamp(min=0.0)
    kl_student_m = kl_student_m.clamp(min=0.0)

    jsd = beta * kl_teacher_m + (1 - beta) * kl_student_m

    loss = agg_loss(loss_mat=jsd, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    if return_metrics:
        valid_tokens = response_mask.sum().clamp(min=1)
        metrics = {
            "jsd_per_token": (jsd * response_mask).sum() / valid_tokens,
            "kl_teacher_m_per_token": (kl_teacher_m * response_mask).sum() / valid_tokens,
            "kl_student_m_per_token": (kl_student_m * response_mask).sum() / valid_tokens,
            "student_log_prob_mean": (student_log_probs * response_mask).sum() / valid_tokens,
            "teacher_log_prob_mean": (teacher_log_probs * response_mask).sum() / valid_tokens,
            "student_entropy": -(student_log_probs * response_mask).sum() / valid_tokens,
            "teacher_entropy": -(teacher_log_probs * response_mask).sum() / valid_tokens,
            "advantage_mean": ((teacher_log_probs - student_log_probs) * response_mask).sum() / valid_tokens,
        }
        return loss, metrics

    return loss


def compute_sampled_token_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    clip_advantage: float = 0.0,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Compute sampled-token advantage-weighted policy gradient loss.

    L = -(1/|y|) * sum_n A_n * log p_S(y_n | x, y_<n)

    where the advantage is:
        A_n = log p_T(y_n | x, y*, y_<n) - log p_S(y_n | x, y_<n)

    This is equivalent to the REINFORCE-style gradient of the JSD objective
    evaluated on sampled tokens from the student.

    Args:
        student_log_probs: (batch, response_len) student log probs
        teacher_log_probs: (batch, response_len) teacher log probs (detached)
        response_mask: (batch, response_len) mask
        loss_agg_mode: aggregation mode
        clip_advantage: if > 0, clip advantages to [-clip, clip] for stability
        return_metrics: if True, return metrics dict

    Returns:
        loss tensor, optionally (loss, metrics_dict) tuple
    """
    # Detach teacher
    teacher_log_probs = teacher_log_probs.detach()

    # Compute per-token advantage: A_n = log p_T - log p_S
    # Detach both for advantage computation (advantage is a fixed signal)
    advantage = teacher_log_probs - student_log_probs.detach()

    if clip_advantage > 0:
        advantage = advantage.clamp(-clip_advantage, clip_advantage)

    # Policy gradient loss: -A_n * log p_S(y_n)
    # student_log_probs carries gradients here
    pg_loss = -advantage * student_log_probs

    loss = agg_loss(loss_mat=pg_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    if return_metrics:
        valid_tokens = response_mask.sum().clamp(min=1)
        metrics = {
            "advantage_mean": (advantage * response_mask).sum() / valid_tokens,
            "advantage_std": ((advantage * response_mask).pow(2).sum() / valid_tokens - ((advantage * response_mask).sum() / valid_tokens).pow(2)).clamp(min=0).sqrt(),
            "advantage_positive_frac": ((advantage > 0).float() * response_mask).sum() / valid_tokens,
            "student_log_prob_mean": (student_log_probs * response_mask).sum() / valid_tokens,
            "teacher_log_prob_mean": (teacher_log_probs * response_mask).sum() / valid_tokens,
            "student_entropy": -(student_log_probs * response_mask).sum() / valid_tokens,
            "teacher_entropy": -(teacher_log_probs * response_mask).sum() / valid_tokens,
        }
        return loss, metrics

    return loss
