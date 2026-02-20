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

Teacher quality filtering:
- positive_advantage_only: Only distill from tokens where teacher > student
- teacher_min_prob: Skip tokens where teacher probability is too low
- sequence_ppl_threshold: Skip entire sequences where teacher perplexity is too high
"""

from typing import Optional

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_teacher_quality_mask(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    positive_advantage_only: bool = False,
    teacher_min_log_prob: Optional[float] = None,
    sequence_ppl_max: Optional[float] = None,
) -> tuple[torch.Tensor, dict]:
    """Compute a quality mask that filters teacher signals by reliability.

    Combines multiple filtering strategies to produce a refined mask. All
    filters are applied on top of the original response_mask so padding
    tokens remain masked.

    Args:
        student_log_probs: (batch, response_len) student log p(token)
        teacher_log_probs: (batch, response_len) teacher log p(token)
        response_mask: (batch, response_len) original valid-token mask
        positive_advantage_only: If True, only keep tokens where
            log p_T > log p_S (teacher is more confident).
        teacher_min_log_prob: If set, mask out tokens where teacher
            log prob < this threshold (teacher is too uncertain).
        sequence_ppl_max: If set, mask out entire sequences where the
            teacher's perplexity exceeds this value.

    Returns:
        (filtered_mask, filter_metrics) where filtered_mask has the same
        shape as response_mask and filter_metrics contains diagnostics.
    """
    filtered_mask = response_mask.clone()
    metrics = {}
    valid_tokens_before = response_mask.sum().clamp(min=1)

    # --- 1. Positive-advantage filter (token-level) ---
    if positive_advantage_only:
        advantage = teacher_log_probs.detach() - student_log_probs.detach()
        pos_mask = (advantage > 0).float()
        filtered_mask = filtered_mask * pos_mask
        metrics["filter/positive_adv_ratio"] = (
            (pos_mask * response_mask).sum() / valid_tokens_before
        )

    # --- 2. Teacher minimum probability filter (token-level) ---
    if teacher_min_log_prob is not None:
        conf_mask = (teacher_log_probs.detach() >= teacher_min_log_prob).float()
        filtered_mask = filtered_mask * conf_mask
        metrics["filter/teacher_conf_ratio"] = (
            (conf_mask * response_mask).sum() / valid_tokens_before
        )

    # --- 3. Sequence-level teacher perplexity filter ---
    if sequence_ppl_max is not None:
        # Per-sequence perplexity: exp( -1/N * sum log p_T )
        seq_lengths = response_mask.sum(dim=-1).clamp(min=1)  # (batch,)
        neg_mean_log_prob = -(teacher_log_probs.detach() * response_mask).sum(dim=-1) / seq_lengths
        seq_ppl = neg_mean_log_prob.exp()  # (batch,)
        seq_keep = (seq_ppl <= sequence_ppl_max).float()  # (batch,)
        filtered_mask = filtered_mask * seq_keep.unsqueeze(-1)
        metrics["filter/seq_ppl_mean"] = seq_ppl.mean()
        metrics["filter/seq_ppl_max"] = seq_ppl.max()
        metrics["filter/seq_keep_ratio"] = seq_keep.mean()

    # --- Overall filter stats ---
    valid_tokens_after = filtered_mask.sum().clamp(min=1)
    metrics["filter/token_keep_ratio"] = valid_tokens_after / valid_tokens_before

    return filtered_mask, metrics


def compute_jsd_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    beta: float = 0.5,
    return_metrics: bool = False,
    positive_advantage_only: bool = False,
    teacher_min_log_prob: Optional[float] = None,
    sequence_ppl_max: Optional[float] = None,
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
        positive_advantage_only: only distill tokens where teacher > student
        teacher_min_log_prob: skip tokens where teacher log prob < threshold
        sequence_ppl_max: skip sequences where teacher perplexity > threshold

    Returns:
        loss tensor, optionally (loss, metrics_dict) tuple
    """
    # Detach teacher: gradient flows only through student
    teacher_log_probs = teacher_log_probs.detach()

    # Apply teacher quality filtering
    filter_metrics = {}
    if positive_advantage_only or teacher_min_log_prob is not None or sequence_ppl_max is not None:
        response_mask, filter_metrics = compute_teacher_quality_mask(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            positive_advantage_only=positive_advantage_only,
            teacher_min_log_prob=teacher_min_log_prob,
            sequence_ppl_max=sequence_ppl_max,
        )

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
            **filter_metrics,
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
    positive_advantage_only: bool = False,
    teacher_min_log_prob: Optional[float] = None,
    sequence_ppl_max: Optional[float] = None,
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
        positive_advantage_only: only distill tokens where teacher > student
        teacher_min_log_prob: skip tokens where teacher log prob < threshold
        sequence_ppl_max: skip sequences where teacher perplexity > threshold

    Returns:
        loss tensor, optionally (loss, metrics_dict) tuple
    """
    # Detach teacher
    teacher_log_probs = teacher_log_probs.detach()

    # Apply teacher quality filtering
    filter_metrics = {}
    if positive_advantage_only or teacher_min_log_prob is not None or sequence_ppl_max is not None:
        response_mask, filter_metrics = compute_teacher_quality_mask(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            positive_advantage_only=positive_advantage_only,
            teacher_min_log_prob=teacher_min_log_prob,
            sequence_ppl_max=sequence_ppl_max,
        )

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
            **filter_metrics,
        }
        return loss, metrics

    return loss
