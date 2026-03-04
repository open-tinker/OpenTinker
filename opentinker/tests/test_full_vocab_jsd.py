# Copyright 2026 OpenTinker
#
# Licensed under the Apache License, Version 2.0.
"""Unit tests for full-vocab JSD utilities."""

import math

import torch

from opentinker.backend_patch.verl.trainer.ppo.full_vocab_jsd import (
    backward_full_vocab_jsd_on_logits,
    backward_topk_tail_reverse_kl_on_logits,
    compute_token_jsd_from_logits,
    compute_token_topk_tail_reverse_kl_from_logits,
    compute_teacher_topk_log_probs_from_logits,
)


def test_jsd_zero_for_identical_logits():
    logits = torch.randn(2, 3, 17)
    jsd = compute_token_jsd_from_logits(logits, logits, beta=0.5)
    assert jsd.shape == (2, 3)
    assert torch.allclose(jsd, torch.zeros_like(jsd), atol=1e-6)


def test_jsd_non_negative():
    student = torch.randn(2, 4, 11)
    teacher = torch.randn(2, 4, 11)
    jsd = compute_token_jsd_from_logits(student, teacher, beta=0.5)
    assert torch.all(jsd >= -1e-7)


def test_jsd_symmetric_at_beta_half():
    student = torch.randn(1, 5, 13)
    teacher = torch.randn(1, 5, 13)
    jsd_st = compute_token_jsd_from_logits(student, teacher, beta=0.5)
    jsd_ts = compute_token_jsd_from_logits(teacher, student, beta=0.5)
    assert torch.allclose(jsd_st, jsd_ts, atol=1e-6)


def test_jsd_chunked_matches_full():
    student = torch.randn(2, 3, 37)
    teacher = torch.randn(2, 3, 37)
    jsd_full = compute_token_jsd_from_logits(
        student, teacher, beta=0.5, vocab_chunk_size=4096
    )
    jsd_chunked = compute_token_jsd_from_logits(
        student, teacher, beta=0.5, vocab_chunk_size=7
    )
    assert torch.allclose(jsd_full, jsd_chunked, atol=1e-6)


def test_jsd_token_chunked_matches_full():
    student = torch.randn(2, 5, 37)
    teacher = torch.randn(2, 5, 37)
    jsd_full = compute_token_jsd_from_logits(
        student, teacher, beta=0.5, vocab_chunk_size=4096, token_chunk_size=0
    )
    jsd_token_chunked = compute_token_jsd_from_logits(
        student, teacher, beta=0.5, vocab_chunk_size=4096, token_chunk_size=2
    )
    assert torch.allclose(jsd_full, jsd_token_chunked, atol=1e-6)


def test_jsd_token_and_vocab_chunked_matches_full():
    student = torch.randn(2, 5, 37)
    teacher = torch.randn(2, 5, 37)
    jsd_full = compute_token_jsd_from_logits(
        student, teacher, beta=0.5, vocab_chunk_size=4096, token_chunk_size=0
    )
    jsd_chunked = compute_token_jsd_from_logits(
        student, teacher, beta=0.5, vocab_chunk_size=7, token_chunk_size=2
    )
    assert torch.allclose(jsd_full, jsd_chunked, atol=1e-6)


def _compute_token_jsd_naive(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    s_logp = torch.log_softmax(student_logits, dim=-1)
    t_logp = torch.log_softmax(teacher_logits, dim=-1)
    mix_logp = torch.logaddexp(
        t_logp + math.log(beta),
        s_logp + math.log(1.0 - beta),
    )
    t_prob = torch.exp(t_logp)
    s_prob = torch.exp(s_logp)
    kl_t_m = torch.sum(t_prob * (t_logp - mix_logp), dim=-1)
    kl_s_m = torch.sum(s_prob * (s_logp - mix_logp), dim=-1)
    return beta * kl_t_m + (1.0 - beta) * kl_s_m


def _log1mexp_naive(logx: torch.Tensor) -> torch.Tensor:
    threshold = -0.6931471805599453  # log(0.5)
    logx = torch.clamp(logx, max=0.0)
    return torch.where(
        logx < threshold,
        torch.log1p(-torch.exp(logx)),
        torch.log(-torch.expm1(logx)),
    )


def _compute_token_topk_tail_reverse_kl_naive(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    s_logp = torch.log_softmax(student_logits, dim=-1)
    t_logp = torch.log_softmax(teacher_logits, dim=-1)
    t_topk_logp, t_topk_idx = torch.topk(t_logp, k=topk, dim=-1)
    s_topk_logp = torch.gather(s_logp, dim=-1, index=t_topk_idx)

    s_topk_logsumexp = torch.logsumexp(s_topk_logp, dim=-1)
    t_topk_logsumexp = torch.logsumexp(t_topk_logp, dim=-1)
    s_tail_logp = _log1mexp_naive(s_topk_logsumexp)
    t_tail_logp = _log1mexp_naive(t_topk_logsumexp)

    s_topk_prob = torch.exp(s_topk_logp)
    s_tail_prob = torch.exp(s_tail_logp)
    topk_term = torch.sum(s_topk_prob * (s_topk_logp - t_topk_logp), dim=-1)
    tail_term = torch.where(
        s_tail_prob > 0,
        s_tail_prob * (s_tail_logp - t_tail_logp),
        torch.zeros_like(s_tail_logp),
    )
    return topk_term + tail_term


def _agg_loss_naive(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
) -> torch.Tensor:
    mask = loss_mask.float()
    if loss_agg_mode == "token-mean":
        return (loss_mat * mask).sum() / mask.sum().clamp_min(1e-8)

    if loss_agg_mode == "seq-mean-token-sum":
        token_sum = (loss_mat * mask).sum(dim=-1)
        seq_mask = (mask.sum(dim=-1) > 0).float()
        return (token_sum * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)

    if loss_agg_mode == "seq-mean-token-mean":
        token_sum = (loss_mat * mask).sum(dim=-1)
        token_cnt = mask.sum(dim=-1)
        seq_mean = token_sum / (token_cnt + 1e-8)
        seq_mask = (token_cnt > 0).float()
        return (seq_mean * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)

    if loss_agg_mode == "seq-mean-token-sum-norm":
        return (loss_mat * mask).sum(dim=-1).mean() / float(mask.shape[-1])

    raise ValueError(f"Unsupported loss_agg_mode for test: {loss_agg_mode}")


def test_jsd_chunk_checkpoint_matches_naive_loss_and_grad():
    torch.manual_seed(0)
    beta = 0.37
    student = torch.randn(2, 3, 19, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(2, 3, 19, dtype=torch.float64)

    jsd_chunk = compute_token_jsd_from_logits(
        student, teacher, beta=beta, vocab_chunk_size=7, token_chunk_size=2
    )
    grad_chunk = torch.autograd.grad(jsd_chunk.sum(), student)[0]

    student_naive = student.detach().clone().requires_grad_(True)
    jsd_naive = _compute_token_jsd_naive(student_naive, teacher, beta=beta)
    grad_naive = torch.autograd.grad(jsd_naive.sum(), student_naive)[0]

    assert torch.allclose(
        jsd_chunk, jsd_naive.to(dtype=jsd_chunk.dtype), atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        grad_chunk, grad_naive.to(dtype=grad_chunk.dtype), atol=1e-6, rtol=1e-6
    )


def test_backward_full_vocab_jsd_matches_naive_grad():
    torch.manual_seed(7)
    beta = 0.41
    loss_scale = 0.73
    student = torch.randn(2, 4, 23, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(2, 4, 23, dtype=torch.float64)
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.float64)

    modes = ["token-mean", "seq-mean-token-mean"]
    for mode in modes:
        student_baseline = student.detach().clone().requires_grad_(True)
        jsd_baseline = _compute_token_jsd_naive(student_baseline, teacher, beta=beta)
        loss_baseline = _agg_loss_naive(jsd_baseline, response_mask, mode)
        (loss_baseline * loss_scale).backward()
        grad_baseline = student_baseline.grad.detach().clone()

        student_chunk = student.detach().clone().requires_grad_(True)
        jsd_loss_chunk, effective_chunk = backward_full_vocab_jsd_on_logits(
            student_logits=student_chunk,
            teacher_logits=teacher,
            response_mask=response_mask,
            beta=beta,
            loss_agg_mode=mode,
            loss_scale=loss_scale,
            vocab_chunk_size=7,
            token_chunk_size=2,
        )
        grad_chunk = student_chunk.grad.detach().clone()

        assert effective_chunk == 2
        assert torch.allclose(
            jsd_loss_chunk, loss_baseline.to(dtype=jsd_loss_chunk.dtype), atol=1e-6, rtol=1e-6
        )
        assert torch.allclose(
            grad_chunk, grad_baseline.to(dtype=grad_chunk.dtype), atol=1e-6, rtol=1e-6
        )


def test_topk_tail_reverse_kl_matches_naive():
    torch.manual_seed(13)
    student = torch.randn(2, 4, 17, dtype=torch.float64)
    teacher = torch.randn(2, 4, 17, dtype=torch.float64)
    topk = 5

    teacher_topk_log_probs, teacher_topk_indices = (
        compute_teacher_topk_log_probs_from_logits(
            teacher_logits=teacher, topk=topk, vocab_chunk_size=7
        )
    )
    impl = compute_token_topk_tail_reverse_kl_from_logits(
        student_logits=student,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_topk_indices=teacher_topk_indices,
        vocab_chunk_size=7,
        token_chunk_size=2,
    )
    naive = _compute_token_topk_tail_reverse_kl_naive(student, teacher, topk=topk)
    assert torch.allclose(impl, naive.to(dtype=impl.dtype), atol=1e-6, rtol=1e-6)


def test_backward_topk_tail_reverse_kl_matches_naive_grad():
    torch.manual_seed(23)
    topk = 6
    loss_scale = 0.83
    student = torch.randn(2, 5, 19, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(2, 5, 19, dtype=torch.float64)
    response_mask = torch.tensor([[1, 1, 1, 0, 1], [1, 0, 1, 1, 1]], dtype=torch.float64)

    teacher_topk_log_probs, teacher_topk_indices = (
        compute_teacher_topk_log_probs_from_logits(
            teacher_logits=teacher, topk=topk, vocab_chunk_size=7
        )
    )

    modes = ["token-mean", "seq-mean-token-mean"]
    for mode in modes:
        student_baseline = student.detach().clone().requires_grad_(True)
        rkl_baseline = _compute_token_topk_tail_reverse_kl_naive(
            student_baseline, teacher, topk=topk
        )
        loss_baseline = _agg_loss_naive(rkl_baseline, response_mask, mode)
        (loss_baseline * loss_scale).backward()
        grad_baseline = student_baseline.grad.detach().clone()

        student_chunk = student.detach().clone().requires_grad_(True)
        rkl_loss_chunk, effective_chunk = backward_topk_tail_reverse_kl_on_logits(
            student_logits=student_chunk,
            teacher_topk_log_probs=teacher_topk_log_probs,
            teacher_topk_indices=teacher_topk_indices,
            response_mask=response_mask,
            loss_agg_mode=mode,
            loss_scale=loss_scale,
            vocab_chunk_size=7,
            token_chunk_size=2,
        )
        grad_chunk = student_chunk.grad.detach().clone()

        assert effective_chunk == 2
        assert torch.allclose(
            rkl_loss_chunk,
            loss_baseline.to(dtype=rkl_loss_chunk.dtype),
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            grad_chunk,
            grad_baseline.to(dtype=grad_chunk.dtype),
            atol=1e-6,
            rtol=1e-6,
        )
