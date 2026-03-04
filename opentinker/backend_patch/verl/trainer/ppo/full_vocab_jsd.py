# Copyright 2026 OpenTinker
#
# Licensed under the Apache License, Version 2.0.
"""Full-vocabulary JSD utilities for OPSD."""

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def _chunked_logsumexp(logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Numerically stable logsumexp over vocab dim with chunking."""
    vocab_size = logits.size(-1)
    if chunk_size <= 0 or chunk_size >= vocab_size:
        return torch.logsumexp(logits, dim=-1).float()

    max_per_token = None
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk_max = logits[..., start:end].amax(dim=-1)
        if max_per_token is None:
            max_per_token = chunk_max
        else:
            max_per_token = torch.maximum(max_per_token, chunk_max)

    assert max_per_token is not None
    # Treat the stabilizing max as a constant to keep autograd graph small.
    max_per_token = max_per_token.float().detach()
    sum_exp = torch.zeros_like(max_per_token, dtype=torch.float32)
    max_unsq = max_per_token.unsqueeze(-1)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = logits[..., start:end].float()
        sum_exp = sum_exp + torch.exp(chunk - max_unsq).sum(dim=-1)

    return max_per_token + torch.log(sum_exp.clamp_min(1e-20))


def _log1mexp(logx: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(logx)) for logx <= 0."""
    threshold = -0.6931471805599453  # log(0.5)
    # Guard slight positive drift from fp errors while preserving logx=0 -> -inf.
    logx = torch.clamp(logx, max=0.0)
    return torch.where(
        logx < threshold,
        torch.log1p(-torch.exp(logx)),
        torch.log(-torch.expm1(logx)),
    )


def compute_teacher_topk_log_probs_from_logits(
    teacher_logits: torch.Tensor,
    topk: int,
    vocab_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build teacher top-K log-prob targets from full logits.

    Returns:
        teacher_topk_log_probs: (bs, resp_len, K)
        teacher_topk_indices: (bs, resp_len, K)
    """
    if teacher_logits.ndim != 3:
        raise ValueError(
            f"Expected teacher_logits to be rank-3 [B,T,V], got {tuple(teacher_logits.shape)}"
        )

    vocab_size = teacher_logits.size(-1)
    k = int(topk)
    if k <= 0:
        raise ValueError(f"topk must be > 0, got {k}")
    k = min(k, vocab_size)

    teacher_logits_f = teacher_logits.float()
    topk_logits, topk_indices = torch.topk(teacher_logits_f, k=k, dim=-1)
    teacher_logz = _chunked_logsumexp(teacher_logits, chunk_size=vocab_chunk_size)
    teacher_topk_log_probs = topk_logits - teacher_logz.unsqueeze(-1)
    return teacher_topk_log_probs, topk_indices.to(dtype=torch.long)


def compute_token_topk_tail_reverse_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 0,
    prob_eps: float = 1e-8,
) -> torch.Tensor:
    """Compute per-token reverse-KL on K+1 bins (teacher top-K + tail).

    D_KL(pi_student^{K+1} || pi_teacher^{K+1})
    """
    if student_logits.ndim != 3:
        raise ValueError(
            f"Expected student_logits rank-3 [B,T,V], got {tuple(student_logits.shape)}"
        )
    if teacher_topk_log_probs.shape != teacher_topk_indices.shape:
        raise ValueError(
            "teacher_topk_log_probs/indices shape mismatch: "
            f"{tuple(teacher_topk_log_probs.shape)} vs "
            f"{tuple(teacher_topk_indices.shape)}"
        )
    if student_logits.shape[:2] != teacher_topk_log_probs.shape[:2]:
        raise ValueError(
            "student/topk prefix mismatch: "
            f"{tuple(student_logits.shape[:2])} vs {tuple(teacher_topk_log_probs.shape[:2])}"
        )

    student_device = student_logits.device
    response_len = student_logits.size(-2)
    vocab_size = student_logits.size(-1)
    chunk_size = int(vocab_chunk_size)
    chunk_size = max(1, min(chunk_size, vocab_size))
    tok_chunk_size = int(token_chunk_size)
    if tok_chunk_size <= 0 or tok_chunk_size >= response_len:
        tok_chunk_size = response_len

    prob_eps = float(prob_eps)
    if prob_eps <= 0.0:
        raise ValueError(f"prob_eps must be > 0, got {prob_eps}")

    loss_token_chunks = []
    for tok_start in range(0, response_len, tok_chunk_size):
        tok_end = min(tok_start + tok_chunk_size, response_len)
        student_logits_chunk = student_logits[:, tok_start:tok_end, :]
        teacher_topk_log_probs_chunk = teacher_topk_log_probs[:, tok_start:tok_end, :]
        teacher_topk_indices_chunk = teacher_topk_indices[:, tok_start:tok_end, :]

        if teacher_topk_log_probs_chunk.device != student_device:
            teacher_topk_log_probs_chunk = teacher_topk_log_probs_chunk.to(
                device=student_device, dtype=torch.float32, non_blocking=True
            )
        else:
            teacher_topk_log_probs_chunk = teacher_topk_log_probs_chunk.float()
        if teacher_topk_indices_chunk.device != student_device:
            teacher_topk_indices_chunk = teacher_topk_indices_chunk.to(
                device=student_device, dtype=torch.long, non_blocking=True
            )
        else:
            teacher_topk_indices_chunk = teacher_topk_indices_chunk.long()

        student_logz = _chunked_logsumexp(student_logits_chunk, chunk_size=chunk_size)
        student_topk_logits = torch.gather(
            student_logits_chunk.float(),
            dim=-1,
            index=teacher_topk_indices_chunk,
        )
        student_topk_log_probs = student_topk_logits - student_logz.unsqueeze(-1)

        student_topk_logsumexp = torch.logsumexp(student_topk_log_probs, dim=-1)
        teacher_topk_logsumexp = torch.logsumexp(teacher_topk_log_probs_chunk, dim=-1)
        student_tail_log_prob = _log1mexp(student_topk_logsumexp)
        teacher_tail_log_prob = _log1mexp(teacher_topk_logsumexp)

        # Stabilize KL when tail mass becomes numerically zero in float32.
        # Build K+1 bins, apply epsilon floor, then renormalize.
        student_bins = torch.cat(
            [
                torch.exp(student_topk_log_probs),
                torch.exp(student_tail_log_prob).unsqueeze(-1),
            ],
            dim=-1,
        )
        teacher_bins = torch.cat(
            [
                torch.exp(teacher_topk_log_probs_chunk),
                torch.exp(teacher_tail_log_prob).unsqueeze(-1),
            ],
            dim=-1,
        )
        student_bins = student_bins.clamp_min(prob_eps)
        teacher_bins = teacher_bins.clamp_min(prob_eps)
        student_bins = student_bins / student_bins.sum(dim=-1, keepdim=True).clamp_min(
            prob_eps
        )
        teacher_bins = teacher_bins / teacher_bins.sum(dim=-1, keepdim=True).clamp_min(
            prob_eps
        )
        student_log_bins = torch.log(student_bins)
        teacher_log_bins = torch.log(teacher_bins)
        loss_token_chunks.append(
            torch.sum(student_bins * (student_log_bins - teacher_log_bins), dim=-1)
        )

    return torch.cat(loss_token_chunks, dim=-1)


def compute_token_jsd_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float,
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 0,
) -> torch.Tensor:
    """Compute per-token JSD(beta) over full vocabulary.

    Args:
        student_logits: (bs, resp_len, vocab)
        teacher_logits: (bs, resp_len, vocab)
        beta: mixture weight in [0, 1]
        vocab_chunk_size: vocab chunk size for memory-efficient exact computation.
        token_chunk_size: response-token chunk size. <=0 disables token chunking.
    Returns:
        jsd_per_token: (bs, resp_len)
    """
    beta = float(beta)
    if beta < 0.0 or beta > 1.0:
        raise ValueError(f"JSD beta must be in [0, 1], got {beta}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student/teacher logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )

    if beta == 0.0 or beta == 1.0:
        return torch.zeros(
            student_logits.shape[:-1], dtype=torch.float32, device=student_logits.device
        )

    student_device = student_logits.device
    response_len = student_logits.size(-2)
    vocab_size = student_logits.size(-1)
    chunk_size = int(vocab_chunk_size)
    chunk_size = max(1, min(chunk_size, vocab_size))
    tok_chunk_size = int(token_chunk_size)
    if tok_chunk_size <= 0 or tok_chunk_size >= response_len:
        tok_chunk_size = response_len

    log_beta = math.log(beta)
    log_one_minus_beta = math.log(1.0 - beta)
    use_checkpoint = bool(student_logits.requires_grad and torch.is_grad_enabled())
    jsd_token_chunks = []
    for tok_start in range(0, response_len, tok_chunk_size):
        tok_end = min(tok_start + tok_chunk_size, response_len)
        student_logits_chunk = student_logits[:, tok_start:tok_end, :]
        teacher_logits_chunk = teacher_logits[:, tok_start:tok_end, :]

        # Avoid materializing a full fp32 [B, T, V] copy for student logits.
        # Chunked logsumexp keeps peak memory bounded by vocab_chunk_size.
        s_logz = _chunked_logsumexp(student_logits_chunk, chunk_size=chunk_size)
        t_logz = _chunked_logsumexp(teacher_logits_chunk, chunk_size=chunk_size)
        if t_logz.device != student_device:
            t_logz = t_logz.to(device=student_device, non_blocking=True)
        s_logz_unsq = s_logz.unsqueeze(-1)
        t_logz_unsq = t_logz.unsqueeze(-1)

        kl_t_m = torch.zeros_like(s_logz, dtype=torch.float32)
        kl_s_m = torch.zeros_like(s_logz, dtype=torch.float32)
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            s_chunk = student_logits_chunk[..., start:end].float()

            def _compute_chunk_kl(
                s_chunk_inner: torch.Tensor,
                chunk_start: int = start,
                chunk_end: int = end,
                teacher_logits_chunk_inner: torch.Tensor = teacher_logits_chunk,
                s_logz_unsq_inner: torch.Tensor = s_logz_unsq,
                t_logz_unsq_inner: torch.Tensor = t_logz_unsq,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if teacher_logits_chunk_inner.device == student_device:
                    t_chunk = teacher_logits_chunk_inner[
                        ..., chunk_start:chunk_end
                    ].float()
                else:
                    t_chunk = teacher_logits_chunk_inner[..., chunk_start:chunk_end].to(
                        device=student_device, dtype=torch.float32, non_blocking=True
                    )

                s_logp = s_chunk_inner - s_logz_unsq_inner
                t_logp = t_chunk - t_logz_unsq_inner
                mix_logp = torch.logaddexp(
                    t_logp + log_beta, s_logp + log_one_minus_beta
                )

                t_prob = torch.exp(t_logp)
                s_prob = torch.exp(s_logp)
                kl_t_chunk = torch.sum(t_prob * (t_logp - mix_logp), dim=-1)
                kl_s_chunk = torch.sum(s_prob * (s_logp - mix_logp), dim=-1)
                return kl_t_chunk, kl_s_chunk

            if use_checkpoint:
                kl_t_chunk, kl_s_chunk = torch_checkpoint(
                    _compute_chunk_kl,
                    s_chunk,
                    use_reentrant=False,
                )
            else:
                kl_t_chunk, kl_s_chunk = _compute_chunk_kl(s_chunk)

            kl_t_m = kl_t_m + kl_t_chunk
            kl_s_m = kl_s_m + kl_s_chunk
            del s_chunk, kl_t_chunk, kl_s_chunk

        jsd_token_chunks.append(beta * kl_t_m + (1.0 - beta) * kl_s_m)
        if not use_checkpoint:
            del student_logits_chunk, teacher_logits_chunk, s_logz, t_logz, s_logz_unsq, t_logz_unsq

    return torch.cat(jsd_token_chunks, dim=-1)


def _build_token_weights(
    response_mask: torch.Tensor,
    loss_agg_mode: str,
) -> torch.Tensor:
    """Build per-token weights equivalent to agg_loss aggregation."""
    mask = response_mask.float()
    if loss_agg_mode == "token-mean":
        denom = mask.sum().clamp_min(1e-8)
        return mask / denom

    if loss_agg_mode == "seq-mean-token-sum":
        seq_mask = (mask.sum(dim=-1) > 0).float()
        seq_denom = seq_mask.sum().clamp_min(1e-8)
        return mask * (seq_mask / seq_denom).unsqueeze(-1)

    if loss_agg_mode == "seq-mean-token-mean":
        seq_token_count = mask.sum(dim=-1)
        seq_mask = (seq_token_count > 0).float()
        seq_denom = seq_mask.sum().clamp_min(1e-8)
        inv_seq_token_count = seq_mask / (seq_token_count + 1e-8)
        return mask * (inv_seq_token_count / seq_denom).unsqueeze(-1)

    if loss_agg_mode == "seq-mean-token-sum-norm":
        return mask / float(mask.shape[-1])

    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def backward_full_vocab_jsd_on_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
    loss_agg_mode: str,
    loss_scale: float,
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 0,
    logits_grad_cpu_offload: bool = True,
) -> tuple[torch.Tensor, int]:
    """Memory-efficient backward for full-vocab JSD on logits.

    This avoids attaching the JSD graph to the model graph over the whole
    response. It computes dL/d(student_logits_chunk) on detached logits and then
    backpropagates chunk gradients to the original logits.

    Returns:
        jsd_loss: unscaled scalar JSD loss (after aggregation)
        effective_token_chunk_size: actual token chunk size used
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student/teacher logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if response_mask.shape != student_logits.shape[:-1]:
        raise ValueError(
            "response_mask shape mismatch with logits: "
            f"{tuple(response_mask.shape)} vs {tuple(student_logits.shape[:-1])}"
        )

    response_len = student_logits.size(-2)
    tok_chunk_size = int(token_chunk_size)
    if tok_chunk_size <= 0 or tok_chunk_size >= response_len:
        tok_chunk_size = response_len

    token_weights = _build_token_weights(response_mask, loss_agg_mode).to(
        device=student_logits.device, dtype=torch.float32
    )

    student_logits_proxy = student_logits.detach().requires_grad_(True)
    jsd_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    num_chunks = (response_len + tok_chunk_size - 1) // tok_chunk_size

    # FSDP can fail when the same forward graph is backpropagated multiple times
    # (retain_graph=True across token chunks), because parameter views may be
    # resharded to empty storages between backward passes.
    logits_grad = None
    logits_grad_host = None
    if num_chunks > 1:
        # Keep the dense [B, T, V] grad buffer off GPU during chunk processing.
        # This lowers peak GPU memory when token_chunk_size > 0.
        if logits_grad_cpu_offload and student_logits.is_cuda:
            logits_grad_host = torch.empty(
                student_logits.shape,
                dtype=student_logits.dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            logits_grad = torch.empty_like(student_logits, dtype=student_logits.dtype)

    for tok_start in range(0, response_len, tok_chunk_size):
        tok_end = min(tok_start + tok_chunk_size, response_len)
        student_chunk_proxy = student_logits_proxy[:, tok_start:tok_end, :]
        teacher_chunk = teacher_logits[:, tok_start:tok_end, :]
        weight_chunk = token_weights[:, tok_start:tok_end]

        jsd_chunk = compute_token_jsd_from_logits(
            student_logits=student_chunk_proxy,
            teacher_logits=teacher_chunk,
            beta=beta,
            vocab_chunk_size=vocab_chunk_size,
            token_chunk_size=0,
        )
        chunk_loss = torch.sum(jsd_chunk * weight_chunk)
        grad_chunk = torch.autograd.grad(
            chunk_loss, student_chunk_proxy, retain_graph=False, create_graph=False
        )[0]

        grad_to_model = (grad_chunk * float(loss_scale)).to(dtype=student_logits.dtype)
        if logits_grad is None and logits_grad_host is None:
            student_logits[:, tok_start:tok_end, :].backward(
                grad_to_model, retain_graph=False
            )
        elif logits_grad_host is not None:
            logits_grad_host[:, tok_start:tok_end, :].copy_(
                grad_to_model, non_blocking=False
            )
        else:
            logits_grad[:, tok_start:tok_end, :] = grad_to_model

        jsd_loss = jsd_loss + chunk_loss.detach()
        del jsd_chunk, chunk_loss, grad_chunk, grad_to_model

    if logits_grad_host is not None:
        logits_grad = logits_grad_host.to(
            device=student_logits.device, non_blocking=True
        )
        del logits_grad_host
    if logits_grad is not None:
        student_logits.backward(logits_grad, retain_graph=False)
        del logits_grad

    return jsd_loss, tok_chunk_size


def backward_topk_tail_reverse_kl_on_logits(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    loss_scale: float,
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 0,
    logits_grad_cpu_offload: bool = True,
) -> tuple[torch.Tensor, int]:
    """Memory-efficient backward for top-K + tail reverse-KL distillation."""
    if student_logits.shape[:2] != teacher_topk_log_probs.shape[:2]:
        raise ValueError(
            "student/topk prefix mismatch: "
            f"{tuple(student_logits.shape[:2])} vs {tuple(teacher_topk_log_probs.shape[:2])}"
        )
    if teacher_topk_log_probs.shape != teacher_topk_indices.shape:
        raise ValueError(
            "teacher_topk_log_probs/indices shape mismatch: "
            f"{tuple(teacher_topk_log_probs.shape)} vs {tuple(teacher_topk_indices.shape)}"
        )
    if response_mask.shape != student_logits.shape[:-1]:
        raise ValueError(
            "response_mask shape mismatch with logits: "
            f"{tuple(response_mask.shape)} vs {tuple(student_logits.shape[:-1])}"
        )

    response_len = student_logits.size(-2)
    tok_chunk_size = int(token_chunk_size)
    if tok_chunk_size <= 0 or tok_chunk_size >= response_len:
        tok_chunk_size = response_len

    token_weights = _build_token_weights(response_mask, loss_agg_mode).to(
        device=student_logits.device, dtype=torch.float32
    )

    student_logits_proxy = student_logits.detach().requires_grad_(True)
    distill_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    num_chunks = (response_len + tok_chunk_size - 1) // tok_chunk_size

    logits_grad = None
    logits_grad_host = None
    if num_chunks > 1:
        if logits_grad_cpu_offload and student_logits.is_cuda:
            logits_grad_host = torch.empty(
                student_logits.shape,
                dtype=student_logits.dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            logits_grad = torch.empty_like(student_logits, dtype=student_logits.dtype)

    for tok_start in range(0, response_len, tok_chunk_size):
        tok_end = min(tok_start + tok_chunk_size, response_len)
        student_chunk_proxy = student_logits_proxy[:, tok_start:tok_end, :]
        teacher_topk_log_probs_chunk = teacher_topk_log_probs[:, tok_start:tok_end, :]
        teacher_topk_indices_chunk = teacher_topk_indices[:, tok_start:tok_end, :]
        weight_chunk = token_weights[:, tok_start:tok_end]

        rkl_chunk = compute_token_topk_tail_reverse_kl_from_logits(
            student_logits=student_chunk_proxy,
            teacher_topk_log_probs=teacher_topk_log_probs_chunk,
            teacher_topk_indices=teacher_topk_indices_chunk,
            vocab_chunk_size=vocab_chunk_size,
            token_chunk_size=0,
        )
        chunk_loss = torch.sum(rkl_chunk * weight_chunk)
        grad_chunk = torch.autograd.grad(
            chunk_loss,
            student_chunk_proxy,
            retain_graph=False,
            create_graph=False,
        )[0]

        grad_to_model = (grad_chunk * float(loss_scale)).to(dtype=student_logits.dtype)
        if logits_grad is None and logits_grad_host is None:
            student_logits[:, tok_start:tok_end, :].backward(
                grad_to_model, retain_graph=False
            )
        elif logits_grad_host is not None:
            logits_grad_host[:, tok_start:tok_end, :].copy_(
                grad_to_model, non_blocking=False
            )
        else:
            logits_grad[:, tok_start:tok_end, :] = grad_to_model

        distill_loss = distill_loss + chunk_loss.detach()
        del rkl_chunk, chunk_loss, grad_chunk, grad_to_model

    if logits_grad_host is not None:
        logits_grad = logits_grad_host.to(
            device=student_logits.device, non_blocking=True
        )
        del logits_grad_host
    if logits_grad is not None:
        student_logits.backward(logits_grad, retain_graph=False)
        del logits_grad

    return distill_loss, tok_chunk_size
