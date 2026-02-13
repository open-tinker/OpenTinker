"""On-Policy Self-Distillation (OPSD) module.

Implements the Self-Distilled Reasoner framework (arXiv:2601.18734) where a single
model acts as both teacher and student by conditioning on different contexts:
- Student: p_S(.|x) = p_theta(.|x) — observes only the problem
- Teacher: p_T(.|x,y*) = p_theta(.|x,y*) — conditions on problem + ground-truth solution

Training minimizes per-token divergence between teacher and student distributions
over the student's own rollouts.
"""

from opentinker.self_distillation.loss import (
    compute_jsd_loss,
    compute_sampled_token_loss,
)
from opentinker.self_distillation.teacher_utils import construct_teacher_batch

__all__ = [
    "compute_jsd_loss",
    "compute_sampled_token_loss",
    "construct_teacher_batch",
]
