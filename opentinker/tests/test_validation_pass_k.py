#!/usr/bin/env python3
"""Unit tests for validation pass@k computation."""

import pytest

from opentinker.server.http_training_server import compute_pass_at_k


def test_compute_pass_at_k_returns_none_when_k_is_one():
    assert compute_pass_at_k(scores=[1.0], uids=["u1"], k=1) is None


def test_compute_pass_at_k_with_regular_grouping():
    scores = [0.2, 0.9, 0.5, 0.7]
    uids = ["a", "a", "b", "b"]
    # best(a)=0.9, best(b)=0.7 => mean=0.8
    assert compute_pass_at_k(scores=scores, uids=uids, k=2) == pytest.approx(0.8)


def test_compute_pass_at_k_is_uid_order_invariant():
    scores = [0.9, 0.7, 0.2, 0.5]
    uids = ["a", "b", "a", "b"]
    # same groups as previous test, just shuffled
    assert compute_pass_at_k(scores=scores, uids=uids, k=2) == pytest.approx(0.8)


def test_compute_pass_at_k_returns_none_for_incomplete_group():
    scores = [0.9, 0.2, 0.7]
    uids = ["a", "a", "b"]
    assert compute_pass_at_k(scores=scores, uids=uids, k=2) is None


def test_compute_pass_at_k_matches_binary_any_success():
    scores = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    uids = ["p1", "p1", "p2", "p2", "p3", "p3"]
    # any-success per uid = [1, 0, 1] => mean=2/3
    assert compute_pass_at_k(scores=scores, uids=uids, k=2) == pytest.approx(2.0 / 3.0)
