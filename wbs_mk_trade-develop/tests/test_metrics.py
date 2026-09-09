"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mktrade.eval.metrics import (
    average_precision,
    compute_all_metrics,
    compute_query_metrics,
    hits_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    roc_auc,
)


def test_perfect_auc() -> None:
    """A perfect classifier should have AUC = 1.0."""
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    labels = np.array([1, 1, 0, 0])
    assert roc_auc(scores, labels) == 1.0


def test_random_auc_around_half() -> None:
    """A random classifier should have AUC ~ 0.5."""
    rng = np.random.RandomState(42)
    scores = rng.rand(1000)
    labels = rng.randint(0, 2, 1000)
    auc = roc_auc(scores, labels)
    assert 0.4 < auc < 0.6


def test_average_precision_perfect() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    labels = np.array([1, 1, 0, 0])
    assert average_precision(scores, labels) == 1.0


def test_precision_at_k() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    labels = np.array([1, 0, 1, 0, 0])
    # Top 2: [0.9(1), 0.8(0)] -> precision = 1/2
    assert precision_at_k(scores, labels, k=2) == 0.5
    # Top 4: [0.9(1), 0.8(0), 0.7(1), 0.6(0)] -> precision = 2/4
    assert precision_at_k(scores, labels, k=4) == 0.5


def test_recall_at_k() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    labels = np.array([1, 0, 1, 0, 0])
    # Top 2: hits 1 of 2 positives -> recall = 0.5
    assert recall_at_k(scores, labels, k=2) == 0.5
    # Top 4: hits 2 of 2 positives -> recall = 1.0
    assert recall_at_k(scores, labels, k=4) == 1.0


def test_hits_at_k_bounds() -> None:
    """Hits@K should be in [0, 1]."""
    scores = np.array([0.9, 0.5, 0.3, 0.1])
    labels = np.array([1, 0, 1, 0])
    h = hits_at_k(scores, labels, k=1)
    assert 0.0 <= h <= 1.0
    # Top 1 hits 1 of 2 pos -> 0.5
    assert h == 0.5


def test_mrr_perfect() -> None:
    scores = np.array([0.9, 0.8, 0.1])
    labels = np.array([1, 0, 0])
    # Only positive is ranked #1 -> MRR = 1.0
    assert mrr(scores, labels) == 1.0


def test_mrr_second_rank() -> None:
    scores = np.array([0.9, 0.8, 0.1])
    labels = np.array([0, 1, 0])
    # Only positive is ranked #2 -> MRR = 0.5
    assert mrr(scores, labels) == 0.5


def test_compute_all_metrics_keys() -> None:
    scores = np.array([0.9, 0.5, 0.3, 0.1])
    labels = np.array([1, 0, 1, 0])
    m = compute_all_metrics(scores, labels, ks=[10, 50])
    assert "roc_auc" in m
    assert "avg_precision" in m
    assert "mrr" in m
    assert "precision@10" in m
    assert "hits@50" in m


def test_all_same_label_returns_zero() -> None:
    """If all labels are the same, AUC/AP should be 0 (not crash)."""
    scores = np.array([0.5, 0.3, 0.7])
    labels = np.array([1, 1, 1])
    assert roc_auc(scores, labels) == 0.0
    assert average_precision(scores, labels) == 0.0


def test_query_metrics_are_macro_averaged() -> None:
    # Query 0 finds its positive first; query 1 finds it second.
    scores = np.array([0.9, 0.1, 0.8, 0.7])
    labels = np.array([1, 0, 0, 1])
    queries = np.array([0, 0, 1, 1])
    metrics = compute_query_metrics(scores, labels, queries, ks=[1, 2])
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["map"] == pytest.approx(0.75)
    assert metrics["precision@1"] == pytest.approx(0.5)
    assert metrics["recall@2"] == 1.0
    assert "map@2" in metrics
    assert "ndcg@2" in metrics


def test_ndcg_perfect_ranking() -> None:
    assert ndcg_at_k(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0]), 3) == 1.0
