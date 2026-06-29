"""Link-prediction evaluation metrics.

ROC-AUC, Average Precision, Precision@K, Recall@K, MRR, Hits@K.
All metrics operate on (scores, labels) arrays where scores are predicted
link probabilities/scores and labels are binary (1=positive, 0=negative).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


def roc_auc(scores: NDArray, labels: NDArray) -> float:
    """Area under the ROC curve."""
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def average_precision(scores: NDArray, labels: NDArray) -> float:
    """Average precision (area under PR curve)."""
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def precision_at_k(scores: NDArray, labels: NDArray, k: int = 50) -> float:
    """Precision among the top-k scored edges."""
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(labels[top_k_idx].sum() / k)


def recall_at_k(scores: NDArray, labels: NDArray, k: int = 50) -> float:
    """Recall among the top-k scored edges."""
    k = min(k, len(scores))
    n_pos = labels.sum()
    if n_pos == 0 or k == 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(labels[top_k_idx].sum() / n_pos)


def mrr(scores: NDArray, labels: NDArray) -> float:
    """Mean reciprocal rank of positive edges."""
    ranked = np.argsort(scores)[::-1]
    ranks = np.empty(len(ranked), dtype=int)
    ranks[ranked] = np.arange(1, len(ranked) + 1)

    pos_ranks = ranks[labels == 1]
    if len(pos_ranks) == 0:
        return 0.0
    return float(np.mean(1.0 / pos_ranks))


def hits_at_k(scores: NDArray, labels: NDArray, k: int = 10) -> float:
    """Fraction of positive edges ranked in the top-k."""
    k = min(k, len(scores))
    n_pos = labels.sum()
    if n_pos == 0 or k == 0:
        return 0.0
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(labels[top_k_idx].sum() / n_pos)


def compute_all_metrics(
    scores: NDArray, labels: NDArray, ks: list[int] | None = None
) -> dict[str, float]:
    """Compute the full metric suite and return as a dict."""
    if ks is None:
        ks = [10, 20, 50, 100]

    results = {
        "roc_auc": roc_auc(scores, labels),
        "avg_precision": average_precision(scores, labels),
        "mrr": mrr(scores, labels),
    }
    for k in ks:
        results[f"precision@{k}"] = precision_at_k(scores, labels, k)
        results[f"recall@{k}"] = recall_at_k(scores, labels, k)
        results[f"hits@{k}"] = hits_at_k(scores, labels, k)

    return results
