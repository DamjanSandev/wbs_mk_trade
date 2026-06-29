"""Link-prediction evaluation metrics.

ROC-AUC, Average Precision, Precision@K, Recall@K, MRR, Hits@K.
All metrics operate on (scores, labels) arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def roc_auc(scores: NDArray, labels: NDArray) -> float:
    """Area under the ROC curve."""
    raise NotImplementedError


def average_precision(scores: NDArray, labels: NDArray) -> float:
    """Average precision (area under PR curve)."""
    raise NotImplementedError


def precision_at_k(scores: NDArray, labels: NDArray, k: int = 50) -> float:
    """Precision among the top-k scored edges."""
    raise NotImplementedError


def recall_at_k(scores: NDArray, labels: NDArray, k: int = 50) -> float:
    """Recall among the top-k scored edges."""
    raise NotImplementedError


def mrr(scores: NDArray, labels: NDArray) -> float:
    """Mean reciprocal rank of positive edges."""
    raise NotImplementedError


def hits_at_k(scores: NDArray, labels: NDArray, k: int = 10) -> float:
    """Fraction of positive edges ranked in the top-k."""
    raise NotImplementedError


def compute_all_metrics(
    scores: NDArray, labels: NDArray, ks: list[int] | None = None
) -> dict[str, float]:
    """Compute the full metric suite and return as a dict."""
    raise NotImplementedError
