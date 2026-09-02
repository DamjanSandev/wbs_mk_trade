"""Classification and query-based ranking metrics for link prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

if TYPE_CHECKING:
    from numpy.typing import NDArray


def roc_auc(scores: NDArray, labels: NDArray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def average_precision(scores: NDArray, labels: NDArray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(average_precision_score(labels, scores))


def precision_at_k(scores: NDArray, labels: NDArray, k: int = 50) -> float:
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    ranked = np.argsort(scores)[::-1][:k]
    return float(np.asarray(labels)[ranked].sum() / k)


def recall_at_k(scores: NDArray, labels: NDArray, k: int = 50) -> float:
    k = min(k, len(scores))
    positives = np.asarray(labels).sum()
    if positives == 0 or k == 0:
        return 0.0
    ranked = np.argsort(scores)[::-1][:k]
    return float(np.asarray(labels)[ranked].sum() / positives)


def average_precision_at_k(scores: NDArray, labels: NDArray, k: int = 50) -> float:
    """Average precision truncated at K for one query."""

    labels = np.asarray(labels)
    positives = int(labels.sum())
    k = min(k, len(scores))
    if positives == 0 or k == 0:
        return 0.0
    ranked_labels = labels[np.argsort(scores)[::-1][:k]]
    precisions = np.cumsum(ranked_labels) / np.arange(1, k + 1)
    return float((precisions * ranked_labels).sum() / min(positives, k))


def ndcg_at_k(scores: NDArray, labels: NDArray, k: int = 50) -> float:
    """Binary normalized discounted cumulative gain for one query."""

    labels = np.asarray(labels)
    k = min(k, len(scores))
    positives = int(labels.sum())
    if positives == 0 or k == 0:
        return 0.0
    ranked = labels[np.argsort(scores)[::-1][:k]]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((ranked * discounts).sum())
    ideal = float(discounts[: min(positives, k)].sum())
    return dcg / ideal if ideal > 0 else 0.0


def mrr(scores: NDArray, labels: NDArray) -> float:
    """Mean reciprocal rank of all positive edges in a single/global query."""

    ranked = np.argsort(scores)[::-1]
    ranks = np.empty(len(ranked), dtype=int)
    ranks[ranked] = np.arange(1, len(ranked) + 1)
    positive_ranks = ranks[np.asarray(labels) == 1]
    if len(positive_ranks) == 0:
        return 0.0
    return float(np.mean(1.0 / positive_ranks))


def reciprocal_rank(scores: NDArray, labels: NDArray) -> float:
    """Reciprocal rank of the first relevant result for one query."""

    ranked_labels = np.asarray(labels)[np.argsort(scores)[::-1]]
    relevant = np.flatnonzero(ranked_labels == 1)
    return 0.0 if len(relevant) == 0 else float(1.0 / (relevant[0] + 1))


def hits_at_k(scores: NDArray, labels: NDArray, k: int = 10) -> float:
    """Fraction of a query's positives recovered in the top K."""

    return recall_at_k(scores, labels, k)


def compute_query_metrics(
    scores: NDArray,
    labels: NDArray,
    query_ids: NDArray,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Macro-average ranking metrics across queries that have positives."""

    if ks is None:
        ks = [10, 20, 50, 100]
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    query_ids = np.asarray(query_ids)
    if not (len(scores) == len(labels) == len(query_ids)):
        raise ValueError("scores, labels, and query_ids must have equal length")

    query_slices = []
    for query in np.unique(query_ids):
        mask = query_ids == query
        if labels[mask].sum() > 0:
            query_slices.append((scores[mask], labels[mask]))
    if not query_slices:
        empty = {"mrr": 0.0, "queries_evaluated": 0.0}
        for k in ks:
            empty.update(
                {
                    f"precision@{k}": 0.0,
                    f"recall@{k}": 0.0,
                    f"hits@{k}": 0.0,
                    f"map@{k}": 0.0,
                    f"ndcg@{k}": 0.0,
                }
            )
        return empty

    result = {
        "mrr": float(np.mean([reciprocal_rank(qs, ql) for qs, ql in query_slices])),
        "queries_evaluated": float(len(query_slices)),
    }
    for k in ks:
        result[f"precision@{k}"] = float(
            np.mean([precision_at_k(qs, ql, k) for qs, ql in query_slices])
        )
        result[f"recall@{k}"] = float(
            np.mean([recall_at_k(qs, ql, k) for qs, ql in query_slices])
        )
        result[f"hits@{k}"] = result[f"recall@{k}"]
        result[f"map@{k}"] = float(
            np.mean([average_precision_at_k(qs, ql, k) for qs, ql in query_slices])
        )
        result[f"ndcg@{k}"] = float(
            np.mean([ndcg_at_k(qs, ql, k) for qs, ql in query_slices])
        )
    return result


def compute_all_metrics(
    scores: NDArray,
    labels: NDArray,
    ks: list[int] | None = None,
    query_ids: NDArray | None = None,
) -> dict[str, float]:
    """Compute global diagnostics plus ranking metrics for complete queries."""

    if ks is None:
        ks = [10, 20, 50, 100]
    results = {
        "roc_auc": roc_auc(scores, labels),
        "avg_precision": average_precision(scores, labels),
    }
    if query_ids is not None:
        results.update(compute_query_metrics(scores, labels, query_ids, ks))
        return results

    results["mrr"] = mrr(scores, labels)
    for k in ks:
        results[f"precision@{k}"] = precision_at_k(scores, labels, k)
        results[f"recall@{k}"] = recall_at_k(scores, labels, k)
        results[f"hits@{k}"] = hits_at_k(scores, labels, k)
        results[f"map@{k}"] = average_precision_at_k(scores, labels, k)
        results[f"ndcg@{k}"] = ndcg_at_k(scores, labels, k)
    return results
