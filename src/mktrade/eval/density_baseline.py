"""Complexity density/COG baseline for Task A.

Uses the economic complexity "density" (probability that a country develops
comparative advantage in a product given its current export basket) as
a link-prediction score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

from mktrade.eval.metrics import compute_all_metrics


def density_baseline(
    density_df: pd.DataFrame,
    train_edges: set[tuple[str, str]],
    test_edges: list[tuple[str, str]],
    test_labels: NDArray,
    score_col: str = "density",
) -> dict[str, float]:
    """Score test edges using the complexity density metric.

    Parameters
    ----------
    density_df : columns [iso3, hs4, density] (from density_matrix.parquet or complexity.parquet).
    train_edges : set of (iso3, hs4) existing in training data.
    test_edges : list of (iso3, hs4) candidate edges.
    test_labels : binary labels for test_edges.
    score_col : column to use as prediction score.
    """
    lookup = density_df.set_index(["iso3", "hs4"])[score_col].to_dict()

    scores = np.array([lookup.get(e, 0.0) for e in test_edges], dtype=np.float64)

    metrics = compute_all_metrics(scores, test_labels, ks=[10, 20, 50, 100])
    logger.info(f"Density baseline: AUC={metrics['roc_auc']:.4f}, AP={metrics['avg_precision']:.4f}")
    return metrics
