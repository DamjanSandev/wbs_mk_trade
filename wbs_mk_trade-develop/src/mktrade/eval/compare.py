"""Compare GNN models against all baselines and produce benchmark tables.

Baselines: complexity density, COG, Adamic-Adar/Jaccard/CN.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a comparison table from {model_name: {metric: value}} dict.

    Returns a DataFrame with models as rows and metrics as columns,
    sorted by Average Precision descending.
    """
    df = pd.DataFrame.from_dict(results, orient="index")
    if "avg_precision" in df.columns:
        df = df.sort_values("avg_precision", ascending=False)
    df.index.name = "model"

    logger.info(f"\nModel comparison ({len(df)} models):")
    # Print top metrics
    display_cols = [
        column
        for column in (
            "ndcg@10",
            "map@10",
            "precision@10",
            "recall@10",
            "mrr",
            "roc_auc",
            "avg_precision",
        )
        if column in df.columns
    ]
    if display_cols:
        logger.info(f"\n{df[display_cols].to_string(float_format='%.4f')}")

    return df


def ablation_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build ablation study table."""
    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "variant"
    return df
