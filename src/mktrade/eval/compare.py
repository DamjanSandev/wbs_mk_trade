"""Compare GNN models against all baselines and produce benchmark tables.

Baselines: complexity density, COG, PPML gravity, Adamic-Adar/Jaccard/CN,
Neo4j GDS link-prediction pipeline.
"""

from __future__ import annotations

import pandas as pd


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a comparison table from {model_name: {metric: value}} dict.

    Returns a DataFrame with models as rows and metrics as columns,
    sorted by Average Precision descending.
    """
    raise NotImplementedError


def ablation_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build ablation study table (homo vs hetero, +/- gravity, etc.)."""
    raise NotImplementedError
