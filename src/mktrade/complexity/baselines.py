"""Complexity-based baselines for link prediction.

Baseline 1 — Density ranking: rank Country-Product pairs by density (omega_cp).
Baseline 2 — COG (Centre of Gravity): rank products by closeness to a country's
current export basket in the product space.
"""

from __future__ import annotations

import pandas as pd


def density_baseline(
    density_df: pd.DataFrame,
    country: str = "MKD",
    top_k: int = 50,
) -> pd.DataFrame:
    """Rank non-exported products by density for a given country.

    Returns top_k products with columns: hs4, density, rank.
    """
    raise NotImplementedError


def cog_baseline(
    complexity_df: pd.DataFrame,
    country: str = "MKD",
    top_k: int = 50,
) -> pd.DataFrame:
    """Rank products by COG (distance to centre of gravity in product space).

    Returns top_k products with columns: hs4, cog_score, rank.
    """
    raise NotImplementedError
