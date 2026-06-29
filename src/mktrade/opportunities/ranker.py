"""Rank predicted export opportunities for North Macedonia.

Task A: top-K new products MKD could export (country-product links).
Task B: top-K new markets for MKD's existing products (country-product-country links).
"""

from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from mktrade.models.link_predictor import LinkPredictor


def rank_product_opportunities(
    model: LinkPredictor,
    data: HeteroData,
    country: str = "MKD",
    top_k: int = 50,
) -> pd.DataFrame:
    """Score and rank all non-exported products for a country.

    Returns columns: hs4, product_name, score, rank, density, pci.
    """
    raise NotImplementedError


def rank_market_opportunities(
    model: LinkPredictor,
    data: HeteroData,
    country: str = "MKD",
    top_k: int = 50,
) -> pd.DataFrame:
    """Score and rank new (product, destination) pairs for a country.

    Returns columns: hs4, product_name, partner_iso3, partner_name,
    score, rank, gravity_pred.
    """
    raise NotImplementedError
