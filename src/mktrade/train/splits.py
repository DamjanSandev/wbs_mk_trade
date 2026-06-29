"""Temporal and random link-split strategies for Task A.

Temporal split (headline experiment):
  - Train graph: all exports from years <= train_end_year (e.g. 2019).
  - Val positive edges: new exports that appear in val_year (2020) but NOT in train.
  - Test positive edges: new exports that appear in test_years (2021-2022) but NOT in train.
  - Negative samples: random (country, product) pairs that never appear.

Random split:
  - PyG's RandomLinkSplit on a single-year snapshot (for ablation).

Both produce train/val/test HeteroData objects with:
  - edge_label_index: [2, N] tensor of candidate edges
  - edge_label: [N] tensor of 0/1 labels
on the supervised edge type ('country', 'exports', 'product').
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from loguru import logger
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

from mktrade.config import TrainConfig


def _build_snapshot_data(
    exports_df: pd.DataFrame,
    country_features: pd.DataFrame,
    product_features: pd.DataFrame,
    proximity_df: pd.DataFrame | None,
    gravity_df: pd.DataFrame | None,
    bilateral_df: pd.DataFrame | None,
    wdi_df: pd.DataFrame | None,
    years: list[int] | int,
    cefta_members: set[str] | None = None,
) -> HeteroData:
    """Build a PyG HeteroData for a set of years (union of edges)."""
    from mktrade.graph.pyg_data import build_hetero_data, enrich_country_features

    if isinstance(years, int):
        years = [years]

    # Filter exports to the given years
    exp = exports_df[exports_df["year"].isin(years)].copy()

    # Use latest year for non-temporal features
    latest = max(years)

    data = build_hetero_data(
        exports_df=exp,
        country_features=country_features,
        product_features=product_features,
        bilateral_df=bilateral_df,
        gravity_df=gravity_df,
        proximity_df=proximity_df,
        year=latest,
        cefta_members=cefta_members,
    )

    if wdi_df is not None:
        data = enrich_country_features(data, wdi_df, data["country"].iso3, year=latest)

    return data


def temporal_split(
    exports_df: pd.DataFrame,
    country_features: pd.DataFrame,
    product_features: pd.DataFrame,
    cfg: TrainConfig,
    proximity_df: pd.DataFrame | None = None,
    gravity_df: pd.DataFrame | None = None,
    bilateral_df: pd.DataFrame | None = None,
    wdi_df: pd.DataFrame | None = None,
    cefta_members: set[str] | None = None,
    neg_ratio: int = 1,
) -> tuple[HeteroData, HeteroData, HeteroData]:
    """Temporal link-split for Task A (product diversification).

    Train: exports with year <= train_end_year.
    Val positives: NEW exports in val_year not present in train.
    Test positives: NEW exports in test_years not present in train.
    Negatives: random country-product pairs never observed.

    Returns (train_data, val_data, test_data) where val/test have
    edge_label_index and edge_label on ('country','exports','product').
    """
    train_end = cfg.train_end_year
    val_year = cfg.val_year
    test_years = cfg.test_years

    logger.info(f"Temporal split: train<=  {train_end}, val={val_year}, test={test_years}")

    # Build the training graph (all years up to train_end)
    train_years = list(range(exports_df["year"].min(), train_end + 1))
    train_data = _build_snapshot_data(
        exports_df, country_features, product_features,
        proximity_df, gravity_df, bilateral_df, wdi_df,
        years=train_years, cefta_members=cefta_members,
    )

    # Get ID mappings from the training graph
    countries = train_data["country"].iso3
    products = train_data["product"].hs4
    country2idx = {c: i for i, c in enumerate(countries)}
    product2idx = {p: i for i, p in enumerate(products)}

    # Existing train edges as a set for fast lookup
    train_edges = set()
    ei = train_data["country", "exports", "product"].edge_index
    for i in range(ei.shape[1]):
        train_edges.add((ei[0, i].item(), ei[1, i].item()))

    # All possible edges for negative sampling
    n_countries = len(countries)
    n_products = len(products)

    # --- Val split ---
    val_data = _make_eval_split(
        exports_df, val_year, country2idx, product2idx,
        train_edges, n_countries, n_products, neg_ratio,
        split_name="val",
    )

    # --- Test split ---
    test_exp = exports_df[exports_df["year"].isin(test_years)]
    test_data = _make_eval_split(
        exports_df, test_years, country2idx, product2idx,
        train_edges, n_countries, n_products, neg_ratio,
        split_name="test",
    )

    # Attach label data to copies of train_data (for message passing graph)
    # Val and test use the TRAIN graph for message passing
    val_data_out = train_data.clone()
    val_data_out["country", "exports", "product"].edge_label_index = val_data["edge_label_index"]
    val_data_out["country", "exports", "product"].edge_label = val_data["edge_label"]

    test_data_out = train_data.clone()
    test_data_out["country", "exports", "product"].edge_label_index = test_data["edge_label_index"]
    test_data_out["country", "exports", "product"].edge_label = test_data["edge_label"]

    # Training supervision: use existing edges as positives + negatives
    train_label_data = _make_train_labels(train_edges, n_countries, n_products, neg_ratio)
    train_data["country", "exports", "product"].edge_label_index = train_label_data["edge_label_index"]
    train_data["country", "exports", "product"].edge_label = train_label_data["edge_label"]

    logger.info(
        f"  Train: {ei.shape[1]} message edges, "
        f"{int(train_label_data['edge_label'].sum())} pos / "
        f"{int((train_label_data['edge_label'] == 0).sum())} neg supervision"
    )
    logger.info(
        f"  Val: {int(val_data['edge_label'].sum())} pos / "
        f"{int((val_data['edge_label'] == 0).sum())} neg"
    )
    logger.info(
        f"  Test: {int(test_data['edge_label'].sum())} pos / "
        f"{int((test_data['edge_label'] == 0).sum())} neg"
    )

    return train_data, val_data_out, test_data_out


def _make_eval_split(
    exports_df: pd.DataFrame,
    years: int | list[int],
    country2idx: dict[str, int],
    product2idx: dict[str, int],
    train_edges: set[tuple[int, int]],
    n_countries: int,
    n_products: int,
    neg_ratio: int,
    split_name: str = "",
) -> dict[str, torch.Tensor]:
    """Build positive + negative edges for a val/test split."""
    if isinstance(years, int):
        years = [years]

    exp = exports_df[
        (exports_df["year"].isin(years)) & (exports_df["export_value"] > 0)
    ]

    # New edges: appear in these years but not in training
    pos_edges = []
    for _, row in exp.iterrows():
        c = country2idx.get(row["iso3"])
        p = product2idx.get(row["hs4"])
        if c is not None and p is not None and (c, p) not in train_edges:
            pos_edges.append((c, p))

    pos_edges = list(set(pos_edges))  # deduplicate

    if not pos_edges:
        logger.warning(f"  {split_name}: no new positive edges found!")
        # Return empty tensors
        return {
            "edge_label_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_label": torch.zeros(0, dtype=torch.float32),
        }

    # Negative sampling: random pairs not in train_edges and not in pos_edges
    all_pos = train_edges | set(pos_edges)
    n_neg = len(pos_edges) * neg_ratio
    neg_edges = []
    rng = np.random.RandomState(42)
    attempts = 0
    while len(neg_edges) < n_neg and attempts < n_neg * 20:
        c = rng.randint(0, n_countries)
        p = rng.randint(0, n_products)
        if (c, p) not in all_pos:
            neg_edges.append((c, p))
            all_pos.add((c, p))
        attempts += 1

    # Combine
    all_edges = pos_edges + neg_edges
    labels = [1.0] * len(pos_edges) + [0.0] * len(neg_edges)

    edge_label_index = torch.tensor(all_edges, dtype=torch.long).t()
    edge_label = torch.tensor(labels, dtype=torch.float32)

    logger.info(f"  {split_name}: {len(pos_edges)} pos, {len(neg_edges)} neg edges")
    return {"edge_label_index": edge_label_index, "edge_label": edge_label}


def _make_train_labels(
    train_edges: set[tuple[int, int]],
    n_countries: int,
    n_products: int,
    neg_ratio: int,
) -> dict[str, torch.Tensor]:
    """Build supervision labels for training (pos edges + negative samples)."""
    pos_list = list(train_edges)
    n_neg = len(pos_list) * neg_ratio

    rng = np.random.RandomState(42)
    neg_edges = []
    all_pos = set(train_edges)
    attempts = 0
    while len(neg_edges) < n_neg and attempts < n_neg * 20:
        c = rng.randint(0, n_countries)
        p = rng.randint(0, n_products)
        if (c, p) not in all_pos:
            neg_edges.append((c, p))
            all_pos.add((c, p))
        attempts += 1

    all_edges = pos_list + neg_edges
    labels = [1.0] * len(pos_list) + [0.0] * len(neg_edges)

    # Shuffle
    perm = rng.permutation(len(all_edges))
    all_edges = [all_edges[i] for i in perm]
    labels = [labels[i] for i in perm]

    return {
        "edge_label_index": torch.tensor(all_edges, dtype=torch.long).t(),
        "edge_label": torch.tensor(labels, dtype=torch.float32),
    }


def random_split(
    data: HeteroData,
    cfg: TrainConfig,
    edge_type: tuple[str, str, str] = ("country", "exports", "product"),
) -> tuple[HeteroData, HeteroData, HeteroData]:
    """Apply PyG RandomLinkSplit on the specified edge type."""
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        neg_sampling_ratio=1.0,
        edge_types=edge_type,
        rev_edge_types=("product", "rev_exports", "country"),
    )
    train_data, val_data, test_data = transform(data)
    logger.info(
        f"Random split on {edge_type}: "
        f"train={train_data[edge_type].edge_label.shape[0]}, "
        f"val={val_data[edge_type].edge_label.shape[0]}, "
        f"test={test_data[edge_type].edge_label.shape[0]}"
    )
    return train_data, val_data, test_data
