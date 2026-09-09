"""Temporal and random link-split strategies for Task A.

Temporal split (headline experiment):
  - Train graph: sustained, meaningful exports achieved by train_end_year.
  - Val/test positives: unseen links meeting the same persistent outcome target.
  - Evaluation candidates: every unseen product for each country by default.
  - Validation positives are filtered out of the later test candidate universe.

Random split:
  - PyG's RandomLinkSplit on a single-year snapshot (for ablation).

Both produce train/val/test HeteroData objects with:
  - edge_label_index: [2, N] tensor of candidate edges
  - edge_label: [N] tensor of 0/1 labels
on the supervised edge type ('country', 'exports', 'product').
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.transforms import RandomLinkSplit

from mktrade.opportunities.targets import (
    SuccessCriteria,
    criteria_from_config,
    sustained_export_snapshot,
    sustained_link_set,
)

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

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
    proximity_top_k: int | None = None,
    success_criteria: SuccessCriteria | None = None,
) -> HeteroData:
    """Build a cutoff graph whose export edges are sustained successes."""
    from mktrade.graph.pyg_data import build_hetero_data, enrich_country_features

    if isinstance(years, int):
        years = [years]

    # Filter exports to the given years and retain one edge per relationship
    # that met the meaningful, consecutive-year target by the cutoff.
    exp = exports_df[exports_df["year"].isin(years)].copy()
    latest = max(years)
    criteria = success_criteria or SuccessCriteria()
    exp = sustained_export_snapshot(exp, latest, criteria)
    if exp.empty:
        raise ValueError(
            f"No sustained export links exist by {latest} under criteria {criteria}"
        )

    data = build_hetero_data(
        exports_df=exp,
        country_features=country_features,
        product_features=product_features,
        bilateral_df=bilateral_df,
        gravity_df=gravity_df,
        proximity_df=proximity_df,
        year=latest,
        cefta_members=cefta_members,
        proximity_top_k=proximity_top_k,
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
    neg_ratio: int | None = None,
    proximity_top_k: int | None = None,
) -> tuple[HeteroData, HeteroData, HeteroData]:
    """Temporal link-split for Task A (product diversification).

    Train: exports with year <= train_end_year.
    Val positives: NEW exports in val_year not present in train.
    Test positives: NEW exports in test_years not present in train.
    Negatives: every other unseen country-product pair by default. Sampled
    negatives remain available only as an explicit compatibility option.

    Returns (train_data, val_data, test_data) where val/test have
    edge_label_index and edge_label on ('country','exports','product').
    """
    train_end = cfg.train_end_year
    val_year = cfg.val_year
    test_years = cfg.test_years

    logger.info(f"Temporal split: train<=  {train_end}, val={val_year}, test={test_years}")

    success_criteria = criteria_from_config(cfg)
    first_year = int(exports_df["year"].min())

    # Validation and test candidates are scored from the graph available at
    # the declared cutoff. Training supervision uses an earlier graph and
    # labels only links that form afterwards, matching the deployment task.
    inference_years = list(range(first_year, train_end + 1))
    inference_data = _build_snapshot_data(
        exports_df, country_features, product_features,
        proximity_df, gravity_df, bilateral_df, wdi_df,
        years=inference_years, cefta_members=cefta_members,
        proximity_top_k=proximity_top_k,
        success_criteria=success_criteria,
    )

    if cfg.transition_target:
        training_cutoff = train_end - success_criteria.min_consecutive_years
        if training_cutoff < first_year:
            raise ValueError("Not enough history to build transition-based training labels")
        train_years = list(range(first_year, training_cutoff + 1))
        train_outcome_years = list(range(training_cutoff + 1, train_end + 1))
        train_data = _build_snapshot_data(
            exports_df, country_features, product_features,
            proximity_df, gravity_df, bilateral_df, wdi_df,
            years=train_years, cefta_members=cefta_members,
            proximity_top_k=proximity_top_k,
            success_criteria=success_criteria,
        )
    else:
        training_cutoff = train_end
        train_outcome_years = []
        train_data = inference_data.clone()

    # Get ID mappings from the training graph
    countries = inference_data["country"].iso3
    products = inference_data["product"].hs4
    country2idx = {c: i for i, c in enumerate(countries)}
    product2idx = {p: i for i, p in enumerate(products)}

    if train_data["country"].iso3 != countries or train_data["product"].hs4 != products:
        raise ValueError("Training and inference snapshots produced incompatible node mappings")

    # Existing train edges as a set for fast lookup
    train_edges = _edge_set(train_data)
    inference_edges = _edge_set(inference_data)
    ei = train_data["country", "exports", "product"].edge_index

    # All possible edges for negative sampling
    n_countries = len(countries)
    n_products = len(products)

    # --- Val split ---
    max_year = int(exports_df["year"].max())
    val_years = list(
        range(val_year, min(max_year, val_year + success_criteria.min_consecutive_years - 1) + 1)
    )
    val_data = _make_eval_split(
        exports_df, val_years, country2idx, product2idx,
        inference_edges, n_countries, n_products, neg_ratio or cfg.train_neg_ratio,
        split_name="val",
        success_criteria=success_criteria,
        all_candidates=cfg.eval_all_candidates,
        random_seed=cfg.seed,
    )

    # --- Test split ---
    val_positive_edges = {
        tuple(edge)
        for edge in val_data["edge_label_index"][:, val_data["edge_label"] == 1]
        .t()
        .tolist()
    }
    test_data = _make_eval_split(
        exports_df, test_years, country2idx, product2idx,
        inference_edges, n_countries, n_products, neg_ratio or cfg.train_neg_ratio,
        split_name="test",
        success_criteria=success_criteria,
        excluded_edges=val_positive_edges,
        all_candidates=cfg.eval_all_candidates,
        random_seed=cfg.seed,
    )

    # Attach label data to copies of train_data (for message passing graph)
    # Val and test use the TRAIN graph for message passing
    val_data_out = inference_data.clone()
    val_data_out["country", "exports", "product"].edge_label_index = val_data["edge_label_index"]
    val_data_out["country", "exports", "product"].edge_label = val_data["edge_label"]

    test_data_out = inference_data.clone()
    test_data_out["country", "exports", "product"].edge_label_index = test_data["edge_label_index"]
    test_data_out["country", "exports", "product"].edge_label = test_data["edge_label"]

    # Training supervision: future transitions are positives. This avoids
    # teaching the model merely to reconstruct links already visible in the
    # message-passing graph.
    if cfg.transition_target:
        transition_links = sustained_link_set(
            exports_df, success_criteria, train_outcome_years
        )
        training_positives = {
            (country2idx[country], product2idx[product])
            for country, product in transition_links
            if country in country2idx
            and product in product2idx
            and (country2idx[country], product2idx[product]) not in train_edges
        }
    else:
        training_positives = train_edges

    hard_scores = _candidate_hard_scores(
        exports_df,
        training_cutoff,
        country2idx,
        product2idx,
        train_edges,
    )
    train_label_data = _make_train_labels(
        training_positives,
        n_countries,
        n_products,
        neg_ratio or cfg.train_neg_ratio,
        excluded_edges=train_edges,
        hard_negative_scores=hard_scores,
        hard_negative_fraction=cfg.hard_negative_fraction,
        random_seed=cfg.seed,
    )
    train_data["country", "exports", "product"].edge_label_index = train_label_data["edge_label_index"]
    train_data["country", "exports", "product"].edge_label = train_label_data["edge_label"]

    logger.info(
        f"  Train: {ei.shape[1]} message edges through {training_cutoff}, "
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


def _edge_set(data: HeteroData) -> set[tuple[int, int]]:
    edge_index = data["country", "exports", "product"].edge_index
    return {tuple(edge) for edge in edge_index.t().tolist()}


def _candidate_hard_scores(
    exports_df: pd.DataFrame,
    cutoff: int,
    country2idx: dict[str, int],
    product2idx: dict[str, int],
    existing_edges: set[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Return leakage-safe density/popularity scores for negative mining."""

    scores: dict[tuple[int, int], float] = {}
    if "density" in exports_df.columns:
        frame = exports_df[exports_df["year"] <= cutoff].copy()
        if not frame.empty:
            frame = frame[frame["year"] == frame["year"].max()]
            frame["density"] = pd.to_numeric(frame["density"], errors="coerce")
            for row in frame[["iso3", "hs4", "density"]].dropna().itertuples(index=False):
                country = str(row.iso3)
                product = str(row.hs4).zfill(4)
                if country in country2idx and product in product2idx:
                    scores[(country2idx[country], product2idx[product])] = float(row.density)

    # Product ubiquity is a useful fallback when Atlas density is unavailable.
    product_counts: dict[int, int] = {}
    for _, product_idx in existing_edges:
        product_counts[product_idx] = product_counts.get(product_idx, 0) + 1
    for country_idx in range(len(country2idx)):
        for product_idx, count in product_counts.items():
            scores.setdefault((country_idx, product_idx), float(count))
    return scores


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
    success_criteria: SuccessCriteria | None = None,
    excluded_edges: set[tuple[int, int]] | None = None,
    all_candidates: bool = False,
    random_seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build a filtered query candidate set for a val/test split."""
    if isinstance(years, int):
        years = [years]

    excluded_edges = excluded_edges or set()
    if success_criteria is None:
        exp = exports_df[
            (exports_df["year"].isin(years)) & (exports_df["export_value"] > 0)
        ]
        positive_links = set(
            zip(exp["iso3"].astype(str), exp["hs4"].astype(str), strict=True)
        )
    else:
        positive_links = sustained_link_set(exports_df, success_criteria, years)

    pos_edges = sorted(
        {
            (country2idx[country], product2idx[product])
            for country, product in positive_links
            if country in country2idx
            and product in product2idx
            and (country2idx[country], product2idx[product]) not in train_edges
            and (country2idx[country], product2idx[product]) not in excluded_edges
        }
    )

    if not pos_edges and not all_candidates:
        logger.warning(f"  {split_name}: no new positive edges found!")
        # Return empty tensors
        return {
            "edge_label_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_label": torch.zeros(0, dtype=torch.float32),
        }

    if all_candidates:
        positive_set = set(pos_edges)
        all_edges = [
            (country_idx, product_idx)
            for country_idx in range(n_countries)
            for product_idx in range(n_products)
            if (country_idx, product_idx) not in train_edges
            and (country_idx, product_idx) not in excluded_edges
        ]
        labels = [1.0 if edge in positive_set else 0.0 for edge in all_edges]
        logger.info(
            f"  {split_name}: {len(pos_edges)} pos among {len(all_edges)} complete candidates"
        )
        return {
            "edge_label_index": torch.tensor(all_edges, dtype=torch.long).t().contiguous(),
            "edge_label": torch.tensor(labels, dtype=torch.float32),
        }

    # Sampled evaluation is retained only as an explicit compatibility option.
    all_pos = train_edges | set(pos_edges)
    n_neg = len(pos_edges) * neg_ratio
    neg_edges = []
    rng = np.random.RandomState(random_seed)
    attempts = 0
    while len(neg_edges) < n_neg and attempts < n_neg * 20:
        c = rng.randint(0, n_countries)
        p = rng.randint(0, n_products)
        if (c, p) not in all_pos and (c, p) not in excluded_edges:
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
    positive_edges: set[tuple[int, int]],
    n_countries: int,
    n_products: int,
    neg_ratio: int,
    excluded_edges: set[tuple[int, int]] | None = None,
    hard_negative_scores: dict[tuple[int, int], float] | None = None,
    hard_negative_fraction: float = 0.7,
    random_seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Build per-country transition labels with a hard/random negative mix."""
    if not positive_edges:
        raise ValueError("Transition training produced no positive links")
    if neg_ratio < 1:
        raise ValueError("neg_ratio must be at least one")

    excluded = set(excluded_edges or set()) | set(positive_edges)
    hard_scores = hard_negative_scores or {}
    rng = np.random.RandomState(random_seed)
    pos_by_country: dict[int, list[tuple[int, int]]] = {}
    for edge in sorted(positive_edges):
        pos_by_country.setdefault(edge[0], []).append(edge)

    neg_edges: list[tuple[int, int]] = []
    hard_fraction = float(np.clip(hard_negative_fraction, 0.0, 1.0))
    for country_idx, country_positives in pos_by_country.items():
        candidates = [
            (country_idx, product_idx)
            for product_idx in range(n_products)
            if (country_idx, product_idx) not in excluded
        ]
        required = min(len(candidates), len(country_positives) * neg_ratio)
        n_hard = min(required, int(round(required * hard_fraction)))
        ordered = sorted(
            candidates,
            key=lambda edge: hard_scores.get(edge, float("-inf")),
            reverse=True,
        )
        selected = ordered[:n_hard]
        remaining = ordered[n_hard:]
        n_random = required - len(selected)
        if n_random and remaining:
            indices = rng.choice(len(remaining), size=n_random, replace=False)
            selected.extend(remaining[index] for index in indices)
        neg_edges.extend(selected)

    pos_list = sorted(positive_edges)
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
