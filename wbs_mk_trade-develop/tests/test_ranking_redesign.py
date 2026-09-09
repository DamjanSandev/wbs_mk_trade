"""Tests for sustained targets, learned reranking, market features, and backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from mktrade.eval.backtest import make_backtest_windows, summarise_rank_stability
from mktrade.opportunities.features import (
    enrich_market_candidates,
    enrich_product_candidate_universe,
)
from mktrade.opportunities.reranker import OpportunityReranker
from mktrade.opportunities.targets import SuccessCriteria, sustained_link_set
from mktrade.train.splits import _make_train_labels
from mktrade.train.trainer import pairwise_ranking_loss, ranking_selection_score


def test_sustained_target_rejects_tiny_and_one_off_exports() -> None:
    exports = pd.DataFrame(
        [
            {"year": 2020, "iso3": "MKD", "hs4": "0101", "export_value": 200_000, "export_rca": 1.2},
            {"year": 2021, "iso3": "MKD", "hs4": "0101", "export_value": 250_000, "export_rca": 1.3},
            {"year": 2020, "iso3": "MKD", "hs4": "0202", "export_value": 10, "export_rca": 4.0},
            {"year": 2020, "iso3": "MKD", "hs4": "0303", "export_value": 500_000, "export_rca": 2.0},
            {"year": 2022, "iso3": "MKD", "hs4": "0303", "export_value": 600_000, "export_rca": 2.1},
        ]
    )
    criteria = SuccessCriteria(100_000, 1.0, 2)
    assert sustained_link_set(exports, criteria) == {("MKD", "0101")}


def test_validation_reranker_can_rescue_non_gnn_candidate() -> None:
    validation = pd.DataFrame(
        {
            "hs4": [f"{index:04d}" for index in range(40)],
            "gnn_score": np.linspace(1.0, 0.0, 40),
            "density": np.linspace(0.0, 1.0, 40),
        }
    )
    validation["label"] = (validation["density"] > 0.75).astype(int)
    reranker = OpportunityReranker(
        ["gnn_score", "density"], n_bootstrap=3, calibrate=False
    ).fit(validation)
    ranked = reranker.rerank(validation.drop(columns="label"), top_k=5)
    assert ranked.iloc[0]["density"] > 0.9
    assert {"predicted_probability", "probability_lower", "rank_stability"}.issubset(ranked.columns)


def test_pairwise_reranker_uses_multiple_country_queries() -> None:
    validation = pd.DataFrame(
        {
            "iso3": ["AAA"] * 20 + ["BBB"] * 20,
            "hs4": [f"{index:04d}" for index in range(40)],
            "gnn_logit": np.tile(np.linspace(1.0, 0.0, 20), 2),
            "density": np.tile(np.linspace(0.0, 1.0, 20), 2),
        }
    )
    validation["label"] = (validation["density"] > 0.8).astype(int)
    reranker = OpportunityReranker(
        ["gnn_logit", "density"],
        model_type="pairwise",
        n_bootstrap=1,
        max_pairs_per_query=20,
    ).fit(validation)

    ranked = reranker.rerank(validation.drop(columns="label"))

    assert reranker.training_queries == 2
    assert reranker.has_probability_calibration
    assert ranked.iloc[0]["density"] > 0.9
    assert "ranking_score" in ranked.columns
    assert ranked["predicted_probability"].between(0, 1).all()
    assert ranked["predicted_probability"].nunique() > 1
    assert ranked["predicted_probability"].max() > 0


def test_hard_negative_sampling_keeps_high_scoring_alternatives() -> None:
    labels = _make_train_labels(
        {(0, 0)},
        n_countries=1,
        n_products=5,
        neg_ratio=2,
        excluded_edges={(0, 0)},
        hard_negative_scores={(0, 4): 10.0, (0, 3): 9.0},
        hard_negative_fraction=1.0,
    )
    negatives = labels["edge_label_index"][:, labels["edge_label"] == 0].t().tolist()
    assert {tuple(edge) for edge in negatives} == {(0, 3), (0, 4)}


def test_pairwise_loss_rewards_positive_above_negative() -> None:
    labels = torch.tensor([1.0, 0.0])
    queries = torch.tensor([0, 0])
    good = pairwise_ranking_loss(torch.tensor([2.0, -1.0]), labels, queries)
    bad = pairwise_ranking_loss(torch.tensor([-1.0, 2.0]), labels, queries)
    assert good < bad
    score = ranking_selection_score(
        {"avg_precision": 0.02, "mrr": 0.1, "ndcg@10": 0.05},
        "ranking_composite",
    )
    assert score > 0


def test_product_candidate_universe_adds_query_relative_features() -> None:
    candidates = pd.DataFrame(
        {
            "iso3": ["AAA", "AAA"],
            "hs4": ["0101", "0202"],
            "gnn_logit": [1.0, -1.0],
            "gnn_score": [0.73, 0.27],
            "label": [1, 0],
        }
    )
    exports = pd.DataFrame(
        [
            {"year": 2017, "iso3": "AAA", "hs4": "0101", "export_value": 100.0},
            {"year": 2018, "iso3": "AAA", "hs4": "0101", "export_value": 200.0},
            {"year": 2018, "iso3": "BBB", "hs4": "0202", "export_value": 400.0},
        ]
    )
    complexity = pd.DataFrame(
        [
            {"year": 2018, "iso3": "AAA", "hs4": "0101", "density": 0.8, "pci": 1.0},
            {"year": 2018, "iso3": "AAA", "hs4": "0202", "density": 0.2, "pci": 0.5},
        ]
    )

    result = enrich_product_candidate_universe(candidates, exports, complexity, 2018)

    assert result.loc[0, "density"] == 0.8
    assert result.loc[0, "gnn_logit_percentile"] > result.loc[1, "gnn_logit_percentile"]
    assert result["global_demand_value"].gt(0).all()


def test_market_features_use_destination_demand_and_gravity() -> None:
    candidates = pd.DataFrame(
        {"hs4": ["0101"], "partner_iso3": ["DEU"], "destination_gnn_score": [0.5]}
    )
    bilateral = pd.DataFrame(
        [
            {"year": 2020, "reporter_iso3": "DEU", "partner_iso3": "FRA", "hs4": "0101", "flow": "M", "value": 1_000_000},
            {"year": 2021, "reporter_iso3": "DEU", "partner_iso3": "FRA", "hs4": "0101", "flow": "M", "value": 2_000_000},
            {"year": 2021, "reporter_iso3": "MKD", "partner_iso3": "DEU", "hs4": "9999", "flow": "X", "value": 500_000},
        ]
    )
    gravity = pd.DataFrame(
        [{"year": 2021, "iso3_o": "MKD", "iso3_d": "DEU", "dist": 1500, "contig": 0, "comlang_off": 0, "fta_wto": 1}]
    )
    result = enrich_market_candidates(
        candidates,
        country="MKD",
        bilateral_df=bilateral,
        gravity_df=gravity,
        as_of_year=2021,
    )
    assert result.loc[0, "destination_import_value"] == 2_000_000
    assert result.loc[0, "destination_import_growth"] > 0
    assert result.loc[0, "distance"] == 1500
    assert result.loc[0, "fta"] == 1
    assert result.loc[0, "market_data_available"] == 1


def test_backtest_windows_and_rank_stability() -> None:
    windows = make_backtest_windows(range(2015, 2023), [2016, 2018, 2020], persistence_years=2)
    assert [window.train_end_year for window in windows] == [2016, 2018]
    assert windows[1].validation_year == 2019
    assert windows[1].test_years == (2021, 2022)
    stability = summarise_rank_stability(
        [
            pd.DataFrame({"hs4": ["0101", "0202"], "rank": [1, 2]}),
            pd.DataFrame({"hs4": ["0101", "0202"], "rank": [2, 1]}),
        ]
    )
    assert set(stability["runs_observed"]) == {2}
    assert (stability["rank_stability"] > 0).all()
