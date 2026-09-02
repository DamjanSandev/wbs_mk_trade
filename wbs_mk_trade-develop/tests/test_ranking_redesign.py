"""Tests for sustained targets, learned reranking, market features, and backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mktrade.eval.backtest import make_backtest_windows, summarise_rank_stability
from mktrade.opportunities.features import enrich_market_candidates
from mktrade.opportunities.reranker import OpportunityReranker
from mktrade.opportunities.targets import SuccessCriteria, sustained_link_set


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
    windows = make_backtest_windows(range(2015, 2023), [2017, 2019, 2021], persistence_years=2)
    assert [window.train_end_year for window in windows] == [2017, 2019]
    stability = summarise_rank_stability(
        [
            pd.DataFrame({"hs4": ["0101", "0202"], "rank": [1, 2]}),
            pd.DataFrame({"hs4": ["0101", "0202"], "rank": [2, 1]}),
        ]
    )
    assert set(stability["runs_observed"]) == {2}
    assert (stability["rank_stability"] > 0).all()
