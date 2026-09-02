"""Tests for Phase 5: opportunity ranking, explanation, and visualization."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from mktrade.models.decoders import DotProductDecoder
from mktrade.models.encoders import HomoEncoder
from mktrade.models.link_predictor import LinkPredictor


# ── Fixtures ──

@pytest.fixture
def toy_hetero_data() -> HeteroData:
    """Build a small toy HeteroData for testing."""
    data = HeteroData()

    # 5 countries, 10 products
    data["country"].x = torch.randn(5, 14)
    data["country"].iso3 = ["ALB", "MKD", "SRB", "DEU", "GRC"]
    data["country"].num_nodes = 5

    data["product"].x = torch.randn(10, 23)
    data["product"].hs4 = [f"{i:04d}" for i in range(10)]
    data["product"].num_nodes = 10

    data["product_section"].x = torch.eye(3)
    data["product_section"].num_nodes = 3

    # MKD (idx=1) exports products 0-4
    src = [1, 1, 1, 1, 1, 0, 0, 3, 3, 4]
    dst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    data["country", "exports", "product"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data["product", "rev_exports", "country"].edge_index = torch.tensor([dst, src], dtype=torch.long)

    # in_section edges
    data["product", "in_section", "product_section"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]], dtype=torch.long
    )
    data["product_section", "rev_in_section", "product"].edge_index = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long
    )

    # proximity (just a few)
    data["product", "proximity", "product"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
    )

    return data


@pytest.fixture
def toy_model(toy_hetero_data) -> LinkPredictor:
    """Build a small model for testing."""
    from torch_geometric.nn import to_hetero

    encoder = HomoEncoder(in_channels=-1, hidden_channels=16, out_channels=16, num_layers=2)
    encoder = to_hetero(encoder, toy_hetero_data.metadata(), aggr="sum")
    decoder = DotProductDecoder()
    model = LinkPredictor(encoder, decoder)

    # Initialize lazy params
    with torch.no_grad():
        edge_label_index = torch.tensor([[0], [0]], dtype=torch.long)
        model(toy_hetero_data, edge_label_index, "country", "product")

    return model


# ── Ranker tests ──

def test_rank_product_opportunities(toy_model, toy_hetero_data):
    from mktrade.opportunities.ranker import rank_product_opportunities

    df = rank_product_opportunities(
        model=toy_model,
        data=toy_hetero_data,
        country="MKD",
        top_k=10,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "hs4" in df.columns
    assert "score" in df.columns
    assert "rank" in df.columns
    # MKD exports 5 products (0-4), so candidates should be products 5-9
    assert len(df) == 5
    assert df["rank"].tolist() == [1, 2, 3, 4, 5]
    # All scores should be probabilities in [0, 1]
    assert (df["score"] >= 0).all() and (df["score"] <= 1).all()


def test_rank_product_unknown_country(toy_model, toy_hetero_data):
    from mktrade.opportunities.ranker import rank_product_opportunities

    with pytest.raises(ValueError, match="not found"):
        rank_product_opportunities(
            model=toy_model, data=toy_hetero_data, country="XXX", top_k=10,
        )


def test_rank_market_opportunities(toy_model, toy_hetero_data):
    from mktrade.opportunities.ranker import rank_market_opportunities

    df = rank_market_opportunities(
        model=toy_model,
        data=toy_hetero_data,
        country="MKD",
        top_k=50,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "hs4" in df.columns
    assert "partner_iso3" in df.columns
    assert "score" in df.columns


def test_ensemble_rankings(toy_model, toy_hetero_data):
    from mktrade.opportunities.ranker import rank_product_opportunities, ensemble_rankings

    gnn_df = rank_product_opportunities(
        model=toy_model, data=toy_hetero_data, country="MKD", top_k=10,
    )

    result = ensemble_rankings(gnn_df)
    assert "ensemble_score" in result.columns
    assert "ensemble_rank" in result.columns
    assert len(result) == len(gnn_df)


# ── Explainer tests ──

def test_explain_link(toy_model, toy_hetero_data):
    from mktrade.opportunities.explainer import explain_link

    explanation = explain_link(
        model=toy_model,
        data=toy_hetero_data,
        src_idx=1,  # MKD
        dst_idx=5,  # product not exported
    )

    assert "score" in explanation
    assert 0 <= explanation["score"] <= 1
    assert "src_importance" in explanation
    assert "dst_importance" in explanation
    assert "src_features" in explanation
    assert len(explanation["src_importance"]) > 0


def test_explain_top_opportunities(toy_model, toy_hetero_data):
    from mktrade.opportunities.ranker import rank_product_opportunities
    from mktrade.opportunities.explainer import explain_top_opportunities

    opp_df = rank_product_opportunities(
        model=toy_model, data=toy_hetero_data, country="MKD", top_k=5,
    )

    explanations = explain_top_opportunities(
        model=toy_model,
        data=toy_hetero_data,
        opportunities_df=opp_df,
        country="MKD",
        top_n=3,
    )

    assert isinstance(explanations, list)
    assert len(explanations) == 3
    assert "hs4" in explanations[0]


def test_extract_attention_weights(toy_model, toy_hetero_data):
    from mktrade.opportunities.explainer import extract_attention_weights

    # HomoEncoder + to_hetero doesn't have attention, should return empty
    weights = extract_attention_weights(toy_model, toy_hetero_data)
    assert isinstance(weights, dict)
    assert len(weights) == 0  # Not attention-based


# ── Visualization tests ──

def test_opportunity_bar_chart():
    from mktrade.viz.plots import opportunity_bar_chart

    df = pd.DataFrame({
        "hs4": ["0101", "0201", "8544", "8708", "3815"],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "rank": [1, 2, 3, 4, 5],
        "section": ["01", "02", "85", "87", "38"],
    })

    fig = opportunity_bar_chart(df, top_k=5)
    assert fig is not None
    assert len(fig.data) > 0


def test_model_comparison_heatmap():
    from mktrade.viz.plots import model_comparison_heatmap

    df = pd.DataFrame({
        "roc_auc": [0.72, 0.69, 0.65],
        "avg_precision": [0.70, 0.68, 0.65],
        "mrr": [0.001, 0.001, 0.001],
    }, index=["gat", "hgt", "density"])

    fig = model_comparison_heatmap(df)
    assert fig is not None


def test_feature_importance_chart():
    from mktrade.viz.plots import feature_importance_chart

    explanation = {
        "src_importance": {"eci": 0.3, "diversity": 0.2, "gdp_log": 0.15,
                           "cefta_member": 0.1, "eu_member": 0.05},
    }

    fig = feature_importance_chart(explanation, "country")
    assert fig is not None


def test_section_distribution_chart():
    from mktrade.viz.plots import section_distribution_chart

    df = pd.DataFrame({
        "hs4": ["0101", "0201", "8544", "8708", "3815"],
        "section": ["01", "02", "85", "87", "38"],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5],
    })

    fig = section_distribution_chart(df)
    assert fig is not None
