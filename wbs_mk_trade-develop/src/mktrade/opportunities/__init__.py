"""Opportunity discovery with lightweight, lazy public imports.

Keeping package import lazy allows feature engineering, target construction,
and reranker fitting on machines where the optional GNN runtime is unavailable.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "rank_product_opportunities",
    "rank_market_opportunities",
    "ensemble_rankings",
    "assemble_candidate_signals",
    "explain_link",
    "explain_top_opportunities",
    "extract_attention_weights",
    "OpportunityReranker",
    "SuccessCriteria",
]


def __getattr__(name: str) -> Any:
    if name in {
        "rank_product_opportunities",
        "rank_market_opportunities",
        "ensemble_rankings",
        "assemble_candidate_signals",
    }:
        from mktrade.opportunities import ranker

        return getattr(ranker, name)
    if name in {"explain_link", "explain_top_opportunities", "extract_attention_weights"}:
        from mktrade.opportunities import explainer

        return getattr(explainer, name)
    if name == "OpportunityReranker":
        from mktrade.opportunities.reranker import OpportunityReranker

        return OpportunityReranker
    if name == "SuccessCriteria":
        from mktrade.opportunities.targets import SuccessCriteria

        return SuccessCriteria
    raise AttributeError(name)
