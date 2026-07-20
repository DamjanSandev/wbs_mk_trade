"""Opportunity discovery — rank and explain predicted links for MKD."""

from mktrade.opportunities.ranker import (
    rank_product_opportunities,
    rank_market_opportunities,
    ensemble_rankings,
)
from mktrade.opportunities.explainer import (
    explain_link,
    explain_top_opportunities,
    extract_attention_weights,
)

__all__ = [
    "rank_product_opportunities",
    "rank_market_opportunities",
    "ensemble_rankings",
    "explain_link",
    "explain_top_opportunities",
    "extract_attention_weights",
]
