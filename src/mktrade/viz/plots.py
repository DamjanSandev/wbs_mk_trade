"""Reusable Plotly chart builders for the dashboard and reports."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def product_space_network(
    proximity_df: pd.DataFrame,
    highlight_products: list[str] | None = None,
) -> go.Figure:
    """Interactive product-space network coloured by HS section."""
    raise NotImplementedError


def opportunity_bar_chart(
    opportunities: pd.DataFrame,
    top_k: int = 20,
    title: str = "Top predicted opportunities",
) -> go.Figure:
    """Horizontal bar chart of top-K opportunity scores."""
    raise NotImplementedError


def model_comparison_heatmap(comparison_df: pd.DataFrame) -> go.Figure:
    """Heatmap of models (rows) x metrics (columns)."""
    raise NotImplementedError


def temporal_validation_chart(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> go.Figure:
    """Show how many top-K predictions appeared in held-out years."""
    raise NotImplementedError
