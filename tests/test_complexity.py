"""Tests for complexity computation (uses synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_trade_matrix() -> pd.DataFrame:
    """Create a synthetic trade matrix for testing RCA/complexity."""
    # 3 countries, 4 products, 1 year
    np.random.seed(42)
    rows = []
    countries = ["AAA", "BBB", "CCC"]
    products = ["0101", "0201", "0301", "0401"]
    for c in countries:
        for p in products:
            val = np.random.exponential(1e6)
            rows.append({"year": 2020, "iso3": c, "hs4": p, "export_value": val})

    # Make AAA strongly specialised in 0101
    rows.append({"year": 2020, "iso3": "AAA", "hs4": "0101", "export_value": 1e9})
    return pd.DataFrame(rows)


def test_ecomplexity_runs() -> None:
    """ecomplexity should run on synthetic data without errors."""
    from mktrade.complexity.metrics import compute_complexity

    df = _make_trade_matrix()
    result = compute_complexity(df, year=2020)
    assert not result.empty
    assert "rca" in result.columns or "mcp" in result.columns


def test_rca_values_non_negative() -> None:
    """RCA values should be non-negative (NaN allowed for zero-trade pairs)."""
    from mktrade.complexity.metrics import compute_complexity

    df = _make_trade_matrix()
    result = compute_complexity(df, year=2020)
    if "rca" in result.columns:
        valid_rca = result["rca"].dropna()
        assert (valid_rca >= 0).all()


def test_mcp_is_binary_or_nan() -> None:
    """Mcp should be 0, 1, or NaN (for zero-trade pairs)."""
    from mktrade.complexity.metrics import compute_complexity

    df = _make_trade_matrix()
    result = compute_complexity(df, year=2020)
    if "mcp" in result.columns:
        valid_mcp = result["mcp"].dropna()
        assert set(valid_mcp.unique()).issubset({0, 1, 0.0, 1.0})
