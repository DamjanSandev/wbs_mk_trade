"""Tests for data cleaning assertions."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_atlas_sample() -> pd.DataFrame:
    """Create a small synthetic Atlas-like DataFrame (Dataverse format)."""
    return pd.DataFrame({
        "year": [2020, 2020, 2020, 2020, 2020],
        "country_iso3_code": ["MKD", "MKD", "DEU", "WLD", "MKD"],
        "product_hs92_code": ["7202", "8544", "8703", "0101", "XXXX"],
        "export_value": [500e6, 300e6, 1e9, 5e9, 100],
        "export_rca": [3.5, 1.2, 0.8, 0.1, 0.0],
    })


def test_clean_atlas_drops_aggregates() -> None:
    """WLD and non-3-char codes should be dropped."""
    from mktrade.data.atlas import clean_atlas
    from mktrade.config import load_data_config

    cfg = load_data_config()
    df = _make_atlas_sample()
    cleaned = clean_atlas(df, cfg)

    assert "WLD" not in cleaned["iso3"].values
    assert all(cleaned["iso3"].str.len() == 3)


def test_clean_atlas_drops_bad_hs_codes() -> None:
    """Non-4-digit HS codes should be dropped."""
    from mktrade.data.atlas import clean_atlas
    from mktrade.config import load_data_config

    cfg = load_data_config()
    df = _make_atlas_sample()
    cleaned = clean_atlas(df, cfg)

    assert all(cleaned["hs4"].str.match(r"^\d{4}$"))


def test_clean_atlas_output_schema() -> None:
    """Cleaned output should have the expected core columns."""
    from mktrade.data.atlas import clean_atlas
    from mktrade.config import load_data_config

    cfg = load_data_config()
    df = _make_atlas_sample()
    cleaned = clean_atlas(df, cfg)

    assert "year" in cleaned.columns
    assert "iso3" in cleaned.columns
    assert "hs4" in cleaned.columns
    assert "export_value" in cleaned.columns


def test_export_values_non_negative() -> None:
    """All export values should be >= 0 after cleaning."""
    from mktrade.data.atlas import clean_atlas
    from mktrade.config import load_data_config

    cfg = load_data_config()
    df = _make_atlas_sample()
    cleaned = clean_atlas(df, cfg)

    assert (cleaned["export_value"] >= 0).all()
