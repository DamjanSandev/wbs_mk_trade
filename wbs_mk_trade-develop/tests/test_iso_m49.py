"""Tests for ISO3 <-> M49 mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from mktrade.data.iso_m49 import build_iso_m49_map, iso3_to_m49, m49_to_iso3


@pytest.fixture(scope="module")
def code_map() -> pd.DataFrame:
    """Load the country code map (hits network on first run, cached after)."""
    from mktrade.config import load_data_config

    cfg = load_data_config()
    return build_iso_m49_map(cfg.external_dir / "country_codes.parquet")


def test_map_has_rows(code_map: pd.DataFrame) -> None:
    assert len(code_map) > 100, "Expected at least 100 country codes"


def test_map_has_required_columns(code_map: pd.DataFrame) -> None:
    for col in ["iso3", "m49", "country_name"]:
        assert col in code_map.columns, f"Missing column: {col}"


def test_mkd_is_807(code_map: pd.DataFrame) -> None:
    """North Macedonia should map to M49=807."""
    assert iso3_to_m49("MKD", None) == 807


def test_807_is_mkd(code_map: pd.DataFrame) -> None:
    assert m49_to_iso3(807, None) == "MKD"


def test_common_countries(code_map: pd.DataFrame) -> None:
    """Spot-check a few well-known mappings."""
    assert iso3_to_m49("ALB", None) == 8
    assert iso3_to_m49("SRB", None) == 688
    assert iso3_to_m49("DEU", None) == 276


def test_unknown_iso3_raises() -> None:
    with pytest.raises(KeyError):
        iso3_to_m49("ZZZ")


def test_unknown_m49_raises() -> None:
    with pytest.raises(KeyError):
        m49_to_iso3(999999)
