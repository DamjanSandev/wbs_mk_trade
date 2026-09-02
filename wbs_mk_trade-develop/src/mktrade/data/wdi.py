"""Download World Development Indicators from the World Bank API.

Supplements country node features (GDP, GDP per capita, population).
Uses the World Bank REST API v2 (JSON format).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from mktrade.config import DataConfig

WB_API_BASE = "https://api.worldbank.org/v2"


def download_wdi(cfg: DataConfig) -> pd.DataFrame:
    """Fetch WDI indicators for all countries over the configured year range.

    Caches to data/external/wdi_indicators.parquet.
    Returns columns: iso3, year, gdp, gdp_pc, population
    """
    cache_path = cfg.external_dir / "wdi_indicators.parquet"
    if cache_path.exists():
        logger.info(f"WDI data cached: {cache_path}")
        return pd.read_parquet(cache_path)

    from mktrade.config import _load_yaml
    raw = _load_yaml("data.yaml")
    indicators = raw.get("wdi", {}).get("indicators", {})
    year_range = raw.get("wdi", {}).get("year_range", [2000, 2022])

    all_dfs = []
    for label, indicator_code in indicators.items():
        logger.info(f"Fetching WDI indicator: {label} ({indicator_code})")
        url = (
            f"{WB_API_BASE}/country/all/indicator/{indicator_code}"
            f"?date={year_range[0]}:{year_range[1]}"
            f"&format=json&per_page=20000"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if len(data) < 2 or data[1] is None:
            logger.warning(f"No data returned for {indicator_code}")
            continue

        records = []
        for entry in data[1]:
            iso3 = entry.get("countryiso3code", "")
            year = entry.get("date", "")
            value = entry.get("value")
            if iso3 and year and value is not None:
                records.append({"iso3": iso3, "year": int(year), label: float(value)})

        df = pd.DataFrame(records)
        all_dfs.append(df)
        logger.info(f"  {label}: {len(df):,} records")

    if not all_dfs:
        return pd.DataFrame()

    # Merge all indicators
    result = all_dfs[0]
    for df in all_dfs[1:]:
        result = result.merge(df, on=["iso3", "year"], how="outer")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    logger.info(f"WDI data saved: {cache_path} ({len(result):,} rows)")
    return result
