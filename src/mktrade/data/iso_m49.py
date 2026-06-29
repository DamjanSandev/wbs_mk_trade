"""ISO-3166 alpha-3 <-> UN M49 numeric code mapping.

Comtrade uses M49 codes (e.g. North Macedonia = 807).
Atlas / CEPII / WDI use ISO3 alpha codes (MKD).
This module builds the bidirectional map from comtradeapicall reference data,
caches it to data/external/country_codes.parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

_CACHE: dict[str, pd.DataFrame] = {}


def build_iso_m49_map(cache_path: Path | None = None) -> pd.DataFrame:
    """Build a DataFrame with columns: iso3, m49, country_name, iso2.

    Loads from cache_path if it exists, otherwise fetches from comtradeapicall
    and saves to cache_path.
    """
    if "map" in _CACHE:
        return _CACHE["map"]

    if cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
        _CACHE["map"] = df
        logger.info(f"Loaded country code map from cache: {len(df)} entries")
        return df

    import comtradeapicall as ct

    logger.info("Fetching reporter reference table from Comtrade API...")
    reporters = ct.getReference("reporter")
    partners = ct.getReference("partner")

    # Normalise column names to lowercase for reliable access
    reporters.columns = reporters.columns.str.lower()
    partners.columns = partners.columns.str.lower()

    r = reporters[reporters["isgroup"] == False].copy()  # noqa: E712
    r = r.rename(columns={
        "reportercode": "m49",
        "reporterdesc": "country_name",
        "reportercodeisoalpha3": "iso3",
        "reportercodeisoalpha2": "iso2",
    })[["m49", "country_name", "iso3", "iso2"]].copy()

    p = partners[partners["isgroup"] == False].copy()  # noqa: E712
    p = p.rename(columns={
        "partnercode": "m49",
        "partnerdesc": "country_name",
        "partnercodeisoalpha3": "iso3",
        "partnercodeisoalpha2": "iso2",
    })[["m49", "country_name", "iso3", "iso2"]].copy()

    df = pd.concat([r, p], ignore_index=True).drop_duplicates(subset=["m49"])
    df["m49"] = df["m49"].astype(int)
    # Drop entries without ISO3 (groups, special areas)
    df = df[df["iso3"].notna() & (df["iso3"] != "")].reset_index(drop=True)

    # Drop expired/historical entries (e.g. West Germany 280) when a current one exists.
    # Keep the entry whose country_name does NOT contain "..." (historical marker).
    df["_is_historical"] = df["country_name"].str.contains(r"\.\.\.", na=False)
    # For each iso3 with multiple m49 codes, prefer the non-historical one
    df = df.sort_values(["iso3", "_is_historical"]).drop_duplicates(subset=["iso3"], keep="first")
    df = df.drop(columns=["_is_historical"]).reset_index(drop=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached country code map ({len(df)} entries) -> {cache_path}")

    _CACHE["map"] = df
    return df


def _get_map(cache_path: Path | None = None) -> pd.DataFrame:
    return build_iso_m49_map(cache_path)


def iso3_to_m49(iso3: str, cache_path: Path | None = None) -> int:
    """Convert ISO-3 alpha code to M49 numeric code."""
    df = _get_map(cache_path)
    matches = df.loc[df["iso3"] == iso3.upper(), "m49"]
    if matches.empty:
        raise KeyError(f"No M49 code found for ISO3={iso3!r}")
    return int(matches.iloc[0])


def m49_to_iso3(m49: int, cache_path: Path | None = None) -> str:
    """Convert M49 numeric code to ISO-3 alpha code."""
    df = _get_map(cache_path)
    matches = df.loc[df["m49"] == m49, "iso3"]
    if matches.empty:
        raise KeyError(f"No ISO3 code found for M49={m49}")
    return str(matches.iloc[0])


def iso3_to_name(iso3: str, cache_path: Path | None = None) -> str:
    """Get country name from ISO3 code."""
    df = _get_map(cache_path)
    matches = df.loc[df["iso3"] == iso3.upper(), "country_name"]
    if matches.empty:
        raise KeyError(f"No name found for ISO3={iso3!r}")
    return str(matches.iloc[0])
