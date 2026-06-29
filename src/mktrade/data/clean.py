"""Harmonise and clean all data sources into tidy parquets.

Steps:
1. Load Atlas data → standardise to (year, iso3, hs4, export_value).
2. Load Comtrade bilateral → standardise to (year, reporter_iso3, partner_iso3, hs4, flow, value).
3. Drop aggregate reporters (World, EU, unspecified, groups).
4. Coerce values to numeric USD, handle missing.
5. Decision: use REPORTED EXPORTS (not mirror imports) as the primary measure.
   Documented rationale: direct reporting is more reliable for the exporter's own data.
6. Output tidy parquets to data/processed/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from mktrade.config import DataConfig
from mktrade.data.iso_m49 import build_iso_m49_map, m49_to_iso3

# Known aggregate/group M49 codes to exclude
AGGREGATE_M49 = {0, 97, 290, 473, 490, 527, 568, 577, 636, 637, 838, 839, 899}
# ISO3 codes for aggregates to exclude
AGGREGATE_ISO3 = {"WLD", "EUN", "EUR", "OAS", "AFR", "AMR", "OCE", "ANT"}


def clean_atlas_exports(cfg: DataConfig) -> pd.DataFrame:
    """Load, clean, and save Atlas export data.

    Returns DataFrame: year, iso3, hs4, export_value
    Saves to data/processed/atlas_exports.parquet
    """
    from mktrade.data.atlas import clean_atlas, load_atlas_raw

    df = load_atlas_raw(cfg)
    df = clean_atlas(df, cfg)

    out_path = cfg.processed_dir / "atlas_exports.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved clean Atlas exports: {out_path} ({len(df):,} rows)")
    return df


def clean_comtrade_bilateral(cfg: DataConfig) -> pd.DataFrame:
    """Load, clean, and save Comtrade bilateral data.

    Returns DataFrame: year, reporter_iso3, partner_iso3, hs4, flow, value
    Saves to data/processed/bilateral_flows.parquet
    """
    from mktrade.data.comtrade import load_comtrade_cached

    code_map = build_iso_m49_map(cfg.external_dir / "country_codes.parquet")

    df = load_comtrade_cached(cfg, task="b")
    if df.empty:
        logger.warning("No Comtrade bilateral data found in cache")
        return df

    logger.info(f"Raw Comtrade bilateral: {len(df):,} rows, columns: {df.columns.tolist()[:15]}")

    # Standardise column names (Comtrade API returns camelCase)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "period":
            rename[c] = "year"
        elif cl == "reportercode":
            rename[c] = "reporter_m49"
        elif cl == "partnercode":
            rename[c] = "partner_m49"
        elif cl == "cmdcode":
            rename[c] = "hs_code"
        elif cl == "flowcode":
            rename[c] = "flow"
        elif cl == "primaryvalue":
            rename[c] = "value"
        elif cl == "fobvalue":
            rename[c] = "fob_value"
        elif cl == "cifvalue":
            rename[c] = "cif_value"

    df = df.rename(columns=rename)

    # Use primaryValue as the main value column
    if "value" not in df.columns:
        # Fallback to fob for exports, cif for imports
        if "fob_value" in df.columns:
            df["value"] = df["fob_value"]
        elif "cif_value" in df.columns:
            df["value"] = df["cif_value"]

    # Ensure required columns
    required = ["year", "reporter_m49", "partner_m49", "hs_code", "flow", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns after rename: {missing}. Available: {df.columns.tolist()}")
        return pd.DataFrame()

    # Coerce year to int
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Drop aggregate reporters/partners
    df["reporter_m49"] = pd.to_numeric(df["reporter_m49"], errors="coerce")
    df["partner_m49"] = pd.to_numeric(df["partner_m49"], errors="coerce")
    df = df[~df["reporter_m49"].isin(AGGREGATE_M49)].copy()
    df = df[~df["partner_m49"].isin(AGGREGATE_M49)].copy()
    df = df.dropna(subset=["reporter_m49", "partner_m49"])

    # Map M49 → ISO3
    m49_lookup = code_map.set_index("m49")["iso3"].to_dict()
    df["reporter_iso3"] = df["reporter_m49"].astype(int).map(m49_lookup)
    df["partner_iso3"] = df["partner_m49"].astype(int).map(m49_lookup)

    # Drop rows where mapping failed
    unmapped_r = df["reporter_iso3"].isna().sum()
    unmapped_p = df["partner_iso3"].isna().sum()
    if unmapped_r > 0:
        logger.warning(f"Dropped {unmapped_r} rows with unmapped reporter M49 codes")
    if unmapped_p > 0:
        logger.warning(f"Dropped {unmapped_p} rows with unmapped partner M49 codes")
    df = df.dropna(subset=["reporter_iso3", "partner_iso3"])

    # Drop aggregates by ISO3 too
    df = df[~df["reporter_iso3"].isin(AGGREGATE_ISO3)]
    df = df[~df["partner_iso3"].isin(AGGREGATE_ISO3)]

    # Standardise HS code
    df["hs4"] = df["hs_code"].astype(str).str.zfill(4)
    df = df[df["hs4"].str.match(r"^\d{4}$")].copy()

    # Coerce value
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    # Keep columns
    out = df[["year", "reporter_iso3", "partner_iso3", "hs4", "flow", "value"]].reset_index(drop=True)

    out_path = cfg.processed_dir / "bilateral_flows.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info(
        f"Saved clean bilateral flows: {out_path} ({len(out):,} rows, "
        f"{out['reporter_iso3'].nunique()} reporters, "
        f"{out['partner_iso3'].nunique()} partners)"
    )
    return out


def clean_comtrade_task_a(cfg: DataConfig) -> pd.DataFrame:
    """Clean Comtrade aggregate (partner=World) data for supplementing Atlas.

    Returns DataFrame: year, iso3, hs4, export_value
    """
    from mktrade.data.comtrade import load_comtrade_cached

    df = load_comtrade_cached(cfg, task="a")
    if df.empty:
        logger.warning("No Comtrade Task A data in cache")
        return df

    # Standardise
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "period":
            rename[c] = "year"
        elif cl == "reportercode":
            rename[c] = "reporter_m49"
        elif cl == "cmdcode":
            rename[c] = "hs_code"
        elif cl == "primaryvalue":
            rename[c] = "export_value"
        elif cl == "fobvalue" and "primaryvalue" not in [x.lower() for x in df.columns]:
            rename[c] = "export_value"

    df = df.rename(columns=rename)

    code_map = build_iso_m49_map(cfg.external_dir / "country_codes.parquet")
    m49_lookup = code_map.set_index("m49")["iso3"].to_dict()

    if "reporter_m49" in df.columns:
        df["reporter_m49"] = pd.to_numeric(df["reporter_m49"], errors="coerce").astype("Int64")
        df["iso3"] = df["reporter_m49"].map(m49_lookup)
    elif "reporteriso" in [c.lower() for c in df.columns]:
        iso_col = [c for c in df.columns if c.lower() == "reporteriso"][0]
        df["iso3"] = df[iso_col]

    if "export_value" not in df.columns:
        val_cols = [c for c in df.columns if "value" in c.lower()]
        if val_cols:
            df["export_value"] = pd.to_numeric(df[val_cols[0]], errors="coerce").fillna(0)

    df["hs4"] = df["hs_code"].astype(str).str.zfill(4) if "hs_code" in df.columns else ""
    df = df[df["hs4"].str.match(r"^\d{4}$")].copy()
    df["export_value"] = pd.to_numeric(df.get("export_value", 0), errors="coerce").fillna(0)

    out = df[["year", "iso3", "hs4", "export_value"]].dropna().reset_index(drop=True)
    logger.info(f"Comtrade Task A cleaned: {len(out):,} rows")
    return out


def validate_mkd_exports(
    atlas_df: pd.DataFrame,
    bilateral_df: pd.DataFrame | None = None,
    latest_year: int | None = None,
) -> dict:
    """Validate MKD data against known facts.

    Checks:
    - Top 10 export products
    - Top 10 destination markets (if bilateral available)
    - Total export value vs known ~$8B benchmark
    """
    results: dict = {}

    mkd = atlas_df[atlas_df["iso3"] == "MKD"].copy()
    if mkd.empty:
        logger.error("No MKD data in atlas!")
        return {"error": "No MKD data"}

    if latest_year is None:
        latest_year = int(mkd["year"].max())

    mkd_latest = mkd[mkd["year"] == latest_year].copy()

    # Top 10 products by export value
    top_products = (
        mkd_latest.groupby("hs4")["export_value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    results["top_10_products"] = top_products
    results["latest_year"] = latest_year

    # Total exports
    total_exports = mkd_latest["export_value"].sum()
    results["total_exports_usd"] = total_exports

    # Known benchmark: MKD exports ~$7-9B in recent years
    benchmark = 8_000_000_000
    gap_pct = (total_exports - benchmark) / benchmark * 100
    results["benchmark_gap_pct"] = gap_pct

    logger.info(f"\n{'='*60}")
    logger.info(f"MKD VALIDATION — Year {latest_year}")
    logger.info(f"{'='*60}")
    logger.info(f"Total exports: ${total_exports/1e9:.2f}B (benchmark: ~$8B, gap: {gap_pct:+.1f}%)")
    logger.info(f"\nTop 10 export products (HS4):")
    for hs4, val in top_products.items():
        logger.info(f"  {hs4}: ${val/1e6:,.1f}M")

    # Top markets from bilateral
    if bilateral_df is not None and not bilateral_df.empty:
        mkd_bil = bilateral_df[
            (bilateral_df["reporter_iso3"] == "MKD")
            & (bilateral_df["flow"] == "X")
            & (bilateral_df["year"] == latest_year)
        ]
        if not mkd_bil.empty:
            top_markets = (
                mkd_bil.groupby("partner_iso3")["value"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            results["top_10_markets"] = top_markets
            logger.info(f"\nTop 10 destination markets:")
            for iso3, val in top_markets.items():
                logger.info(f"  {iso3}: ${val/1e6:,.1f}M")

    return results
