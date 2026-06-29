"""Download and clean Harvard Growth Lab Atlas export data.

The Atlas provides the primary country x HS-product x year export values
used for Task A (product diversification). The public data file covers
~1995-2024 at HS 4-digit (HS92 classification).

Source: Harvard Dataverse doi:10.7910/DVN/T4CHWJ
File: hs92_country_product_year_4.csv (file ID 13685110, ~452 MB)

The Atlas pre-computes: export_rca, eci, pci, distance (=density), cog, coi, diversity.
We still run ecomplexity independently for the proximity matrix and for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from mktrade.config import DataConfig

# Harvard Dataverse direct-access URL for hs92_country_product_year_4.csv
ATLAS_DATAVERSE_FILE_ID = "13685110"
ATLAS_URL = f"https://dataverse.harvard.edu/api/access/datafile/{ATLAS_DATAVERSE_FILE_ID}"

# Data dictionary file
ATLAS_DICT_FILE_ID = "13685113"
ATLAS_DICT_URL = f"https://dataverse.harvard.edu/api/access/datafile/{ATLAS_DICT_FILE_ID}"


def download_atlas(cfg: DataConfig) -> Path:
    """Download the Atlas HS-4 country-product-year CSV to data/raw/ (cached).

    Returns the path to the downloaded file.
    """
    out_path = cfg.raw_dir / "hs92_country_product_year_4.csv"
    if out_path.exists():
        logger.info(f"Atlas data already cached: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Atlas HS4 data from Harvard Dataverse (file {ATLAS_DATAVERSE_FILE_ID})...")

    resp = requests.get(ATLAS_URL, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(out_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="Atlas HS4") as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Atlas data saved: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")

    # Also download data dictionary
    dict_path = cfg.raw_dir / "hs92_data_dictionary.csv"
    if not dict_path.exists():
        r = requests.get(ATLAS_DICT_URL, timeout=30)
        r.raise_for_status()
        dict_path.write_bytes(r.content)
        logger.info(f"Data dictionary saved: {dict_path}")

    return out_path


def load_atlas_raw(cfg: DataConfig) -> pd.DataFrame:
    """Load raw Atlas CSV into a DataFrame.

    Columns from Dataverse: country_id, country_iso3_code, product_id,
    product_hs92_code, year, export_value, import_value, global_market_share,
    export_rca, eci, distance, cog, coi, diversity, pci, ...
    """
    path = cfg.raw_dir / "hs92_country_product_year_4.csv"
    if not path.exists():
        path = download_atlas(cfg)

    logger.info(f"Loading Atlas raw data from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Atlas raw: {len(df):,} rows, columns: {df.columns.tolist()}")
    return df


def clean_atlas(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Filter years, standardise columns.

    Returns a clean DataFrame with columns:
        year, iso3, hs4, export_value, import_value,
        export_rca, eci, pci, density, cog, coi, diversity
    """
    # Rename to our standard names
    rename_map = {
        "country_iso3_code": "iso3",
        "product_hs92_code": "hs4",
        "distance": "density",  # Atlas calls density "distance"
    }
    # Only rename columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Filter to configured year range
    df = df[(df["year"] >= cfg.year_start) & (df["year"] <= cfg.year_end)].copy()

    # Drop rows without a proper ISO3 code (3 chars) and known aggregates
    _AGGREGATES = {"WLD", "EUN", "EUR", "OAS", "AFR", "AMR", "OCE", "ANT", "XXB", "XXX"}
    df = df[df["iso3"].astype(str).str.len() == 3].copy()
    df = df[~df["iso3"].isin(_AGGREGATES)].copy()

    # Pad HS code to 4 digits
    df["hs4"] = df["hs4"].astype(str).str.zfill(4)
    # Keep only valid 4-digit HS codes
    df = df[df["hs4"].str.match(r"^\d{4}$")].copy()

    # Coerce numeric columns
    numeric_cols = ["export_value", "import_value", "export_rca", "eci", "pci",
                    "density", "cog", "coi", "diversity"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing export values with 0
    df["export_value"] = df["export_value"].fillna(0)

    # Keep relevant columns
    keep_cols = ["year", "iso3", "hs4"] + [c for c in numeric_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    logger.info(
        f"Atlas cleaned: {len(df):,} rows, "
        f"{df['iso3'].nunique()} countries, "
        f"{df['hs4'].nunique()} products, "
        f"years {df['year'].min()}-{df['year'].max()}"
    )
    return df
