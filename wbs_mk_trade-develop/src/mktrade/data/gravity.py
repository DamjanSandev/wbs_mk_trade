"""Download and process the CEPII Gravity dataset.

Provides country-pair covariates (distance, contiguity, common language,
colonial ties, FTA/RTA dummies, GDP, population) used as edge features
for Task B and as inputs to the PPML gravity baseline.

Source: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from mktrade.config import DataConfig

GRAVITY_URL = "http://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip"

# Only read the columns we actually need (the full CSV has 70+ columns, ~1.2 GB)
USECOLS = [
    "year", "iso3_o", "iso3_d", "dist", "contig", "comlang_off",
    "comcol", "col45", "fta_wto", "gdp_o", "gdp_d", "pop_o", "pop_d",
]


def download_gravity(cfg: DataConfig) -> Path:
    """Download CEPII Gravity ZIP to data/external/ (cached)."""
    out_path = cfg.external_dir / "gravity_v202211.csv"
    if out_path.exists():
        logger.info(f"Gravity data already cached: {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = cfg.external_dir / "gravity_v202211.zip"

    if not zip_path.exists():
        logger.info(f"Downloading CEPII Gravity dataset from {GRAVITY_URL} ...")
        resp = requests.get(GRAVITY_URL, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))

        with open(zip_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="CEPII Gravity") as pbar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Extract CSV from ZIP
    logger.info("Extracting gravity CSV from ZIP...")
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV found in gravity ZIP")
        csv_name = [n for n in csv_names if "gravity" in n.lower()]
        csv_name = csv_name[0] if csv_name else csv_names[0]
        with zf.open(csv_name) as src, open(out_path, "wb") as dst:
            dst.write(src.read())

    logger.info(f"Gravity CSV extracted: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def load_gravity(cfg: DataConfig) -> pd.DataFrame:
    """Load and filter the Gravity dataset for configured year range.

    Reads only the columns we need (USECOLS) to avoid OOM on the full 1.2 GB file.

    Returns columns: iso3_o, iso3_d, year, dist, contig, comlang_off,
    comcol, colony, fta_wto, gdp_o, gdp_d, pop_o, pop_d
    """
    cache_parquet = cfg.external_dir / "gravity_clean.parquet"
    if cache_parquet.exists():
        logger.info(f"Loading cached gravity parquet: {cache_parquet}")
        return pd.read_parquet(cache_parquet)

    path = download_gravity(cfg)

    grav_years = _load_yaml_gravity_years()
    logger.info(f"Loading gravity data (columns: {USECOLS}, years {grav_years[0]}-{grav_years[1]})...")

    # Read only needed columns to stay within memory
    df = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype={"iso3_o": str, "iso3_d": str},
        low_memory=True,
    )
    logger.info(f"Gravity raw subset: {len(df):,} rows")

    # Filter years
    df = df[(df["year"] >= grav_years[0]) & (df["year"] <= grav_years[1])].copy()

    # Rename col45 -> colony for clarity
    df = df.rename(columns={"col45": "colony"})

    # Coerce numerics
    for c in ["dist", "contig", "comlang_off", "colony", "comcol", "fta_wto",
              "gdp_o", "gdp_d", "pop_o", "pop_d"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.reset_index(drop=True)

    # Cache as parquet for future fast loads
    df.to_parquet(cache_parquet, index=False)

    logger.info(
        f"Gravity cleaned: {len(df):,} rows, "
        f"years {df['year'].min()}-{df['year'].max()}, "
        f"{df['iso3_o'].nunique()} origin countries"
    )
    return df


def _load_yaml_gravity_years() -> list[int]:
    """Get gravity year range from config YAML."""
    from mktrade.config import _load_yaml
    raw = _load_yaml("data.yaml")
    return raw.get("gravity", {}).get("year_range", [2000, 2020])
