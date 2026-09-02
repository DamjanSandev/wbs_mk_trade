"""Compute economic-complexity metrics using the ecomplexity package.

Wraps ecomplexity.ecomplexity() and ecomplexity.proximity() to produce:
- RCA, Mcp (binary comparative advantage matrix)
- ECI (Economic Complexity Index per country)
- PCI (Product Complexity Index per product)
- Density (omega_cp — how close a product is to a country's capabilities)
- COI / COG (Complexity Outlook Index / Gain)
- Diversity, Ubiquity
- Product-product proximity matrix (phi_ij)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ecomplexity import ecomplexity, proximity
from loguru import logger


def compute_complexity(
    trade_df: pd.DataFrame,
    year: int | None = None,
) -> pd.DataFrame:
    """Run the ecomplexity pipeline on a country x product x value matrix.

    Parameters
    ----------
    trade_df : DataFrame with columns [year, iso3, hs4, export_value].
    year : If given, filter to this year before computing.

    Returns
    -------
    DataFrame with ecomplexity output columns including:
        year, iso3, hs4, export_value, rca, mcp, eci, pci,
        density, coi, cog, diversity, ubiquity.
    """
    # Feed ecomplexity ONLY the 4 required columns to avoid merge collisions
    # (the Atlas DataFrame has pre-computed pci/density/cog/eci which would cause _x/_y suffixes)
    df = trade_df[["year", "iso3", "hs4", "export_value"]].copy()
    if year is not None:
        df = df[df["year"] == year].copy()

    if df.empty:
        logger.warning("Empty input to compute_complexity")
        return pd.DataFrame()

    # ecomplexity expects columns: time, loc, prod, val
    cols_input = {
        "time": "year",
        "loc": "iso3",
        "prod": "hs4",
        "val": "export_value",
    }

    logger.info(
        f"Computing complexity: {df['iso3'].nunique()} countries, "
        f"{df['hs4'].nunique()} products, "
        f"years {df['year'].min()}-{df['year'].max()}"
    )

    result = ecomplexity(
        data=df,
        cols_input=cols_input,
        presence_test="rca",
        val_errors_flag="coerce",
        rca_mcp_threshold=1,
        verbose=True,
    )

    logger.info(f"Complexity computed: {len(result):,} rows, columns: {result.columns.tolist()}")
    return result


def compute_proximity_matrix(
    trade_df: pd.DataFrame,
    year: int | None = None,
) -> pd.DataFrame:
    """Compute the product-product proximity matrix.

    Returns a DataFrame in edge-list format: hs4_i, hs4_j, proximity.
    """
    df = trade_df[["year", "iso3", "hs4", "export_value"]].copy()
    if year is not None:
        df = df[df["year"] == year].copy()

    cols_input = {
        "time": "year",
        "loc": "iso3",
        "prod": "hs4",
        "val": "export_value",
    }

    logger.info("Computing proximity matrix...")
    prox = proximity(
        data=df,
        cols_input=cols_input,
        presence_test="rca",
        val_errors_flag="coerce",
        rca_mcp_threshold=1,
    )

    logger.info(f"Proximity matrix: {len(prox):,} product-product pairs")
    return prox


def get_country_complexity(complexity_df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-country ECI and diversity (one row per country per year)."""
    cols = ["year", "iso3"]
    if "eci" in complexity_df.columns:
        cols.append("eci")
    if "diversity" in complexity_df.columns:
        cols.append("diversity")

    # ECI and diversity are constant per country-year, so just take first
    out = complexity_df[cols].drop_duplicates(subset=["year", "iso3"]).reset_index(drop=True)
    logger.info(f"Country complexity: {len(out)} country-year rows")
    return out


def get_product_complexity(complexity_df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-product PCI and ubiquity (one row per product per year)."""
    cols = ["year", "hs4"]
    if "pci" in complexity_df.columns:
        cols.append("pci")
    if "ubiquity" in complexity_df.columns:
        cols.append("ubiquity")

    out = complexity_df[cols].drop_duplicates(subset=["year", "hs4"]).reset_index(drop=True)
    logger.info(f"Product complexity: {len(out)} product-year rows")
    return out


def get_density_matrix(complexity_df: pd.DataFrame) -> pd.DataFrame:
    """Extract country x product density values (omega_cp)."""
    cols = ["year", "iso3", "hs4"]
    if "density" in complexity_df.columns:
        cols.append("density")
    else:
        logger.warning("No 'density' column in complexity output")
        return pd.DataFrame()

    return complexity_df[cols].copy().reset_index(drop=True)


def save_complexity_outputs(
    complexity_df: pd.DataFrame,
    proximity_df: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    """Persist complexity outputs to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full complexity table
    complexity_df.to_parquet(output_dir / "complexity.parquet", index=False)
    logger.info(f"Saved complexity table: {len(complexity_df):,} rows")

    # Country-level
    country = get_country_complexity(complexity_df)
    country.to_parquet(output_dir / "country_complexity.parquet", index=False)

    # Product-level
    product = get_product_complexity(complexity_df)
    product.to_parquet(output_dir / "product_complexity.parquet", index=False)

    # Density matrix
    density = get_density_matrix(complexity_df)
    if not density.empty:
        density.to_parquet(output_dir / "density_matrix.parquet", index=False)

    # Proximity matrix
    if proximity_df is not None and not proximity_df.empty:
        proximity_df.to_parquet(output_dir / "proximity_matrix.parquet", index=False)
        logger.info(f"Saved proximity matrix: {len(proximity_df):,} pairs")
