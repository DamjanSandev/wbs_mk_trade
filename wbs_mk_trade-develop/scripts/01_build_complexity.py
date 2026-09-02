"""Phase 1b — Compute economic complexity metrics.

Feeds the cleaned Atlas country x product x year export table into the
Harvard `ecomplexity` package to produce:
  - RCA, Mcp (binary comparative advantage)
  - ECI (per country), PCI (per product)
  - Density (omega_cp), COI, COG
  - Diversity, Ubiquity
  - Product proximity matrix (phi_ij)

Outputs to data/processed/:
  - complexity.parquet (full table)
  - country_complexity.parquet (ECI, diversity per country-year)
  - product_complexity.parquet (PCI, ubiquity per product-year)
  - density_matrix.parquet (country x product density)
  - proximity_matrix.parquet (product x product proximity)

Usage:
    python scripts/01_build_complexity.py
    python scripts/01_build_complexity.py --year 2021    # single year only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from loguru import logger

from mktrade.config import load_data_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute economic complexity metrics")
    p.add_argument("--year", type=int, default=None, help="Compute for a single year only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_data_config()

    # Load cleaned Atlas exports
    atlas_path = cfg.processed_dir / "atlas_exports.parquet"
    if not atlas_path.exists():
        logger.error(f"Atlas exports not found at {atlas_path}. Run 00_download_data.py first.")
        sys.exit(1)

    atlas_df = pd.read_parquet(atlas_path)
    logger.info(f"Loaded Atlas exports: {len(atlas_df):,} rows")

    from mktrade.complexity.metrics import (
        compute_complexity,
        compute_proximity_matrix,
        save_complexity_outputs,
    )

    # Compute complexity (for single year or all years)
    if args.year:
        logger.info(f"Computing complexity for year {args.year} only")
        complexity_df = compute_complexity(atlas_df, year=args.year)
    else:
        # Compute year by year (ecomplexity works best per-year)
        years = sorted(atlas_df["year"].unique())
        logger.info(f"Computing complexity for {len(years)} years: {years[0]}..{years[-1]}")
        dfs = []
        for y in years:
            logger.info(f"  Year {y}...")
            df_y = compute_complexity(atlas_df, year=y)
            if not df_y.empty:
                dfs.append(df_y)
        complexity_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if complexity_df.empty:
        logger.error("Complexity computation returned empty results!")
        sys.exit(1)

    logger.info(f"Complexity output: {len(complexity_df):,} rows, cols: {complexity_df.columns.tolist()}")

    # Compute proximity for the latest available year
    latest_year = int(atlas_df["year"].max())
    logger.info(f"\nComputing proximity matrix for year {latest_year}")
    proximity_df = compute_proximity_matrix(atlas_df, year=latest_year)

    # Save all outputs
    save_complexity_outputs(complexity_df, proximity_df, cfg.processed_dir)

    # ── Quick MKD summary ─────────────────────────────────
    mkd = complexity_df[complexity_df["iso3"] == "MKD"].copy()
    if not mkd.empty:
        latest_mkd = mkd[mkd["year"] == mkd["year"].max()]

        logger.info(f"\n{'='*60}")
        logger.info(f"MKD COMPLEXITY SUMMARY — Year {latest_mkd['year'].iloc[0]}")
        logger.info(f"{'='*60}")

        if "eci" in latest_mkd.columns:
            eci = latest_mkd["eci"].iloc[0]
            logger.info(f"ECI: {eci:.4f}")

        if "diversity" in latest_mkd.columns:
            div = latest_mkd["diversity"].iloc[0]
            logger.info(f"Diversity (# products with RCA>1): {div:.0f}")

        if "mcp" in latest_mkd.columns:
            n_rca = latest_mkd["mcp"].sum()
            logger.info(f"Products with RCA > 1: {n_rca:.0f}")

        if "density" in latest_mkd.columns:
            top_density = (
                latest_mkd[latest_mkd["mcp"] == 0]
                .nlargest(10, "density")[["hs4", "density", "pci"]]
            )
            if not top_density.empty:
                logger.info(f"\nTop 10 products by density (NOT yet exported by MKD):")
                for _, row in top_density.iterrows():
                    logger.info(
                        f"  {row['hs4']}: density={row['density']:.4f}, PCI={row.get('pci', 'N/A')}"
                    )

    logger.info("\n>>> Phase 1b (complexity) COMPLETE")


if __name__ == "__main__":
    main()
