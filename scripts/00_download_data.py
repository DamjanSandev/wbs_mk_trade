"""Phase 1a — Download all raw data sources and clean into processed parquets.

Downloads:
  1. Country code map (ISO3 <-> M49) from Comtrade reference tables.
  2. Harvard Growth Lab Atlas HS-4 exports → data/raw/
  3. UN Comtrade Task A (aggregate) + Task B (bilateral) → data/raw/comtrade/
  4. CEPII Gravity dataset → data/external/
  5. World Bank WDI indicators → data/external/

Then cleans and harmonises into data/processed/:
  - atlas_exports.parquet
  - bilateral_flows.parquet

Finally runs validation checks on MKD data.

Usage:
    python scripts/00_download_data.py
    python scripts/00_download_data.py --skip-comtrade   # skip API calls
    python scripts/00_download_data.py --skip-atlas      # skip Atlas download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from mktrade.config import load_data_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and clean all data sources")
    p.add_argument("--skip-comtrade", action="store_true", help="Skip Comtrade API calls")
    p.add_argument("--skip-atlas", action="store_true", help="Skip Atlas download")
    p.add_argument("--skip-gravity", action="store_true", help="Skip CEPII Gravity download")
    p.add_argument("--skip-wdi", action="store_true", help="Skip WDI download")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_data_config()

    logger.info(f"Data config: HS{cfg.hs_level}, years {cfg.year_start}-{cfg.year_end}")
    logger.info(f"Focus country: {cfg.focus_country}")
    logger.info(f"Comtrade key: {'SET' if cfg.comtrade_api_key else 'NOT SET'}")

    # ── Step 1: Build country code map ────────────────────
    logger.info("\n>>> Step 1: Building ISO3 <-> M49 country code map")
    from mktrade.data.iso_m49 import build_iso_m49_map

    code_map = build_iso_m49_map(cfg.external_dir / "country_codes.parquet")
    logger.info(f"Country code map: {len(code_map)} entries")

    # ── Step 2: Download Atlas ────────────────────────────
    atlas_df = None
    if not args.skip_atlas:
        logger.info("\n>>> Step 2: Downloading Harvard Growth Lab Atlas data")
        from mktrade.data.atlas import download_atlas

        download_atlas(cfg)

        logger.info(">>> Cleaning Atlas data")
        from mktrade.data.clean import clean_atlas_exports

        atlas_df = clean_atlas_exports(cfg)
    else:
        logger.info(">>> Step 2: SKIPPED (--skip-atlas)")
        processed = cfg.processed_dir / "atlas_exports.parquet"
        if processed.exists():
            import pandas as pd
            atlas_df = pd.read_parquet(processed)
            logger.info(f"Loaded existing Atlas: {len(atlas_df):,} rows")

    # ── Step 3: Download Comtrade ─────────────────────────
    bilateral_df = None
    if not args.skip_comtrade and cfg.comtrade_api_key:
        logger.info("\n>>> Step 3: Downloading UN Comtrade data")
        from mktrade.data.comtrade import download_task_a, download_task_b, get_session_stats

        logger.info(">>> Task A (aggregate exports, partner=World)")
        download_task_a(cfg)

        logger.info(">>> Task B (bilateral flows)")
        download_task_b(cfg)

        stats = get_session_stats()
        logger.info(f"Comtrade API stats: {stats}")

        logger.info(">>> Cleaning Comtrade bilateral data")
        from mktrade.data.clean import clean_comtrade_bilateral

        bilateral_df = clean_comtrade_bilateral(cfg)
    elif not cfg.comtrade_api_key:
        logger.warning(">>> Step 3: SKIPPED (no COMTRADE_KEY in .env)")
    else:
        logger.info(">>> Step 3: SKIPPED (--skip-comtrade)")
        processed = cfg.processed_dir / "bilateral_flows.parquet"
        if processed.exists():
            import pandas as pd
            bilateral_df = pd.read_parquet(processed)
            logger.info(f"Loaded existing bilateral: {len(bilateral_df):,} rows")

    # ── Step 4: Download CEPII Gravity ────────────────────
    if not args.skip_gravity:
        logger.info("\n>>> Step 4: Downloading CEPII Gravity dataset")
        from mktrade.data.gravity import load_gravity

        gravity_df = load_gravity(cfg)
        logger.info(f"Gravity: {len(gravity_df):,} rows")
    else:
        logger.info(">>> Step 4: SKIPPED (--skip-gravity)")

    # ── Step 5: Download WDI ──────────────────────────────
    if not args.skip_wdi:
        logger.info("\n>>> Step 5: Downloading World Bank WDI indicators")
        from mktrade.data.wdi import download_wdi

        wdi_df = download_wdi(cfg)
        logger.info(f"WDI: {len(wdi_df):,} rows")
    else:
        logger.info(">>> Step 5: SKIPPED (--skip-wdi)")

    # ── Step 6: Validation ────────────────────────────────
    logger.info("\n>>> Step 6: Validating MKD data")
    if atlas_df is not None:
        from mktrade.data.clean import validate_mkd_exports

        results = validate_mkd_exports(atlas_df, bilateral_df)
    else:
        logger.warning("No Atlas data available for validation")

    logger.info("\n>>> Phase 1a COMPLETE")


if __name__ == "__main__":
    main()
