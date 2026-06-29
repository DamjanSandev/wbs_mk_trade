"""Download bilateral trade flows from UN Comtrade via comtradeapicall.

Provides exporter x importer x product x year flows for Task B (market expansion).
All API calls are cached to data/raw/comtrade/ keyed by query params.
Respects rate limits and uses exponential backoff on errors.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import comtradeapicall as ct
import pandas as pd
from loguru import logger

from mktrade.config import DataConfig

# Module-level call counter for rate-limit awareness
_call_count = 0
_calls_this_session = 0


def _cache_key(reporter: str, flow: str, year: str, hs_level: int, partner: str) -> str:
    """Build a deterministic cache filename from query params."""
    return f"{reporter}_{flow}_{year}_hs{hs_level}_p{partner}.parquet"


def _cache_path(cfg: DataConfig, reporter: str, flow: str, year: str, partner: str) -> Path:
    """Full cache path for a Comtrade query."""
    ct_dir = cfg.raw_dir / "comtrade"
    ct_dir.mkdir(parents=True, exist_ok=True)
    fname = _cache_key(reporter, flow, year, cfg.hs_level, partner)
    return ct_dir / fname


def _cmd_code(hs_level: int) -> str:
    """Map HS level to Comtrade cmdCode parameter."""
    return {2: "AG2", 4: "AG4", 6: "AG6"}.get(hs_level, "AG4")


def _call_api_with_backoff(
    func: callable,
    max_retries: int = 3,
    base_delay: float = 5.0,
    **kwargs,
) -> pd.DataFrame:
    """Call a comtradeapicall function with exponential backoff on failure."""
    global _call_count, _calls_this_session

    for attempt in range(max_retries + 1):
        try:
            result = func(**kwargs)
            _call_count += 1
            _calls_this_session += 1
            if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                logger.warning(f"Empty result for {kwargs}")
                return pd.DataFrame()
            return result
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed after {max_retries} retries: {e}")
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay}s")
            time.sleep(delay)

    return pd.DataFrame()


def check_data_availability(
    cfg: DataConfig,
    reporter: str,
    years: list[int],
) -> list[int]:
    """Check which years have published data for a reporter.

    Returns list of available years (subset of input).
    """
    period_str = ",".join(str(y) for y in years)
    try:
        avail = ct.getFinalDataAvailability(
            subscription_key=cfg.comtrade_api_key,
            typeCode="C",
            freqCode="A",
            clCode="HS",
            period=period_str,
            reporterCode=reporter,
        )
        if avail is None or (isinstance(avail, pd.DataFrame) and avail.empty):
            logger.warning(f"No availability info for reporter={reporter}")
            return years  # Assume all available rather than skip

        if isinstance(avail, pd.DataFrame) and "period" in avail.columns:
            available = avail["period"].astype(int).unique().tolist()
            skipped = [y for y in years if y not in available]
            if skipped:
                logger.info(f"Reporter {reporter}: skipping unavailable years {skipped}")
            return [y for y in years if y in available]
        return years
    except Exception as e:
        logger.warning(f"Availability check failed ({e}), proceeding with all years")
        return years


def download_task_a(cfg: DataConfig) -> list[Path]:
    """Download country x product aggregate exports for Task A.

    Uses partnerCode='0' (World) to get total exports per product.
    One call per (reporter, flow='X', year) at the configured HS level.
    """
    reporter = cfg.comtrade_reporters_m49[0]  # Focus country M49
    years = list(range(cfg.comtrade_year_start, cfg.comtrade_year_end + 1))
    cmd = _cmd_code(cfg.hs_level)
    cached_paths: list[Path] = []
    api_calls = 0

    # Check availability
    available_years = check_data_availability(cfg, reporter, years)

    for year in available_years:
        path = _cache_path(cfg, reporter, "X", str(year), "0")
        if path.exists():
            logger.debug(f"Cache hit: {path.name}")
            cached_paths.append(path)
            continue

        logger.info(f"Fetching Task A: reporter={reporter}, year={year}, cmd={cmd}")
        df = _call_api_with_backoff(
            ct.getFinalData,
            subscription_key=cfg.comtrade_api_key,
            typeCode="C",
            freqCode="A",
            clCode="HS",
            period=str(year),
            reporterCode=reporter,
            cmdCode=cmd,
            flowCode="X",
            partnerCode="0",
            partner2Code="0",
            customsCode="C00",
            motCode="0",
            maxRecords=cfg.comtrade_max_records,
            includeDesc=True,
        )
        api_calls += 1

        if not df.empty:
            df.to_parquet(path, index=False)
            logger.info(f"Cached {len(df)} rows -> {path.name}")
            cached_paths.append(path)
        else:
            logger.warning(f"No data for reporter={reporter}, year={year}")

        # Brief pause between calls
        time.sleep(1.5)

    logger.info(
        f"Task A download: {len(cached_paths)} files, "
        f"{api_calls} API calls this batch, "
        f"{_calls_this_session} total this session"
    )
    return cached_paths


def download_task_b(cfg: DataConfig) -> list[Path]:
    """Download bilateral flows for Task B (market expansion).

    Uses partnerCode=None to get all partner breakdowns.
    One call per (reporter, flow, year).
    """
    reporters = cfg.comtrade_reporters_m49
    flows = cfg.comtrade_flow_codes
    years = list(range(cfg.comtrade_year_start, cfg.comtrade_year_end + 1))
    cmd = _cmd_code(cfg.hs_level)
    cached_paths: list[Path] = []
    api_calls = 0

    for reporter in reporters:
        available_years = check_data_availability(cfg, reporter, years)
        for flow in flows:
            for year in available_years:
                path = _cache_path(cfg, reporter, flow, str(year), "all")
                if path.exists():
                    logger.debug(f"Cache hit: {path.name}")
                    cached_paths.append(path)
                    continue

                logger.info(
                    f"Fetching Task B: reporter={reporter}, flow={flow}, year={year}"
                )
                df = _call_api_with_backoff(
                    ct.getFinalData,
                    subscription_key=cfg.comtrade_api_key,
                    typeCode="C",
                    freqCode="A",
                    clCode="HS",
                    period=str(year),
                    reporterCode=reporter,
                    cmdCode=cmd,
                    flowCode=flow,
                    partnerCode=None,
                    partner2Code="0",
                    customsCode="C00",
                    motCode="0",
                    maxRecords=cfg.comtrade_max_records,
                    includeDesc=True,
                )
                api_calls += 1

                if not df.empty:
                    df.to_parquet(path, index=False)
                    logger.info(f"Cached {len(df)} rows -> {path.name}")
                    cached_paths.append(path)
                else:
                    logger.warning(f"No data for {reporter}/{flow}/{year}")

                time.sleep(1.5)

    logger.info(
        f"Task B download: {len(cached_paths)} files, "
        f"{api_calls} API calls this batch, "
        f"{_calls_this_session} total this session"
    )
    return cached_paths


def load_comtrade_cached(cfg: DataConfig, task: str = "a") -> pd.DataFrame:
    """Load all cached Comtrade parquets for a given task into one DataFrame."""
    ct_dir = cfg.raw_dir / "comtrade"
    if not ct_dir.exists():
        return pd.DataFrame()

    if task.lower() == "a":
        pattern = f"*_X_*_p0.parquet"
    else:
        pattern = f"*_p{('all')}.parquet"

    files = sorted(ct_dir.glob(pattern))
    if not files:
        logger.warning(f"No cached Comtrade files for task {task}")
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(files)} Comtrade cache files: {len(df):,} total rows")
    return df


def get_session_stats() -> dict[str, int]:
    """Return API call statistics for this session."""
    return {
        "api_calls_this_session": _calls_this_session,
        "total_calls": _call_count,
    }
