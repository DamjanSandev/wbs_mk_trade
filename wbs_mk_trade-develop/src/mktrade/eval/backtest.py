"""Utilities for rolling-origin evaluation and recommendation stability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class BacktestWindow:
    """One leakage-safe train/validation/test time window."""

    train_end_year: int
    validation_year: int
    test_years: tuple[int, ...]


def make_backtest_windows(
    available_years: Sequence[int],
    cutoffs: Sequence[int],
    *,
    persistence_years: int = 2,
) -> list[BacktestWindow]:
    """Create rolling windows that have enough future data for persistence."""

    if persistence_years < 1:
        raise ValueError("persistence_years must be at least one")
    available = {int(year) for year in available_years}
    windows = []
    for cutoff in sorted({int(year) for year in cutoffs}):
        validation_year = cutoff + 1
        validation_years = set(range(validation_year, validation_year + persistence_years))
        test_start = validation_year + persistence_years
        test_years = tuple(range(test_start, test_start + persistence_years))
        required = {*validation_years, *test_years}
        if required.issubset(available):
            windows.append(BacktestWindow(cutoff, validation_year, test_years))
    return windows


def aggregate_backtest_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Report mean and standard deviation across cutoffs and random seeds."""

    if results.empty:
        return pd.DataFrame()
    group_columns = [column for column in ("model",) if column in results.columns]
    excluded = {"cutoff", "seed", *group_columns}
    metric_columns = [
        column
        for column in results.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    if not metric_columns:
        return pd.DataFrame()
    if not group_columns:
        frame = results.assign(model="all")
        group_columns = ["model"]
    else:
        frame = results
    aggregate = frame.groupby(group_columns)[metric_columns].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    return aggregate.reset_index()


def summarise_rank_stability(
    ranking_runs: Sequence[pd.DataFrame],
    *,
    key_columns: Sequence[str] = ("hs4",),
    rank_column: str = "rank",
) -> pd.DataFrame:
    """Summarise average rank, dispersion, and coverage across model runs."""

    if not ranking_runs:
        return pd.DataFrame(columns=[*key_columns, "average_rank", "rank_std", "rank_stability"])
    frames = []
    for run_id, ranking in enumerate(ranking_runs):
        missing = [column for column in (*key_columns, rank_column) if column not in ranking.columns]
        if missing:
            raise ValueError(f"Ranking run {run_id} is missing columns {missing}")
        frame = ranking[[*key_columns, rank_column]].copy()
        frame["run_id"] = run_id
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    result = combined.groupby(list(key_columns), as_index=False).agg(
        average_rank=(rank_column, "mean"),
        rank_std=(rank_column, lambda values: float(np.std(values, ddof=0))),
        runs_observed=("run_id", "nunique"),
    )
    result["rank_stability"] = 1.0 / (1.0 + result["rank_std"])
    result["run_coverage"] = result["runs_observed"] / len(ranking_runs)
    return result.sort_values("average_rank").reset_index(drop=True)
