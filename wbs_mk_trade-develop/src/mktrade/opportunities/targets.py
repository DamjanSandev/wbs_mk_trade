"""Outcome definitions for product and market opportunity ranking.

The original pipeline treated any positive shipment as a successful link.  This
module centralises the stricter, time-aware definitions used by training,
reranking, and backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class SuccessCriteria:
    """Definition of a meaningful and persistent export relationship."""

    min_export_value: float = 100_000.0
    min_rca: float | None = 1.0
    min_consecutive_years: int = 2

    def __post_init__(self) -> None:
        if self.min_export_value < 0:
            raise ValueError("min_export_value must be non-negative")
        if self.min_rca is not None and self.min_rca < 0:
            raise ValueError("min_rca must be non-negative or None")
        if self.min_consecutive_years < 1:
            raise ValueError("min_consecutive_years must be at least one")


def criteria_from_config(cfg: object) -> SuccessCriteria:
    """Create :class:`SuccessCriteria` from a TrainConfig-like object."""

    return SuccessCriteria(
        min_export_value=float(getattr(cfg, "min_export_value", 100_000.0)),
        min_rca=float(getattr(cfg, "min_rca", 1.0)),
        min_consecutive_years=int(getattr(cfg, "min_consecutive_years", 2)),
    )


def _resolve_column(df: pd.DataFrame, names: Sequence[str], *, required: bool = True) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"Expected one of columns {list(names)}, got {list(df.columns)}")
    return None


def _normalise_years(years: int | Iterable[int] | None) -> set[int] | None:
    if years is None:
        return None
    if isinstance(years, (int, np.integer)):
        return {int(years)}
    return {int(year) for year in years}


def _has_consecutive_run(years: Iterable[int], length: int) -> bool:
    ordered = sorted({int(year) for year in years})
    if length <= 1:
        return bool(ordered)
    run = 1
    for previous, current in zip(ordered, ordered[1:], strict=False):
        run = run + 1 if current == previous + 1 else 1
        if run >= length:
            return True
    return False


def _link_value(column: str, value: object) -> str:
    text = str(value)
    return text.zfill(4) if column.lower() in {"hs4", "hs_code"} else text


def qualifying_export_rows(
    exports_df: pd.DataFrame,
    criteria: SuccessCriteria,
    years: int | Iterable[int] | None = None,
    *,
    value_columns: Sequence[str] = ("export_value", "value"),
    rca_columns: Sequence[str] = ("export_rca", "rca"),
) -> pd.DataFrame:
    """Return annual rows that meet the value and RCA thresholds."""

    if exports_df.empty:
        return exports_df.copy()
    if "year" not in exports_df.columns:
        raise ValueError("exports_df must contain a 'year' column")

    value_col = _resolve_column(exports_df, value_columns)
    rca_col = _resolve_column(exports_df, rca_columns, required=False)
    frame = exports_df.copy()
    selected_years = _normalise_years(years)
    if selected_years is not None:
        frame = frame[frame["year"].isin(selected_years)]

    value = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    mask = value >= criteria.min_export_value
    if criteria.min_rca is not None:
        if rca_col is None:
            raise ValueError("An RCA column is required when min_rca is set")
        rca = pd.to_numeric(frame[rca_col], errors="coerce").fillna(0.0)
        mask &= rca >= criteria.min_rca
    return frame.loc[mask].copy()


def sustained_link_set(
    exports_df: pd.DataFrame,
    criteria: SuccessCriteria,
    years: int | Iterable[int] | None = None,
    *,
    group_columns: Sequence[str] = ("iso3", "hs4"),
    value_columns: Sequence[str] = ("export_value", "value"),
    rca_columns: Sequence[str] = ("export_rca", "rca"),
) -> set[tuple[str, ...]]:
    """Return links with a qualifying run of consecutive years."""

    if exports_df.empty:
        return set()
    missing = [column for column in group_columns if column not in exports_df.columns]
    if missing:
        raise ValueError(f"Missing link columns: {missing}")

    qualifying = qualifying_export_rows(
        exports_df,
        criteria,
        years,
        value_columns=value_columns,
        rca_columns=rca_columns,
    )
    if qualifying.empty:
        return set()

    annual = qualifying[list(group_columns) + ["year"]].drop_duplicates()
    links: set[tuple[str, ...]] = set()
    for key, group in annual.groupby(list(group_columns), sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        if _has_consecutive_run(group["year"], criteria.min_consecutive_years):
            links.add(
                tuple(
                    _link_value(column, value)
                    for column, value in zip(group_columns, key_tuple, strict=True)
                )
            )
    return links


def sustained_export_snapshot(
    exports_df: pd.DataFrame,
    cutoff_year: int,
    criteria: SuccessCriteria,
) -> pd.DataFrame:
    """Build one edge row per sustained country-product link as of a cutoff.

    The most recent qualifying annual observation supplies edge attributes.  Its
    year is set to the cutoff so graph builders can combine a historical union
    of successful links with cutoff-year node and macro features.
    """

    history = exports_df[exports_df["year"] <= cutoff_year].copy()
    links = sustained_link_set(history, criteria)
    if not links:
        return history.iloc[0:0].copy()

    qualifying = qualifying_export_rows(history, criteria)
    qualifying["iso3"] = qualifying["iso3"].astype(str)
    qualifying["hs4"] = qualifying["hs4"].astype(str).str.zfill(4)
    key_index = pd.MultiIndex.from_frame(qualifying[["iso3", "hs4"]].astype(str))
    wanted = pd.MultiIndex.from_tuples(sorted(links), names=["iso3", "hs4"])
    qualifying = qualifying[key_index.isin(wanted)]
    snapshot = (
        qualifying.sort_values("year")
        .groupby(["iso3", "hs4"], as_index=False, sort=False)
        .tail(1)
        .copy()
    )
    snapshot["year"] = int(cutoff_year)
    return snapshot.reset_index(drop=True)


def label_product_candidates(
    candidates: pd.DataFrame,
    exports_df: pd.DataFrame,
    country: str,
    outcome_years: Iterable[int],
    criteria: SuccessCriteria,
) -> np.ndarray:
    """Label candidate products by future sustained export success."""

    links = sustained_link_set(exports_df, criteria, outcome_years)
    return np.asarray(
        [1 if (str(country), str(hs4).zfill(4)) in links else 0 for hs4 in candidates["hs4"]],
        dtype=np.int64,
    )


def label_market_candidates(
    candidates: pd.DataFrame,
    bilateral_df: pd.DataFrame,
    country: str,
    outcome_years: Iterable[int],
    *,
    min_export_value: float = 100_000.0,
    min_consecutive_years: int = 2,
) -> np.ndarray:
    """Label origin-product-destination candidates using bilateral exports."""

    if bilateral_df is None or bilateral_df.empty:
        return np.zeros(len(candidates), dtype=np.int64)
    required = {"reporter_iso3", "partner_iso3", "hs4", "flow", "year"}
    if not required.issubset(bilateral_df.columns):
        raise ValueError(f"bilateral_df is missing columns: {sorted(required - set(bilateral_df.columns))}")

    bilateral = bilateral_df[
        (bilateral_df["reporter_iso3"].astype(str) == str(country))
        & (bilateral_df["flow"].astype(str).str.upper() == "X")
    ].copy()
    market_criteria = SuccessCriteria(
        min_export_value=min_export_value,
        min_rca=None,
        min_consecutive_years=min_consecutive_years,
    )
    links = sustained_link_set(
        bilateral,
        market_criteria,
        outcome_years,
        group_columns=("hs4", "partner_iso3"),
        value_columns=("value", "export_value"),
    )
    return np.asarray(
        [
            1 if (str(row.hs4).zfill(4), str(row.partner_iso3)) in links else 0
            for row in candidates[["hs4", "partner_iso3"]].itertuples(index=False)
        ],
        dtype=np.int64,
    )
