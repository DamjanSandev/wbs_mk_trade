"""Leakage-safe candidate features for product and destination reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _growth(first: float, last: float, periods: int) -> float:
    if periods <= 0 or first <= 0:
        return 0.0
    return float(np.clip((max(last, 0.0) / first) ** (1.0 / periods) - 1.0, -1.0, 10.0))


def _latest_at_or_before(frame: pd.DataFrame | None, year: int | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    result = frame.copy()
    if year is not None and "year" in result.columns:
        result = result[result["year"] <= year]
        if result.empty:
            return result
        result = result[result["year"] == result["year"].max()]
    return result


def enrich_product_candidates(
    candidates: pd.DataFrame,
    exports_df: pd.DataFrame | None,
    country: str,
    as_of_year: int | None,
    *,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """Add global demand, concentration, persistence, and supply features."""

    result = candidates.copy()
    defaults = {
        "global_demand_value": 0.0,
        "global_demand_growth": 0.0,
        "global_market_concentration": 0.0,
        "historical_active_years": 0.0,
        "historical_export_persistence": 0.0,
        "supply_capacity": 0.0,
    }
    if exports_df is None or exports_df.empty:
        return result.assign(**defaults)

    history = exports_df.copy()
    if as_of_year is not None:
        history = history[history["year"] <= as_of_year]
    if history.empty:
        return result.assign(**defaults)
    value_col = "export_value" if "export_value" in history.columns else "value"
    history[value_col] = _numeric(history[value_col])
    history["hs4"] = history["hs4"].astype(str).str.zfill(4)
    latest_year = int(history["year"].max())
    first_year = max(int(history["year"].min()), latest_year - lookback_years + 1)
    recent = history[history["year"] >= first_year]

    annual_world = recent.groupby(["hs4", "year"], as_index=False)[value_col].sum()
    demand_rows: list[dict[str, float | str]] = []
    for hs4, group in annual_world.groupby("hs4", sort=False):
        ordered = group.sort_values("year")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        demand_rows.append(
            {
                "hs4": hs4,
                "global_demand_value": float(last[value_col]),
                "global_demand_growth": _growth(
                    float(first[value_col]),
                    float(last[value_col]),
                    int(last["year"] - first["year"]),
                ),
            }
        )
    result = result.merge(pd.DataFrame(demand_rows), on="hs4", how="left")

    latest = history[history["year"] == latest_year]
    country_product = latest.groupby(["iso3", "hs4"], as_index=False)[value_col].sum()
    totals = country_product.groupby("hs4")[value_col].transform("sum").replace(0, np.nan)
    country_product["share_sq"] = (country_product[value_col] / totals) ** 2
    hhi = country_product.groupby("hs4", as_index=False)["share_sq"].sum().rename(
        columns={"share_sq": "global_market_concentration"}
    )
    result = result.merge(hhi, on="hs4", how="left")

    own = history[history["iso3"].astype(str) == str(country)].copy()
    active = own[own[value_col] > 0].groupby("hs4")["year"].nunique()
    persistence = own.groupby("hs4").apply(
        lambda group: _longest_consecutive_run(group.loc[group[value_col] > 0, "year"]),
        include_groups=False,
    ) if not own.empty else pd.Series(dtype=float)
    result["historical_active_years"] = result["hs4"].map(active)
    result["historical_export_persistence"] = result["hs4"].map(persistence)

    latest_own = own[own["year"] == latest_year].copy()
    latest_own["section"] = latest_own["hs4"].str[:2]
    section_supply = latest_own.groupby("section")[value_col].sum()
    if "section" not in result.columns:
        result["section"] = result["hs4"].astype(str).str[:2]
    result["supply_capacity"] = result["section"].map(section_supply)

    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
        result[column] = _numeric(result[column])
    for column in (
        "gnn_logit", "gnn_score", "density", "pci", "cog",
        "global_demand_value", "global_demand_growth", "supply_capacity",
    ):
        if column in result.columns and result[column].nunique(dropna=True) > 1:
            result[f"{column}_percentile"] = result[column].rank(
                pct=True, method="average"
            )
    return result


def enrich_product_candidate_universe(
    candidates: pd.DataFrame,
    exports_df: pd.DataFrame,
    complexity_df: pd.DataFrame | None,
    as_of_year: int,
    *,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """Vectorised leakage-safe features for all country-product queries."""

    required = {"iso3", "hs4"}
    if not required.issubset(candidates.columns):
        raise ValueError("candidates must contain iso3 and hs4")
    result = candidates.copy()
    result["iso3"] = result["iso3"].astype(str)
    result["hs4"] = result["hs4"].astype(str).str.zfill(4)
    result["section"] = result["hs4"].str[:2]

    if complexity_df is not None and not complexity_df.empty:
        complexity = complexity_df.copy()
        if "year" in complexity.columns:
            complexity = complexity[complexity["year"] <= as_of_year]
            if not complexity.empty:
                complexity = complexity[complexity["year"] == complexity["year"].max()]
        complexity["iso3"] = complexity["iso3"].astype(str)
        complexity["hs4"] = complexity["hs4"].astype(str).str.zfill(4)
        columns = [
            column
            for column in ("iso3", "hs4", "density", "pci", "cog", "complexity_gain")
            if column in complexity.columns
        ]
        result = result.merge(
            complexity[columns].drop_duplicates(["iso3", "hs4"], keep="last"),
            on=["iso3", "hs4"],
            how="left",
            suffixes=("", "_complexity"),
        )

    history = exports_df[exports_df["year"] <= as_of_year].copy()
    value_col = "export_value" if "export_value" in history.columns else "value"
    history[value_col] = _numeric(history[value_col])
    history["iso3"] = history["iso3"].astype(str)
    history["hs4"] = history["hs4"].astype(str).str.zfill(4)
    latest_year = int(history["year"].max())
    first_year = max(int(history["year"].min()), latest_year - lookback_years + 1)
    recent = history[history["year"] >= first_year]

    annual_world = recent.groupby(["hs4", "year"], as_index=False)[value_col].sum()
    demand = annual_world.sort_values("year").groupby("hs4", as_index=False).agg(
        first_value=(value_col, "first"),
        last_value=(value_col, "last"),
        first_year=("year", "first"),
        last_year=("year", "last"),
    )
    demand["global_demand_value"] = demand["last_value"]
    demand["global_demand_growth"] = demand.apply(
        lambda row: _growth(
            float(row.first_value),
            float(row.last_value),
            int(row.last_year - row.first_year),
        ),
        axis=1,
    )
    result = result.merge(
        demand[["hs4", "global_demand_value", "global_demand_growth"]],
        on="hs4",
        how="left",
    )

    latest = history[history["year"] == latest_year]
    country_product = latest.groupby(["iso3", "hs4"], as_index=False)[value_col].sum()
    product_totals = country_product.groupby("hs4")[value_col].transform("sum").replace(0, np.nan)
    country_product["share_sq"] = (country_product[value_col] / product_totals) ** 2
    concentration = country_product.groupby("hs4", as_index=False)["share_sq"].sum().rename(
        columns={"share_sq": "global_market_concentration"}
    )
    result = result.merge(concentration, on="hs4", how="left")

    active = history[history[value_col] > 0]
    active_years = active.groupby(["iso3", "hs4"], as_index=False)["year"].nunique().rename(
        columns={"year": "historical_active_years"}
    )
    persistence = active.groupby(["iso3", "hs4"])["year"].apply(
        _longest_consecutive_run
    ).rename("historical_export_persistence").reset_index()
    result = result.merge(active_years, on=["iso3", "hs4"], how="left")
    result = result.merge(persistence, on=["iso3", "hs4"], how="left")

    latest = latest.copy()
    latest["section"] = latest["hs4"].str[:2]
    supply = latest.groupby(["iso3", "section"], as_index=False)[value_col].sum().rename(
        columns={value_col: "supply_capacity"}
    )
    result = result.merge(supply, on=["iso3", "section"], how="left")

    feature_defaults = {
        "density": 0.0,
        "pci": 0.0,
        "cog": 0.0,
        "global_demand_value": 0.0,
        "global_demand_growth": 0.0,
        "global_market_concentration": 0.0,
        "historical_active_years": 0.0,
        "historical_export_persistence": 0.0,
        "supply_capacity": 0.0,
    }
    for column, default in feature_defaults.items():
        if column not in result.columns:
            result[column] = default
        result[column] = _numeric(result[column])

    # Query-relative ranks make scores comparable across countries and remove
    # much of the scale sensitivity of raw economic variables.
    rank_columns = [
        "gnn_logit", "gnn_score", "density", "pci", "cog",
        "global_demand_value", "global_demand_growth", "supply_capacity",
    ]
    for column in rank_columns:
        if column in result.columns and result[column].nunique(dropna=True) > 1:
            result[f"{column}_percentile"] = result.groupby("iso3")[column].rank(
                pct=True, method="average"
            )
    return result


def _longest_consecutive_run(years: Iterable[int]) -> int:
    ordered = sorted({int(year) for year in years})
    longest = current = 0
    previous: int | None = None
    for year in ordered:
        current = current + 1 if previous is not None and year == previous + 1 else 1
        longest = max(longest, current)
        previous = year
    return longest


def enrich_market_candidates(
    candidates: pd.DataFrame,
    *,
    country: str,
    bilateral_df: pd.DataFrame | None = None,
    gravity_df: pd.DataFrame | None = None,
    tariff_df: pd.DataFrame | None = None,
    wdi_df: pd.DataFrame | None = None,
    exports_df: pd.DataFrame | None = None,
    as_of_year: int | None = None,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """Add origin-product-destination demand and feasibility features."""

    result = candidates.copy()
    feature_defaults = {
        "destination_import_value": 0.0,
        "destination_import_growth": 0.0,
        "market_concentration": 0.0,
        "competitor_count": 0.0,
        "mkd_existing_export_value": 0.0,
        "mkd_historical_export_value": 0.0,
        "mkd_bilateral_trade": 0.0,
        "market_data_available": 0.0,
        "gravity_data_available": 0.0,
        "tariff_data_available": 0.0,
        "macro_data_available": 0.0,
        "distance": 0.0,
        "contiguous": 0.0,
        "common_language": 0.0,
        "fta": 0.0,
        "tariff_rate": 0.0,
        "destination_gdp": 0.0,
        "destination_gdp_pc": 0.0,
        "destination_population": 0.0,
        "mkd_supply_capacity": 0.0,
        "mkd_supply_growth": 0.0,
    }

    if bilateral_df is not None and not bilateral_df.empty:
        bilateral = bilateral_df.copy()
        if as_of_year is not None:
            bilateral = bilateral[bilateral["year"] <= as_of_year]
        if not bilateral.empty:
            bilateral["value"] = _numeric(bilateral["value"])
            bilateral["hs4"] = bilateral["hs4"].astype(str).str.zfill(4)
            bilateral["flow"] = bilateral["flow"].astype(str).str.upper()
            covered_markets = set(bilateral["reporter_iso3"].astype(str))
            result["market_data_available"] = result["partner_iso3"].astype(str).isin(
                covered_markets
            ).astype(float)
            latest_year = int(bilateral["year"].max())
            start_year = max(int(bilateral["year"].min()), latest_year - lookback_years + 1)
            recent = bilateral[bilateral["year"] >= start_year]

            imports = recent[recent["flow"] == "M"]
            annual_imports = imports.groupby(
                ["reporter_iso3", "hs4", "year"], as_index=False
            )["value"].sum()
            demand_rows: list[dict[str, float | str]] = []
            for (destination, hs4), group in annual_imports.groupby(
                ["reporter_iso3", "hs4"], sort=False
            ):
                ordered = group.sort_values("year")
                first, last = ordered.iloc[0], ordered.iloc[-1]
                demand_rows.append(
                    {
                        "partner_iso3": str(destination),
                        "hs4": str(hs4),
                        "destination_import_value": float(last["value"]),
                        "destination_import_growth": _growth(
                            float(first["value"]),
                            float(last["value"]),
                            int(last["year"] - first["year"]),
                        ),
                    }
                )
            if demand_rows:
                result = result.merge(pd.DataFrame(demand_rows), on=["partner_iso3", "hs4"], how="left")

            # For an import record, reporter=destination and partner=supplier.
            supplier = imports.groupby(
                ["reporter_iso3", "hs4", "partner_iso3"], as_index=False
            )["value"].sum()
            if not supplier.empty:
                totals = supplier.groupby(["reporter_iso3", "hs4"])["value"].transform(
                    "sum"
                ).replace(0, np.nan)
                supplier["share_sq"] = (supplier["value"] / totals) ** 2
                competition = supplier.groupby(
                    ["reporter_iso3", "hs4"], as_index=False
                ).agg(
                    market_concentration=("share_sq", "sum"),
                    competitor_count=("partner_iso3", "nunique"),
                ).rename(columns={"reporter_iso3": "partner_iso3"})
                result = result.merge(competition, on=["partner_iso3", "hs4"], how="left")

            own = bilateral[
                (bilateral["reporter_iso3"].astype(str) == str(country))
                & (bilateral["flow"] == "X")
            ]
            own_latest = own[own["year"] == latest_year].groupby(
                ["partner_iso3", "hs4"], as_index=False
            )["value"].sum().rename(columns={"value": "mkd_existing_export_value"})
            own_history = own.groupby(["partner_iso3", "hs4"], as_index=False)["value"].sum().rename(
                columns={"value": "mkd_historical_export_value"}
            )
            result = result.merge(own_latest, on=["partner_iso3", "hs4"], how="left")
            result = result.merge(own_history, on=["partner_iso3", "hs4"], how="left")

            origin_trade = bilateral[
                bilateral["reporter_iso3"].astype(str) == str(country)
            ].groupby("partner_iso3", as_index=False)["value"].sum().rename(
                columns={"value": "mkd_bilateral_trade"}
            )
            result = result.merge(origin_trade, on="partner_iso3", how="left")

    gravity = _latest_at_or_before(gravity_df, as_of_year)
    if not gravity.empty and {"iso3_o", "iso3_d"}.issubset(gravity.columns):
        gravity = gravity[gravity["iso3_o"].astype(str) == str(country)].copy()
        gravity_destinations = set(gravity["iso3_d"].astype(str))
        result["gravity_data_available"] = result["partner_iso3"].astype(str).isin(
            gravity_destinations
        ).astype(float)
        aliases = {
            "dist": "distance",
            "contig": "contiguous",
            "comlang_off": "common_language",
            "fta_wto": "fta",
        }
        available = {source: target for source, target in aliases.items() if source in gravity.columns}
        gravity = gravity.rename(columns=available).rename(columns={"iso3_d": "partner_iso3"})
        columns = ["partner_iso3", *available.values()]
        gravity = gravity[columns].drop_duplicates("partner_iso3", keep="last")
        result = result.merge(gravity, on="partner_iso3", how="left", suffixes=("", "_gravity"))
        for column in available.values():
            alternate = f"{column}_gravity"
            if alternate in result.columns:
                result[column] = result[alternate].combine_first(result.get(column))
                result = result.drop(columns=alternate)

    tariffs = _latest_at_or_before(tariff_df, as_of_year)
    if not tariffs.empty:
        origin_col = next(
            (column for column in ("origin_iso3", "exporter_iso3", "iso3_o") if column in tariffs.columns),
            None,
        )
        destination_col = next(
            (
                column
                for column in ("partner_iso3", "destination_iso3", "importer_iso3", "iso3_d")
                if column in tariffs.columns
            ),
            None,
        )
        rate_col = next(
            (
                column
                for column in ("tariff_rate", "applied_tariff", "simple_average_tariff")
                if column in tariffs.columns
            ),
            None,
        )
        if destination_col and rate_col:
            if origin_col:
                tariffs = tariffs[tariffs[origin_col].astype(str) == str(country)]
            tariffs = tariffs.rename(
                columns={destination_col: "partner_iso3", rate_col: "tariff_rate"}
            )
            join_columns = ["partner_iso3"]
            if "hs4" in tariffs.columns:
                tariffs["hs4"] = tariffs["hs4"].astype(str).str.zfill(4)
                join_columns.append("hs4")
            rates = tariffs.groupby(join_columns, as_index=False)["tariff_rate"].mean()
            result = result.merge(rates, on=join_columns, how="left", suffixes=("", "_tariff"))
            if "tariff_rate_tariff" in result.columns:
                result["tariff_rate"] = result["tariff_rate_tariff"].combine_first(
                    result.get("tariff_rate")
                )
                result = result.drop(columns="tariff_rate_tariff")
            result["tariff_data_available"] = result["tariff_rate"].notna().astype(float)

    wdi = _latest_at_or_before(wdi_df, as_of_year)
    if not wdi.empty and "iso3" in wdi.columns:
        macro_countries = set(wdi["iso3"].astype(str))
        result["macro_data_available"] = result["partner_iso3"].astype(str).isin(
            macro_countries
        ).astype(float)
        aliases = {"gdp": "destination_gdp", "gdp_pc": "destination_gdp_pc", "population": "destination_population"}
        available = {source: target for source, target in aliases.items() if source in wdi.columns}
        macro = wdi.rename(columns=available).rename(columns={"iso3": "partner_iso3"})
        macro = macro[["partner_iso3", *available.values()]].drop_duplicates("partner_iso3", keep="last")
        result = result.merge(macro, on="partner_iso3", how="left", suffixes=("", "_wdi"))
        for column in available.values():
            alternate = f"{column}_wdi"
            if alternate in result.columns:
                result[column] = result[alternate].combine_first(result.get(column))
                result = result.drop(columns=alternate)

    if exports_df is not None and not exports_df.empty:
        exports = exports_df[exports_df["iso3"].astype(str) == str(country)].copy()
        if as_of_year is not None:
            exports = exports[exports["year"] <= as_of_year]
        if not exports.empty:
            value_col = "export_value" if "export_value" in exports.columns else "value"
            exports[value_col] = _numeric(exports[value_col])
            exports["hs4"] = exports["hs4"].astype(str).str.zfill(4)
            annual = exports.groupby(["hs4", "year"], as_index=False)[value_col].sum()
            supply_rows = []
            for hs4, group in annual.groupby("hs4", sort=False):
                ordered = group.sort_values("year")
                first, last = ordered.iloc[0], ordered.iloc[-1]
                supply_rows.append(
                    {
                        "hs4": hs4,
                        "mkd_supply_capacity": float(last[value_col]),
                        "mkd_supply_growth": _growth(
                            float(first[value_col]), float(last[value_col]), int(last["year"] - first["year"])
                        ),
                    }
                )
            result = result.merge(pd.DataFrame(supply_rows), on="hs4", how="left")

    for column, default in feature_defaults.items():
        if column not in result.columns:
            result[column] = default
        result[column] = _numeric(result[column])
    return result
