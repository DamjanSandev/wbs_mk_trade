"""PPML gravity model baseline for bilateral trade prediction (Task B).

Estimates a Poisson Pseudo-Maximum Likelihood gravity equation using
statsmodels GLM with a log-link. This is the standard econometric baseline
for predicting trade flows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def fit_ppml_gravity(
    bilateral_df: pd.DataFrame,
    gravity_df: pd.DataFrame,
    train_years: list[int],
) -> object:
    """Fit PPML gravity model on training years.

    The gravity equation: trade_ij = exp(b0 + b1*ln(dist) + b2*contig + b3*lang + b4*fta
                                        + b5*ln(gdp_o) + b6*ln(gdp_d)) + epsilon

    Returns a fitted statsmodels GLM result object.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not installed — skipping PPML gravity baseline")
        return None

    # Aggregate bilateral exports by reporter-partner
    bil = bilateral_df[
        (bilateral_df["year"].isin(train_years)) & (bilateral_df["flow"] == "X")
    ]
    if bil.empty:
        logger.warning("No bilateral export data for training years")
        return None

    trade = bil.groupby(["reporter_iso3", "partner_iso3"])["value"].sum().reset_index()
    trade = trade.rename(columns={"reporter_iso3": "iso3_o", "partner_iso3": "iso3_d"})

    # Get gravity features (use latest available year)
    grav = gravity_df.copy()
    if "year" in grav.columns:
        max_train_year = max(train_years)
        available_years = grav["year"].unique()
        use_year = max(y for y in available_years if y <= max_train_year) if any(
            y <= max_train_year for y in available_years
        ) else grav["year"].max()
        grav = grav[grav["year"] == use_year]

    # Merge
    merged = trade.merge(grav, on=["iso3_o", "iso3_d"], how="inner")
    if merged.empty:
        logger.warning("No matched gravity-trade pairs")
        return None

    # Prepare features
    merged["ln_dist"] = np.log(merged["dist"].clip(lower=1))
    merged["ln_gdp_o"] = np.log(merged["gdp_o"].clip(lower=1))
    merged["ln_gdp_d"] = np.log(merged["gdp_d"].clip(lower=1))

    feature_cols = ["ln_dist", "contig", "comlang_off", "fta_wto", "ln_gdp_o", "ln_gdp_d"]
    merged = merged.dropna(subset=feature_cols + ["value"])

    X = merged[feature_cols].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = merged["value"].astype(float)

    # Fit PPML (Poisson with log link)
    model = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.Log()))
    result = model.fit(maxiter=100)
    # Store feature columns for consistent prediction
    result._gravity_feature_cols = list(X.columns)

    logger.info(f"PPML gravity fit: {len(merged)} observations, pseudo-R²={result.pseudo_rsquared():.4f}")
    logger.info(f"  Coefficients: {dict(zip(X.columns, result.params.round(4)))}")

    return result


def predict_gravity(
    model: object,
    gravity_df: pd.DataFrame,
    test_years: list[int],
    country_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Generate predicted trade flows for test years.

    Returns columns: iso3_o, iso3_d, predicted_value.
    """
    if model is None:
        return pd.DataFrame()

    try:
        import statsmodels.api as sm
    except ImportError:
        return pd.DataFrame()

    grav = gravity_df.copy()
    if "year" in grav.columns:
        available_years = grav["year"].unique()
        use_year = max(y for y in available_years if y <= max(test_years)) if any(
            y <= max(test_years) for y in available_years
        ) else grav["year"].max()
        grav = grav[grav["year"] == use_year]

    if country_pairs is not None:
        pairs_df = pd.DataFrame(country_pairs, columns=["iso3_o", "iso3_d"])
        grav = grav.merge(pairs_df, on=["iso3_o", "iso3_d"], how="inner")

    grav["ln_dist"] = np.log(grav["dist"].clip(lower=1))
    grav["ln_gdp_o"] = np.log(grav["gdp_o"].clip(lower=1))
    grav["ln_gdp_d"] = np.log(grav["gdp_d"].clip(lower=1))

    feature_cols = ["ln_dist", "contig", "comlang_off", "fta_wto", "ln_gdp_o", "ln_gdp_d"]
    grav = grav.dropna(subset=feature_cols)

    X = grav[feature_cols].astype(float)
    X = sm.add_constant(X, has_constant="add")
    # Ensure columns match what the model was trained on
    if hasattr(model, "_gravity_feature_cols"):
        X = X.reindex(columns=model._gravity_feature_cols, fill_value=0)

    grav["predicted_value"] = model.predict(X)

    return grav[["iso3_o", "iso3_d", "predicted_value"]].reset_index(drop=True)
