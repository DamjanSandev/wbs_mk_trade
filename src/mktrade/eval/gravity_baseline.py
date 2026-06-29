"""PPML gravity model baseline for bilateral trade prediction (Task B).

Estimates a Poisson Pseudo-Maximum Likelihood gravity equation using
statsmodels GLM with a log-link.
"""

from __future__ import annotations

import pandas as pd


def fit_ppml_gravity(
    bilateral_df: pd.DataFrame,
    gravity_df: pd.DataFrame,
    train_years: list[int],
) -> object:
    """Fit PPML gravity model on training years.

    Returns a fitted statsmodels GLM result object.
    """
    raise NotImplementedError


def predict_gravity(
    model: object,
    gravity_df: pd.DataFrame,
    test_years: list[int],
) -> pd.DataFrame:
    """Generate predicted trade flows for test years.

    Returns columns: reporter_iso3, partner_iso3, hs4, year, predicted_value.
    """
    raise NotImplementedError
