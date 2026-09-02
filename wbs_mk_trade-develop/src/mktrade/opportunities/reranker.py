"""Validation-trained probability reranker with bootstrap rank stability."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from collections.abc import Sequence

_NON_FEATURE_COLUMNS = {
    "label", "rank", "ensemble_rank", "section", "hs4", "iso3", "partner_iso3",
    "score", "predicted_probability", "probability_lower", "probability_upper",
    "average_rank", "rank_std", "rank_stability", "ensemble_method",
}


class OpportunityReranker:
    """Learn how candidate signals map to historical success probabilities.

    Every estimator owns its imputation and scaling statistics, so inference is
    normalised with validation-era statistics rather than the current candidate
    set. Bootstrap refits provide probability intervals and rank stability.
    """

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        *,
        random_state: int = 42,
        n_bootstrap: int = 20,
        calibrate: bool = True,
    ) -> None:
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.random_state = int(random_state)
        self.n_bootstrap = max(0, int(n_bootstrap))
        self.calibrate = bool(calibrate)
        self.models: list[object] = []
        self.training_rows = 0
        self.positive_rate = 0.0

    @staticmethod
    def _numeric_features(frame: pd.DataFrame) -> list[str]:
        return [
            column
            for column in frame.select_dtypes(include=[np.number, "bool"]).columns
            if column not in _NON_FEATURE_COLUMNS and not column.endswith("_rank")
        ]

    def _make_model(self, labels: np.ndarray) -> object:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        counts = np.bincount(labels.astype(int), minlength=2)
        folds = int(min(3, counts.min()))
        if self.calibrate and folds >= 2:
            return CalibratedClassifierCV(pipeline, method="sigmoid", cv=folds)
        return pipeline

    def fit(self, validation_df: pd.DataFrame, label_col: str = "label") -> OpportunityReranker:
        """Fit on validation candidates scored by every available method."""

        if label_col not in validation_df.columns:
            raise ValueError(f"validation_df must contain '{label_col}'")
        labels = pd.to_numeric(validation_df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Reranker training requires at least one positive and one negative")

        if self.feature_columns is None:
            self.feature_columns = self._numeric_features(validation_df)
        if not self.feature_columns:
            raise ValueError("No numeric reranking features are available")
        missing = [column for column in self.feature_columns if column not in validation_df.columns]
        if missing:
            raise ValueError(f"Missing reranker features: {missing}")

        features = validation_df[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        self.models = []
        base = self._make_model(labels)
        base.fit(features, labels)
        self.models.append(base)

        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_bootstrap):
            for _attempt in range(20):
                indices = rng.integers(0, len(validation_df), len(validation_df))
                sampled_labels = labels[indices]
                if len(np.unique(sampled_labels)) == 2:
                    break
            else:
                continue
            model = self._make_model(sampled_labels)
            model.fit(features.iloc[indices], sampled_labels)
            self.models.append(model)

        self.training_rows = len(validation_df)
        self.positive_rate = float(labels.mean())
        return self

    def predict_distribution(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Return probability and rank distributions across bootstrap models."""

        if not self.models or not self.feature_columns:
            raise RuntimeError("OpportunityReranker must be fitted before prediction")
        missing = [column for column in self.feature_columns if column not in candidates.columns]
        if missing:
            raise ValueError(f"Missing reranker features: {missing}")
        features = candidates[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        probabilities = np.column_stack(
            [model.predict_proba(features)[:, 1] for model in self.models]
        )
        ranks = np.column_stack(
            [pd.Series(-probabilities[:, idx]).rank(method="average").to_numpy() for idx in range(probabilities.shape[1])]
        )
        return pd.DataFrame(
            {
                "predicted_probability": probabilities.mean(axis=1),
                "probability_lower": np.quantile(probabilities, 0.05, axis=1),
                "probability_upper": np.quantile(probabilities, 0.95, axis=1),
                "average_rank": ranks.mean(axis=1),
                "rank_std": ranks.std(axis=1),
                "rank_stability": 1.0 / (1.0 + ranks.std(axis=1)),
            },
            index=candidates.index,
        )

    def rerank(self, candidates: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
        """Attach calibrated probabilities and select the final top candidates."""

        result = candidates.copy()
        distribution = self.predict_distribution(result)
        for column in distribution.columns:
            result[column] = distribution[column]
        result["score"] = result["predicted_probability"]
        result = result.sort_values(
            ["predicted_probability", "average_rank"], ascending=[False, True]
        ).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        result["ensemble_score"] = result["predicted_probability"]
        result["ensemble_rank"] = result["rank"]
        result["ensemble_method"] = "validation_logistic_bootstrap"
        return result if top_k is None else result.head(top_k).reset_index(drop=True)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path | str) -> OpportunityReranker:
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain an OpportunityReranker")
        return model
