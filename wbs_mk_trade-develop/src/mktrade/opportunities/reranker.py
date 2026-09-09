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
    "average_rank", "rank_std", "rank_stability", "ensemble_method", "ranking_score",
    "origin_iso3", "cutoff", "query_id",
}

_LOG_FEATURE_HINTS = (
    "value", "gdp", "population", "capacity", "trade", "pref_attach", "competitor_count"
)


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
        model_type: str = "logistic",
        max_pairs_per_query: int = 200,
    ) -> None:
        self.feature_columns = list(feature_columns) if feature_columns is not None else None
        self.random_state = int(random_state)
        self.n_bootstrap = max(0, int(n_bootstrap))
        self.calibrate = bool(calibrate)
        self.model_type = str(model_type).lower()
        if self.model_type not in {"logistic", "pairwise"}:
            raise ValueError("model_type must be 'logistic' or 'pairwise'")
        self.max_pairs_per_query = max(1, int(max_pairs_per_query))
        self.models: list[object] = []
        # Pairwise classifiers are trained on feature differences. Their raw
        # decision values rank candidates correctly, but sigmoid(decision) is
        # not an item-level probability. A separate validation-fitted mapper
        # turns within-query score percentiles into success probabilities.
        self.score_calibrators: list[object] = []
        self.training_rows = 0
        self.training_queries = 0
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
                        class_weight=None if self.model_type == "pairwise" else "balanced",
                        max_iter=2_000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        counts = np.bincount(labels.astype(int), minlength=2)
        folds = int(min(3, counts.min()))
        if self.calibrate and self.model_type == "logistic" and folds >= 2:
            return CalibratedClassifierCV(pipeline, method="sigmoid", cv=folds)
        return pipeline

    @staticmethod
    def _query_ids(frame: pd.DataFrame) -> pd.Series:
        for columns in (("cutoff", "iso3"), ("cutoff", "origin_iso3"), ("iso3",), ("origin_iso3",)):
            if all(column in frame.columns for column in columns):
                return frame[list(columns)].astype(str).agg("|".join, axis=1)
        return pd.Series("all", index=frame.index)

    @staticmethod
    def _transform_features(features: pd.DataFrame) -> pd.DataFrame:
        """Apply stable transforms to strongly skewed monetary/size features."""

        result = features.copy()
        for column in result.columns:
            values = pd.to_numeric(result[column], errors="coerce")
            if any(hint in column.lower() for hint in _LOG_FEATURE_HINTS):
                values = np.sign(values) * np.log1p(np.abs(values))
            result[column] = values
        return result

    def _pairwise_examples(
        self,
        frame: pd.DataFrame,
        features: pd.DataFrame,
        labels: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Construct symmetric positive-minus-hard-negative training pairs."""

        queries = self._query_ids(frame).to_numpy()
        rng = np.random.default_rng(self.random_state)
        differences: list[np.ndarray] = []
        pair_labels: list[int] = []
        hardness_columns = [
            column
            for column in ("gnn_logit", "gnn_score", "density", "global_demand_value")
            if column in frame.columns
        ]
        if hardness_columns:
            hardness = frame[hardness_columns].rank(pct=True).mean(axis=1).fillna(0.0).to_numpy()
        else:
            hardness = np.zeros(len(frame), dtype=float)

        values = features.to_numpy(dtype=float)
        for query in np.unique(queries):
            query_indices = np.flatnonzero(queries == query)
            positives = query_indices[labels[query_indices] == 1]
            negatives = query_indices[labels[query_indices] == 0]
            if not len(positives) or not len(negatives):
                continue
            ordered_negatives = negatives[np.argsort(hardness[negatives])[::-1]]
            pairs_per_positive = max(1, self.max_pairs_per_query // len(positives))
            for positive in positives:
                hard_count = min(len(ordered_negatives), max(1, pairs_per_positive // 2))
                selected = list(ordered_negatives[:hard_count])
                remaining = ordered_negatives[hard_count:]
                random_count = min(pairs_per_positive - hard_count, len(remaining))
                if random_count:
                    selected.extend(rng.choice(remaining, random_count, replace=False).tolist())
                for negative in selected:
                    difference = values[positive] - values[negative]
                    differences.extend((difference, -difference))
                    pair_labels.extend((1, 0))

        if not differences:
            raise ValueError("Pairwise reranking requires a query with positives and negatives")
        return pd.DataFrame(differences, columns=features.columns), np.asarray(pair_labels)

    def _fit_model(
        self,
        frame: pd.DataFrame,
        labels: np.ndarray,
    ) -> object:
        features = self._transform_features(frame[self.feature_columns])
        model_labels = labels
        if self.model_type == "pairwise":
            features, model_labels = self._pairwise_examples(frame, features, labels)
        model = self._make_model(model_labels)
        model.fit(features, model_labels)
        return model

    def _model_scores(self, model: object, frame: pd.DataFrame) -> np.ndarray:
        """Return an estimator's unbounded candidate-ranking scores."""

        features = self._transform_features(
            frame[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        )
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(features), dtype=float)
        probability = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        return np.log(clipped / (1.0 - clipped))

    def _within_query_percentiles(
        self,
        scores: np.ndarray,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        """Normalise relative pairwise scores without mixing country queries."""

        scored = pd.DataFrame(
            {
                "score": np.asarray(scores, dtype=float),
                "query": self._query_ids(frame).to_numpy(),
            },
            index=frame.index,
        )
        return scored.groupby("query", sort=False)["score"].rank(pct=True).to_numpy()

    def _fit_score_calibrator(
        self,
        model: object,
        frame: pd.DataFrame,
        labels: np.ndarray,
    ) -> object:
        """Map pairwise rank position to an item-level success probability."""

        raw_scores = self._model_scores(model, frame)
        percentiles = self._within_query_percentiles(raw_scores, frame).reshape(-1, 1)
        calibrator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2_000, random_state=self.random_state),
                ),
            ]
        )
        calibrator.fit(percentiles, labels)
        return calibrator

    @property
    def has_probability_calibration(self) -> bool:
        """Whether this reranker can emit meaningful item-level probabilities."""

        return self.model_type != "pairwise" or (
            len(self.score_calibrators) == len(self.models) and bool(self.models)
        )

    def fit(self, validation_df: pd.DataFrame, label_col: str = "label") -> OpportunityReranker:
        """Fit on validation candidates scored by every available method."""

        if label_col not in validation_df.columns:
            raise ValueError(f"validation_df must contain '{label_col}'")
        labels = pd.to_numeric(validation_df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Reranker training requires at least one positive and one negative")

        if self.feature_columns is None:
            self.feature_columns = self._numeric_features(validation_df)
        self.feature_columns = [
            column
            for column in self.feature_columns
            if column in validation_df.columns
            and pd.to_numeric(validation_df[column], errors="coerce").nunique(dropna=True) > 1
        ]
        if not self.feature_columns:
            raise ValueError("No numeric reranking features are available")
        missing = [column for column in self.feature_columns if column not in validation_df.columns]
        if missing:
            raise ValueError(f"Missing reranker features: {missing}")

        clean_frame = validation_df.replace([np.inf, -np.inf], np.nan)
        self.models = []
        self.score_calibrators = []
        base = self._fit_model(clean_frame, labels)
        self.models.append(base)
        if self.model_type == "pairwise":
            self.score_calibrators.append(
                self._fit_score_calibrator(base, clean_frame, labels)
            )

        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_bootstrap):
            for _attempt in range(20):
                indices = rng.integers(0, len(validation_df), len(validation_df))
                sampled_labels = labels[indices]
                if len(np.unique(sampled_labels)) == 2:
                    break
            else:
                continue
            model = self._fit_model(clean_frame.iloc[indices], sampled_labels)
            self.models.append(model)
            if self.model_type == "pairwise":
                self.score_calibrators.append(
                    self._fit_score_calibrator(model, clean_frame, labels)
                )

        self.training_rows = len(validation_df)
        self.training_queries = int(self._query_ids(validation_df).nunique())
        self.positive_rate = float(labels.mean())
        return self

    def predict_distribution(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Return probability and rank distributions across bootstrap models."""

        if not self.models or not self.feature_columns:
            raise RuntimeError("OpportunityReranker must be fitted before prediction")
        missing = [column for column in self.feature_columns if column not in candidates.columns]
        if missing:
            raise ValueError(f"Missing reranker features: {missing}")
        features = self._transform_features(
            candidates[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        )
        model_scores = []
        probabilities = []
        for model_index, model in enumerate(self.models):
            if hasattr(model, "decision_function"):
                score = np.asarray(model.decision_function(features), dtype=float)
            else:
                probability = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
                clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
                score = np.log(clipped / (1.0 - clipped))
            if self.model_type == "pairwise":
                if not self.has_probability_calibration:
                    raise RuntimeError(
                        "This saved pairwise reranker predates probability calibration; "
                        "refit it before generating opportunities."
                    )
                percentile = self._within_query_percentiles(score, candidates).reshape(-1, 1)
                probability = np.asarray(
                    self.score_calibrators[model_index].predict_proba(percentile)[:, 1],
                    dtype=float,
                )
            elif hasattr(model, "decision_function"):
                probability = 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))
            model_scores.append(score)
            probabilities.append(probability)
        score_matrix = np.column_stack(model_scores)
        probability_matrix = np.column_stack(probabilities)
        ranks = np.column_stack(
            [pd.Series(-score_matrix[:, idx]).rank(method="average").to_numpy()
             for idx in range(score_matrix.shape[1])]
        )
        return pd.DataFrame(
            {
                "ranking_score": score_matrix.mean(axis=1),
                "predicted_probability": probability_matrix.mean(axis=1),
                "probability_lower": np.quantile(probability_matrix, 0.05, axis=1),
                "probability_upper": np.quantile(probability_matrix, 0.95, axis=1),
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
        # Keep the dashboard score bounded while sorting by the unsaturated
        # pairwise decision value.
        result["score"] = result["predicted_probability"]
        result = result.sort_values(
            ["ranking_score", "average_rank"], ascending=[False, True]
        ).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
        result["ensemble_score"] = result["predicted_probability"]
        result["ensemble_rank"] = result["rank"]
        result["ensemble_method"] = f"validation_{self.model_type}_bootstrap"
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
        # Migrate rerankers saved before pairwise/query-aware training existed.
        if not hasattr(model, "model_type"):
            model.model_type = "logistic"
        if not hasattr(model, "max_pairs_per_query"):
            model.max_pairs_per_query = 200
        if not hasattr(model, "training_queries"):
            model.training_queries = 1
        if not hasattr(model, "score_calibrators"):
            model.score_calibrators = []
        return model
