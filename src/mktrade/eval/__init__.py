"""Evaluation metrics and baseline comparison."""

from mktrade.eval.metrics import compute_all_metrics
from mktrade.eval.compare import compare_models

__all__ = ["compute_all_metrics", "compare_models"]
