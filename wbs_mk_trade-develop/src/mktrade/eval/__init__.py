"""Evaluation metrics and baseline comparison."""

from mktrade.eval.metrics import compute_all_metrics, compute_query_metrics
from mktrade.eval.compare import compare_models
from mktrade.eval.backtest import aggregate_backtest_metrics, make_backtest_windows

__all__ = [
    "compute_all_metrics",
    "compute_query_metrics",
    "compare_models",
    "aggregate_backtest_metrics",
    "make_backtest_windows",
]
