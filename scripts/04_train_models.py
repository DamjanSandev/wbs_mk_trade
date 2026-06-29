"""Phase 4 — Train GNN models and run baselines.

Trains all configured encoder-decoder combinations, evaluates on the
temporal holdout, compares against baselines, and logs to MLflow.

Usage:
    python scripts/04_train_models.py                   # train + eval
    python scripts/04_train_models.py --eval-only       # eval only
    python scripts/04_train_models.py --model gat       # single model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mktrade.config import load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN link-prediction models")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate only")
    parser.add_argument("--model", type=str, default=None, help="Train a single model config")
    parser.add_argument("--task", type=str, default=None, help="task_a | task_b | both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config()
    if args.task:
        cfg.task = args.task
    # TODO: implement in Phase 4
    raise NotImplementedError("Phase 4: implement training pipeline")


if __name__ == "__main__":
    main()
