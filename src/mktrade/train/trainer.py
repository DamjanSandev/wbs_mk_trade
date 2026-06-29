"""Training loop with early stopping, MLflow logging, and checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import HeteroData

from mktrade.config import TrainConfig
from mktrade.models.link_predictor import LinkPredictor


class Trainer:
    """Manages the train/val loop for link-prediction models.

    Features:
    - Mini-batch training via LinkNeighborLoader.
    - Early stopping on validation AP.
    - MLflow metric/param logging.
    - Best-model checkpointing.
    """

    def __init__(
        self,
        model: LinkPredictor,
        train_data: HeteroData,
        val_data: HeteroData,
        cfg: TrainConfig,
    ) -> None:
        raise NotImplementedError

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch; return {loss, ...}."""
        raise NotImplementedError

    def validate(self) -> dict[str, float]:
        """Run validation; return {val_loss, val_auc, val_ap, ...}."""
        raise NotImplementedError

    def fit(self) -> Path:
        """Full training loop with early stopping. Returns path to best checkpoint."""
        raise NotImplementedError
