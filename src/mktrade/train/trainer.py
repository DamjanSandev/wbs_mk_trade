"""Training loop with early stopping, logging, and checkpointing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch_geometric.data import HeteroData

from mktrade.config import TrainConfig
from mktrade.eval.metrics import compute_all_metrics
from mktrade.models.link_predictor import LinkPredictor


class Trainer:
    """Manages the train/val loop for link-prediction models.

    Features:
    - Full-batch training (graph fits in memory).
    - Early stopping on validation Average Precision.
    - Best-model checkpointing to models/ directory.
    """

    def __init__(
        self,
        model: LinkPredictor,
        train_data: HeteroData,
        val_data: HeteroData,
        cfg: TrainConfig,
        model_name: str = "model",
        checkpoint_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.cfg = cfg
        self.model_name = model_name

        # Device
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        self.model = self.model.to(self.device)
        self.train_data = self.train_data.to(self.device)
        self.val_data = self.val_data.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.checkpoint_dir = checkpoint_dir or Path("models")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.edge_type = ("country", "exports", "product")
        self.src_type = "country"
        self.dst_type = "product"

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch; return {loss}."""
        self.model.train()
        self.optimizer.zero_grad()

        edge_label_index = self.train_data[self.edge_type].edge_label_index
        edge_label = self.train_data[self.edge_type].edge_label

        pred = self.model(self.train_data, edge_label_index, self.src_type, self.dst_type)
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)

        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation; return metrics dict."""
        self.model.eval()

        edge_label_index = self.val_data[self.edge_type].edge_label_index
        edge_label = self.val_data[self.edge_type].edge_label

        pred = self.model(self.val_data, edge_label_index, self.src_type, self.dst_type)
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)

        scores = torch.sigmoid(pred).cpu().numpy()
        labels = edge_label.cpu().numpy()

        metrics = compute_all_metrics(scores, labels, ks=[10, 20, 50, 100])
        metrics["val_loss"] = loss.item()
        return metrics

    def fit(self) -> Path:
        """Full training loop with early stopping. Returns path to best checkpoint."""
        best_ap = -1.0
        patience_counter = 0
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"

        logger.info(f"Training {self.model_name} on {self.device} "
                    f"(epochs={self.cfg.epochs}, patience={self.cfg.patience})")

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            ap = val_metrics["avg_precision"]
            auc = val_metrics["roc_auc"]

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"  Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
                    f"val_AP={ap:.4f} val_AUC={auc:.4f}"
                )

            if ap > best_ap:
                best_ap = ap
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_ap": ap,
                    "val_auc": auc,
                }, best_path)
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.patience:
                logger.info(f"  Early stopping at epoch {epoch} (best AP={best_ap:.4f})")
                break

        # Load best model
        checkpoint = torch.load(best_path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"  Best model: epoch {checkpoint['epoch']}, AP={checkpoint['val_ap']:.4f}")

        return best_path

    @torch.no_grad()
    def evaluate(self, test_data: HeteroData) -> dict[str, float]:
        """Evaluate the model on test data. Returns full metrics dict."""
        self.model.eval()
        test_data = test_data.to(self.device)

        edge_label_index = test_data[self.edge_type].edge_label_index
        edge_label = test_data[self.edge_type].edge_label

        pred = self.model(test_data, edge_label_index, self.src_type, self.dst_type)

        scores = torch.sigmoid(pred).cpu().numpy()
        labels = edge_label.cpu().numpy()

        return compute_all_metrics(scores, labels, ks=[10, 20, 50, 100])
