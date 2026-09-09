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


def pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_ids: torch.Tensor,
    max_negatives: int = 20,
) -> torch.Tensor:
    """RankNet-style loss using each query's highest-scoring negatives."""

    losses: list[torch.Tensor] = []
    for query in torch.unique(query_ids):
        query_mask = query_ids == query
        positives = logits[query_mask & (labels > 0.5)]
        negatives = logits[query_mask & (labels <= 0.5)]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        limit = min(max(1, int(max_negatives)), negatives.numel())
        hard_negatives = torch.topk(negatives, k=limit).values
        differences = positives.unsqueeze(1) - hard_negatives.unsqueeze(0)
        losses.append(F.softplus(-differences).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def ranking_selection_score(metrics: dict[str, float], metric_name: str) -> float:
    """Resolve checkpoint selection, including a balanced ranking composite."""

    if metric_name != "ranking_composite":
        return float(metrics.get(metric_name, metrics["avg_precision"]))
    components = [
        max(float(metrics.get("avg_precision", 0.0)), 0.0),
        max(float(metrics.get("mrr", 0.0)), 0.0),
        max(float(metrics.get("ndcg@10", 0.0)), 0.0),
    ]
    if any(component == 0.0 for component in components):
        return 0.0
    return float(np.prod(components) ** (1.0 / len(components)))


class Trainer:
    """Manages the train/val loop for link-prediction models.

    Features:
    - Full-batch training (graph fits in memory).
    - Early stopping on a configurable validation ranking metric.
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
        positive_count = edge_label.sum().clamp_min(1.0)
        negative_count = (edge_label <= 0.5).sum().float().clamp_min(1.0)
        pos_weight = (negative_count / positive_count).detach()
        classification_loss = F.binary_cross_entropy_with_logits(
            pred, edge_label, pos_weight=pos_weight
        )
        ranking_loss = pairwise_ranking_loss(
            pred,
            edge_label,
            edge_label_index[0],
            max_negatives=self.cfg.ranking_hard_negatives,
        )
        ranking_weight = float(np.clip(self.cfg.ranking_loss_weight, 0.0, 1.0))
        loss = (1.0 - ranking_weight) * classification_loss + ranking_weight * ranking_loss

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "classification_loss": classification_loss.item(),
            "ranking_loss": ranking_loss.item(),
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation; return metrics dict."""
        self.model.eval()

        edge_label_index = self.val_data[self.edge_type].edge_label_index
        edge_label = self.val_data[self.edge_type].edge_label

        pred = self.model(self.val_data, edge_label_index, self.src_type, self.dst_type)
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)

        # Ranking metrics operate on raw logits. This preserves ordering when
        # sigmoid probabilities saturate to exactly zero or one in float32.
        scores = pred.cpu().numpy()
        labels = edge_label.cpu().numpy()

        query_ids = edge_label_index[0].cpu().numpy()
        metrics = compute_all_metrics(
            scores, labels, ks=[10, 20, 50, 100], query_ids=query_ids
        )
        metrics["val_loss"] = loss.item()
        return metrics

    def fit(self) -> Path:
        """Full training loop with early stopping. Returns path to best checkpoint."""
        selection_metric = self.cfg.selection_metric
        best_score = -1.0
        patience_counter = 0
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"

        logger.info(f"Training {self.model_name} on {self.device} "
                    f"(epochs={self.cfg.epochs}, patience={self.cfg.patience})")

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            ap = val_metrics["avg_precision"]
            auc = val_metrics["roc_auc"]
            selection_score = ranking_selection_score(val_metrics, selection_metric)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"  Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} "
                    f"val_AP={ap:.4f} val_AUC={auc:.4f} "
                    f"val_{selection_metric}={selection_score:.4f}"
                )

            if selection_score > best_score:
                best_score = selection_score
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_ap": ap,
                    "val_auc": auc,
                    "val_mrr": val_metrics.get("mrr", 0.0),
                    "val_ndcg_at_10": val_metrics.get("ndcg@10", 0.0),
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                }, best_path)
            else:
                patience_counter += 1

            if patience_counter >= self.cfg.patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(best {selection_metric}={best_score:.4f})"
                )
                break

        # Load best model
        checkpoint = torch.load(best_path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"  Best model: epoch {checkpoint['epoch']}, "
            f"{checkpoint.get('selection_metric', 'avg_precision')}="
            f"{checkpoint.get('selection_score', checkpoint['val_ap']):.4f}"
        )

        return best_path

    @torch.no_grad()
    def evaluate(self, test_data: HeteroData) -> dict[str, float]:
        """Evaluate the model on test data. Returns full metrics dict."""
        self.model.eval()
        test_data = test_data.to(self.device)

        edge_label_index = test_data[self.edge_type].edge_label_index
        edge_label = test_data[self.edge_type].edge_label

        pred = self.model(test_data, edge_label_index, self.src_type, self.dst_type)

        scores = pred.cpu().numpy()
        labels = edge_label.cpu().numpy()

        query_ids = edge_label_index[0].cpu().numpy()
        return compute_all_metrics(
            scores, labels, ks=[10, 20, 50, 100], query_ids=query_ids
        )
