"""Model explanations for predicted opportunities.

Uses GNNExplainer and attention-weight extraction to explain
why a particular link was predicted.
"""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, GNNExplainer

from mktrade.models.link_predictor import LinkPredictor


def explain_link(
    model: LinkPredictor,
    data: HeteroData,
    src_idx: int,
    dst_idx: int,
    src_type: str = "country",
    dst_type: str = "product",
) -> dict[str, Any]:
    """Explain a single predicted link using GNNExplainer.

    Returns a dict with:
        - feature_importance: per-feature scores for src and dst nodes.
        - edge_mask: importance of neighbouring edges.
        - subgraph: the computation subgraph used.
    """
    raise NotImplementedError


def extract_attention_weights(
    model: LinkPredictor,
    data: HeteroData,
) -> dict[str, torch.Tensor]:
    """Extract attention weights from GAT/HGT layers (if applicable)."""
    raise NotImplementedError
