"""Composable LinkPredictor: encoder + decoder, optionally variational (GAE/VGAE).

This is the top-level model assembled from an encoder and a decoder.
It supports both GAE (deterministic) and VGAE (variational) modes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class LinkPredictor(nn.Module):
    """Encoder-decoder link-prediction model.

    Parameters
    ----------
    encoder : GNN encoder module.
    decoder : Decoder module (dot_product / distmult / mlp).
    variational : If True, encoder outputs (mu, logstd) and KL loss is added.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        variational: bool = False,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Run the encoder; return node-type → embedding dict."""
        raise NotImplementedError

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_label_index: torch.Tensor,
        src_type: str,
        dst_type: str,
    ) -> torch.Tensor:
        """Score candidate edges using the decoder."""
        raise NotImplementedError

    def forward(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        src_type: str,
        dst_type: str,
    ) -> torch.Tensor:
        """Full forward pass: encode → decode."""
        raise NotImplementedError


def build_model(model_cfg: dict[str, Any], data: HeteroData) -> LinkPredictor:
    """Factory: build a LinkPredictor from a model config dict."""
    raise NotImplementedError
