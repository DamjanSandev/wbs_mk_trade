"""Link-prediction decoders: dot-product, DistMult, MLP.

Each decoder takes source and target node embeddings and produces
a link probability score.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DotProductDecoder(nn.Module):
    """Simple dot-product decoder: score = z_u^T z_v."""

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DistMultDecoder(nn.Module):
    """DistMult decoder: score = z_u^T diag(R) z_v."""

    def __init__(self, hidden_channels: int, num_relations: int = 1) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self, z_src: torch.Tensor, z_dst: torch.Tensor, rel_type: int = 0
    ) -> torch.Tensor:
        raise NotImplementedError


class MLPDecoder(nn.Module):
    """MLP decoder: score = MLP(z_u || z_v)."""

    def __init__(self, hidden_channels: int, mlp_hidden: int = 64) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
