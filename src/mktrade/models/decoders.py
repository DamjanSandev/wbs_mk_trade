"""Link-prediction decoders: dot-product, DistMult, MLP.

Each decoder takes source and target node embeddings and produces
a link probability score (logit, apply sigmoid for probability).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductDecoder(nn.Module):
    """Simple dot-product decoder: score = z_u^T z_v."""

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return (z_src * z_dst).sum(dim=-1)


class DistMultDecoder(nn.Module):
    """DistMult decoder: score = z_u^T diag(R) z_v.

    Learns a diagonal relation matrix per relation type.
    """

    def __init__(self, hidden_channels: int, num_relations: int = 1) -> None:
        super().__init__()
        self.rel_emb = nn.Parameter(torch.randn(num_relations, hidden_channels))
        nn.init.xavier_uniform_(self.rel_emb)

    def forward(
        self, z_src: torch.Tensor, z_dst: torch.Tensor, rel_type: int = 0
    ) -> torch.Tensor:
        rel = self.rel_emb[rel_type]
        return (z_src * rel * z_dst).sum(dim=-1)


class MLPDecoder(nn.Module):
    """MLP decoder: score = MLP(z_u || z_v)."""

    def __init__(self, hidden_channels: int, mlp_hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels * 2, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_src, z_dst], dim=-1)
        return self.net(h).squeeze(-1)
