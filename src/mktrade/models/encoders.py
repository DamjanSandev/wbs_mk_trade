"""GNN encoder modules: GraphSAGE, GAT, GCN, R-GCN, HGT.

Each encoder maps node features to latent embeddings.
Heterogeneous wrapping is handled via PyG's to_hetero().
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, HGTConv, RGCNConv, SAGEConv


class HomoEncoder(nn.Module):
    """Generic homogeneous GNN encoder (SAGEConv / GATConv / GCNConv).

    Wrapped by to_hetero() at model-build time for heterogeneous graphs.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        conv_type: str = "SAGEConv",
        heads: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RGCNEncoder(nn.Module):
    """R-GCN encoder for heterogeneous graphs (native multi-relation)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class HGTEncoder(nn.Module):
    """Heterogeneous Graph Transformer encoder."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError
