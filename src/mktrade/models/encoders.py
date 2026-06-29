"""GNN encoder modules: GraphSAGE, GAT, GCN, HGT.

Each encoder maps node features to latent embeddings.

HomoEncoder supports SAGEConv and is converted to heterogeneous via
PyG's to_hetero(). GCNConv is replaced by SAGEConv for heterogeneous
graphs (GCN doesn't support bipartite edges).

GATEncoder is natively heterogeneous with per-node-type projections.
HGTEncoder is natively heterogeneous.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HGTConv, SAGEConv, Linear


class HomoEncoder(nn.Module):
    """Generic homogeneous GNN encoder (SAGEConv-based).

    Wrapped by to_hetero() at model-build time for heterogeneous graphs.
    Always uses SAGEConv which supports bipartite message passing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATEncoder(nn.Module):
    """Natively heterogeneous GAT encoder.

    Uses per-node-type linear projections to a shared hidden dimension,
    then applies GATConv layers. This avoids the bipartite dimension
    mismatch issue with to_hetero() + GATConv.
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        node_types: list[str],
        metadata: tuple,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.node_types = node_types
        self.edge_types = metadata[1]

        # Per-node-type input projection
        self.lin_dict = nn.ModuleDict()
        for nt in node_types:
            self.lin_dict[nt] = Linear(-1, hidden_channels)

        # Per-layer, per-edge-type GAT convolutions
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            conv_dict = nn.ModuleDict()
            for et in metadata[1]:
                key = "__".join(et)
                if layer_idx < num_layers - 1:
                    conv_dict[key] = GATConv(
                        hidden_channels, hidden_channels // num_heads,
                        heads=num_heads, concat=True, add_self_loops=False,
                    )
                else:
                    conv_dict[key] = GATConv(
                        hidden_channels, out_channels,
                        heads=1, concat=False, add_self_loops=False,
                    )
            self.convs.append(conv_dict)

    def forward(
        self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Project inputs
        x_dict = {nt: F.relu(self.lin_dict[nt](x)) for nt, x in x_dict.items()}

        for layer_dict in self.convs[:-1]:
            out_dict = {nt: [] for nt in self.node_types}
            for et, ei in edge_index_dict.items():
                key = "__".join(et)
                if key in layer_dict:
                    src_type, _, dst_type = et
                    out = layer_dict[key]((x_dict[src_type], x_dict[dst_type]), ei)
                    out_dict[dst_type].append(out)

            # Aggregate messages from different edge types (mean)
            new_x = {}
            for nt in self.node_types:
                if out_dict[nt]:
                    new_x[nt] = torch.stack(out_dict[nt]).mean(dim=0)
                    new_x[nt] = F.relu(new_x[nt])
                    new_x[nt] = F.dropout(new_x[nt], p=self.dropout, training=self.training)
                else:
                    new_x[nt] = x_dict[nt]
            x_dict = new_x

        # Last layer
        out_dict = {nt: [] for nt in self.node_types}
        last_layer = self.convs[-1]
        for et, ei in edge_index_dict.items():
            key = "__".join(et)
            if key in last_layer:
                src_type, _, dst_type = et
                out = last_layer[key]((x_dict[src_type], x_dict[dst_type]), ei)
                out_dict[dst_type].append(out)

        result = {}
        for nt in self.node_types:
            if out_dict[nt]:
                result[nt] = torch.stack(out_dict[nt]).mean(dim=0)
            else:
                result[nt] = x_dict[nt]

        return result


class HGTEncoder(nn.Module):
    """Heterogeneous Graph Transformer encoder (natively heterogeneous)."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        node_types: list[str],
        metadata: tuple,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
            )

        self.out_lin = nn.ModuleDict()
        for node_type in node_types:
            self.out_lin[node_type] = Linear(hidden_channels, out_channels)

    def forward(
        self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        x_dict = {key: F.relu(self.lin_dict[key](x)) for key, x in x_dict.items()}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training)
                      for key, x in x_dict.items()}

        return {key: self.out_lin[key](x) for key, x in x_dict.items()}
