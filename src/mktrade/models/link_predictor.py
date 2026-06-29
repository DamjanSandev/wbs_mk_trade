"""Composable LinkPredictor: encoder + decoder.

For SAGE/GCN: HomoEncoder + to_hetero() (SAGEConv handles bipartite natively).
For GAT: GATEncoder (natively heterogeneous, with per-node-type projections).
For HGT: HGTEncoder (natively heterogeneous).
"""

from __future__ import annotations

from typing import Any

import yaml
import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

from mktrade.config import PROJECT_ROOT
from mktrade.models.decoders import DistMultDecoder, DotProductDecoder, MLPDecoder
from mktrade.models.encoders import GATEncoder, HGTEncoder, HomoEncoder


class LinkPredictor(nn.Module):
    """Encoder-decoder link-prediction model."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._is_native_hetero = isinstance(encoder, (HGTEncoder, GATEncoder))

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Run the encoder; return node-type -> embedding dict."""
        x_dict = {nt: data[nt].x for nt in data.node_types}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        return self.encoder(x_dict, edge_index_dict)

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_label_index: torch.Tensor,
        src_type: str,
        dst_type: str,
    ) -> torch.Tensor:
        """Score candidate edges using the decoder."""
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        return self.decoder(z_src, z_dst)

    def forward(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        src_type: str = "country",
        dst_type: str = "product",
    ) -> torch.Tensor:
        """Full forward pass: encode -> decode."""
        z_dict = self.encode(data)
        return self.decode(z_dict, edge_label_index, src_type, dst_type)


def build_model(model_name: str, data: HeteroData) -> LinkPredictor:
    """Factory: build a LinkPredictor from a model config YAML.

    Parameters
    ----------
    model_name : one of 'graphsage', 'gat', 'gcn', 'rgcn', 'hgt', 'vgae'.
    data : HeteroData to extract metadata and feature dimensions from.
    """
    cfg_path = PROJECT_ROOT / "configs" / "model" / f"{model_name}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    enc_cfg = cfg["encoder"]
    dec_cfg = cfg["decoder"]
    hidden = enc_cfg.get("hidden_channels", 128)
    num_layers = enc_cfg.get("num_layers", 2)
    dropout = enc_cfg.get("dropout", 0.3)

    metadata = data.metadata()

    # Build encoder
    enc_name = enc_cfg.get("name", "GraphSAGE")

    if enc_name == "HGT":
        encoder = HGTEncoder(
            hidden_channels=hidden,
            out_channels=hidden,
            num_heads=enc_cfg.get("heads", 4),
            num_layers=num_layers,
            node_types=list(data.node_types),
            metadata=metadata,
            dropout=dropout,
        )
    elif enc_name == "GAT":
        # Native heterogeneous GAT with per-node-type projections
        encoder = GATEncoder(
            hidden_channels=hidden,
            out_channels=hidden,
            num_heads=enc_cfg.get("heads", 4),
            num_layers=num_layers,
            node_types=list(data.node_types),
            metadata=metadata,
            dropout=dropout,
        )
    else:
        # SAGE, GCN, VGAE all use HomoEncoder (SAGEConv) + to_hetero
        homo_encoder = HomoEncoder(
            in_channels=-1,
            hidden_channels=hidden,
            out_channels=hidden,
            num_layers=num_layers,
            dropout=dropout,
        )
        encoder = to_hetero(homo_encoder, metadata, aggr="sum")

    # Build decoder
    dec_name = dec_cfg.get("name", "dot_product")
    if dec_name == "dot_product":
        decoder = DotProductDecoder()
    elif dec_name == "distmult":
        decoder = DistMultDecoder(hidden, num_relations=1)
    elif dec_name == "mlp":
        decoder = MLPDecoder(hidden, mlp_hidden=dec_cfg.get("mlp_hidden", 64))
    else:
        raise ValueError(f"Unknown decoder: {dec_name}")

    model = LinkPredictor(encoder, decoder)

    # Dummy forward pass to initialize lazy modules
    with torch.no_grad():
        edge_label_index = torch.tensor([[0], [0]], dtype=torch.long)
        model(data, edge_label_index, "country", "product")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Built model '{model_name}': {n_params:,} parameters "
                f"(encoder={enc_name}, decoder={dec_name})")

    return model
