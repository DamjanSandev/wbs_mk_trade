"""Convert the knowledge graph to PyTorch Geometric HeteroData objects.

Builds node feature tensors, edge-index tensors, and edge attributes
for the heterogeneous graph consumed by the GNN models.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def build_hetero_data(
    exports_df: pd.DataFrame,
    country_features: pd.DataFrame,
    product_features: pd.DataFrame,
    bilateral_df: pd.DataFrame | None = None,
    gravity_df: pd.DataFrame | None = None,
    proximity_df: pd.DataFrame | None = None,
) -> HeteroData:
    """Assemble a PyG HeteroData object from processed DataFrames.

    Node types & features:
        'country': GDP, GDP_pc, population, ECI, diversity, region_onehot,
                   landlocked, EU, CEFTA, [structural_embedding].
        'product': PCI, ubiquity, hs_section_onehot, [structural_embedding].

    Edge types:
        ('country','exports','product'), ('country','imports','product'),
        ('country','trades_with','country'), ('product','proximity','product'),
        ('country','neighbor_of','country').
    """
    raise NotImplementedError


def add_structural_embeddings(
    data: HeteroData,
    country_emb: torch.Tensor | None = None,
    product_emb: torch.Tensor | None = None,
) -> HeteroData:
    """Concatenate FastRP / Node2Vec embeddings to node feature matrices."""
    raise NotImplementedError
