"""Temporal and random link-split strategies.

Temporal: train on edges with year <= T, validate on T+1, test on T+2..T+k.
Random: use PyG's RandomLinkSplit with configurable ratios.
"""

from __future__ import annotations

from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

from mktrade.config import TrainConfig


def temporal_split(
    data: HeteroData,
    cfg: TrainConfig,
    edge_type: tuple[str, str, str] = ("country", "exports", "product"),
) -> tuple[HeteroData, HeteroData, HeteroData]:
    """Split edges by year into train / val / test HeteroData objects.

    Positive supervision edges in val/test are removed from the message-
    passing graph; negative samples are drawn uniformly.
    """
    raise NotImplementedError


def random_split(
    data: HeteroData,
    cfg: TrainConfig,
    edge_type: tuple[str, str, str] = ("country", "exports", "product"),
) -> tuple[HeteroData, HeteroData, HeteroData]:
    """Apply PyG RandomLinkSplit on the specified edge type."""
    raise NotImplementedError
