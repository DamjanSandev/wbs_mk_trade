"""Tests for GNN model building and forward pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from mktrade.models.decoders import DistMultDecoder, DotProductDecoder, MLPDecoder
from mktrade.models.encoders import HomoEncoder


@pytest.fixture
def mini_hetero_data() -> HeteroData:
    """Minimal HeteroData for testing model forward pass."""
    data = HeteroData()

    # 5 countries, 10 products, 3 sections
    data["country"].x = torch.randn(5, 14)
    data["country"].num_nodes = 5
    data["product"].x = torch.randn(10, 23)
    data["product"].num_nodes = 10
    data["product_section"].x = torch.randn(3, 21)
    data["product_section"].num_nodes = 3

    # Exports: 20 edges
    data["country", "exports", "product"].edge_index = torch.randint(0, 5, (2, 20))
    data["country", "exports", "product"].edge_index[1] = torch.randint(0, 10, (20,))

    # Reverse
    data["product", "rev_exports", "country"].edge_index = torch.stack([
        data["country", "exports", "product"].edge_index[1],
        data["country", "exports", "product"].edge_index[0],
    ])

    # In section
    data["product", "in_section", "product_section"].edge_index = torch.stack([
        torch.arange(10), torch.randint(0, 3, (10,))
    ])
    data["product_section", "rev_in_section", "product"].edge_index = torch.stack([
        data["product", "in_section", "product_section"].edge_index[1],
        data["product", "in_section", "product_section"].edge_index[0],
    ])

    # Proximity: 15 edges
    src = torch.randint(0, 10, (15,))
    dst = torch.randint(0, 10, (15,))
    data["product", "proximity", "product"].edge_index = torch.stack([src, dst])

    # Neighbor_of: 8 edges
    data["country", "neighbor_of", "country"].edge_index = torch.randint(0, 5, (2, 8))

    # Trades_with: 4 edges
    data["country", "trades_with", "country"].edge_index = torch.randint(0, 5, (2, 4))

    return data


class TestDecoders:
    def test_dot_product(self):
        dec = DotProductDecoder()
        z_src = torch.randn(10, 64)
        z_dst = torch.randn(10, 64)
        scores = dec(z_src, z_dst)
        assert scores.shape == (10,)

    def test_distmult(self):
        dec = DistMultDecoder(64, num_relations=2)
        z_src = torch.randn(10, 64)
        z_dst = torch.randn(10, 64)
        scores = dec(z_src, z_dst, rel_type=0)
        assert scores.shape == (10,)
        scores2 = dec(z_src, z_dst, rel_type=1)
        assert scores2.shape == (10,)

    def test_mlp(self):
        dec = MLPDecoder(64, mlp_hidden=32)
        z_src = torch.randn(10, 64)
        z_dst = torch.randn(10, 64)
        scores = dec(z_src, z_dst)
        assert scores.shape == (10,)


class TestHomoEncoder:
    def test_sage_forward(self):
        enc = HomoEncoder(in_channels=14, hidden_channels=32, out_channels=32, num_layers=2)
        x = torch.randn(5, 14)
        edge_index = torch.randint(0, 5, (2, 10))
        out = enc(x, edge_index)
        assert out.shape == (5, 32)

    def test_three_layer(self):
        enc = HomoEncoder(in_channels=14, hidden_channels=32, out_channels=16, num_layers=3)
        x = torch.randn(5, 14)
        edge_index = torch.randint(0, 5, (2, 10))
        out = enc(x, edge_index)
        assert out.shape == (5, 16)


class TestLinkPredictor:
    def test_build_graphsage(self, mini_hetero_data):
        from mktrade.models.link_predictor import build_model
        model = build_model("graphsage", mini_hetero_data)
        # Forward pass
        edge_label_index = torch.tensor([[0, 1, 2], [3, 5, 7]], dtype=torch.long)
        scores = model(mini_hetero_data, edge_label_index, "country", "product")
        assert scores.shape == (3,)

    def test_build_gcn(self, mini_hetero_data):
        from mktrade.models.link_predictor import build_model
        model = build_model("gcn", mini_hetero_data)
        edge_label_index = torch.tensor([[0, 1], [3, 5]], dtype=torch.long)
        scores = model(mini_hetero_data, edge_label_index, "country", "product")
        assert scores.shape == (2,)

    def test_build_gat(self, mini_hetero_data):
        from mktrade.models.link_predictor import build_model
        model = build_model("gat", mini_hetero_data)
        edge_label_index = torch.tensor([[0, 1], [3, 5]], dtype=torch.long)
        scores = model(mini_hetero_data, edge_label_index, "country", "product")
        assert scores.shape == (2,)

    def test_build_hgt(self, mini_hetero_data):
        from mktrade.models.link_predictor import build_model
        model = build_model("hgt", mini_hetero_data)
        edge_label_index = torch.tensor([[0, 1], [3, 5]], dtype=torch.long)
        scores = model(mini_hetero_data, edge_label_index, "country", "product")
        assert scores.shape == (2,)

    def test_gradient_flows(self, mini_hetero_data):
        """Check that gradients flow through the model."""
        from mktrade.models.link_predictor import build_model
        model = build_model("graphsage", mini_hetero_data)
        edge_label_index = torch.tensor([[0, 1], [3, 5]], dtype=torch.long)
        labels = torch.tensor([1.0, 0.0])

        scores = model(mini_hetero_data, edge_label_index, "country", "product")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()

        # At least some parameters should have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.parameters() if p.requires_grad)
        assert has_grad
