"""Tests for graph construction (NetworkX, PyG, Neo4j graceful skip)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──

@pytest.fixture
def synthetic_exports() -> pd.DataFrame:
    """Minimal 3-country × 4-product export matrix."""
    rows = []
    countries = ["AAA", "BBB", "CCC"]
    products = ["0101", "2701", "8544", "9401"]
    np.random.seed(42)
    for c in countries:
        for p in products:
            rows.append({"year": 2022, "iso3": c, "hs4": p,
                        "export_value": np.random.exponential(1e6),
                        "export_rca": np.random.uniform(0, 3)})
    # Make AAA strongly specialised in 0101
    rows.append({"year": 2022, "iso3": "AAA", "hs4": "0101",
                "export_value": 1e9, "export_rca": 5.0})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_country_features() -> pd.DataFrame:
    return pd.DataFrame([
        {"year": 2022, "iso3": "AAA", "eci": 0.5, "diversity": 100},
        {"year": 2022, "iso3": "BBB", "eci": -0.3, "diversity": 50},
        {"year": 2022, "iso3": "CCC", "eci": 1.2, "diversity": 200},
    ])


@pytest.fixture
def synthetic_product_features() -> pd.DataFrame:
    return pd.DataFrame([
        {"year": 2022, "hs4": "0101", "pci": -1.0, "ubiquity": 50},
        {"year": 2022, "hs4": "2701", "pci": -0.5, "ubiquity": 30},
        {"year": 2022, "hs4": "8544", "pci": 1.0, "ubiquity": 20},
        {"year": 2022, "hs4": "9401", "pci": 0.5, "ubiquity": 25},
    ])


@pytest.fixture
def synthetic_proximity() -> pd.DataFrame:
    return pd.DataFrame([
        {"year": 2022, "hs4_1": "0101", "hs4_2": "2701", "proximity": 0.3},
        {"year": 2022, "hs4_1": "2701", "hs4_2": "0101", "proximity": 0.3},
        {"year": 2022, "hs4_1": "8544", "hs4_2": "9401", "proximity": 0.7},
        {"year": 2022, "hs4_1": "9401", "hs4_2": "8544", "proximity": 0.7},
    ])


@pytest.fixture
def synthetic_bilateral() -> pd.DataFrame:
    return pd.DataFrame([
        {"year": 2022, "reporter_iso3": "AAA", "partner_iso3": "BBB", "hs4": "0101", "flow": "X", "value": 1e6},
        {"year": 2022, "reporter_iso3": "BBB", "partner_iso3": "AAA", "hs4": "2701", "flow": "X", "value": 5e5},
        {"year": 2022, "reporter_iso3": "AAA", "partner_iso3": "BBB", "hs4": "0101", "flow": "M", "value": 2e5},
    ])


# ── NetworkX Tests ──

class TestBuildNX:
    def test_bipartite_graph_structure(self, synthetic_exports, synthetic_country_features, synthetic_product_features):
        from mktrade.graph.build_nx import build_bipartite_graph

        G = build_bipartite_graph(synthetic_exports, synthetic_country_features,
                                  synthetic_product_features, year=2022)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

        # Check bipartite sets
        countries = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}
        products = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 1}
        assert len(countries) == 3
        assert len(products) == 4

        # All edges connect country to product
        for u, v in G.edges():
            assert (u in countries and v in products) or (u in products and v in countries)

    def test_bipartite_node_attributes(self, synthetic_exports, synthetic_country_features, synthetic_product_features):
        from mktrade.graph.build_nx import build_bipartite_graph

        G = build_bipartite_graph(synthetic_exports, synthetic_country_features,
                                  synthetic_product_features, year=2022)
        # Country attrs
        assert G.nodes["AAA"]["eci"] == 0.5
        # Product attrs
        assert G.nodes["P_8544"]["pci"] == 1.0

    def test_full_hetero_graph(self, synthetic_exports, synthetic_bilateral, synthetic_proximity,
                               synthetic_country_features, synthetic_product_features):
        from mktrade.graph.build_nx import build_full_hetero_graph

        G = build_full_hetero_graph(
            exports_df=synthetic_exports,
            bilateral_df=synthetic_bilateral,
            proximity_df=synthetic_proximity,
            country_attrs=synthetic_country_features,
            product_attrs=synthetic_product_features,
            year=2022,
        )
        assert G.is_directed()
        assert G.is_multigraph()

        # Check node types present
        node_types = {d.get("node_type") for _, d in G.nodes(data=True)}
        assert "country" in node_types
        assert "product" in node_types
        assert "product_section" in node_types

        # Check edge types present
        edge_types = {d.get("edge_type") for _, _, d in G.edges(data=True)}
        assert "EXPORTS" in edge_types
        assert "TRADES_WITH" in edge_types
        assert "PROXIMITY" in edge_types
        assert "IN_SECTION" in edge_types

    def test_classical_baselines(self, synthetic_exports):
        from mktrade.graph.build_nx import build_bipartite_graph, classical_link_baselines

        G = build_bipartite_graph(synthetic_exports, year=2022)
        products = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
        # Candidate: edges that don't exist for AAA
        existing = set(G.neighbors("AAA"))
        candidates = [("AAA", p) for p in products if p not in existing]

        if candidates:
            df = classical_link_baselines(G, candidates)
            assert "adamic_adar" in df.columns
            assert "jaccard" in df.columns
            assert "cn" in df.columns
            assert "pref_attach" in df.columns
        else:
            # All products connected, test with empty candidates
            df = classical_link_baselines(G, [])
            assert df.empty


# ── PyG Tests ──

class TestPyGData:
    def test_hetero_data_construction(self, synthetic_exports, synthetic_country_features,
                                      synthetic_product_features, synthetic_proximity):
        from mktrade.graph.pyg_data import build_hetero_data

        data = build_hetero_data(
            exports_df=synthetic_exports,
            country_features=synthetic_country_features,
            product_features=synthetic_product_features,
            proximity_df=synthetic_proximity,
            year=2022,
        )

        # Node types
        assert "country" in data.node_types
        assert "product" in data.node_types
        assert "product_section" in data.node_types

        # Node counts
        assert data["country"].num_nodes == 3
        assert data["product"].num_nodes == 4

        # Feature dimensions
        assert data["country"].x.shape[0] == 3
        assert data["country"].x.shape[1] > 0
        assert data["product"].x.shape[0] == 4
        assert data["product"].x.shape[1] > 0

    def test_export_edges_present(self, synthetic_exports, synthetic_country_features,
                                  synthetic_product_features):
        from mktrade.graph.pyg_data import build_hetero_data

        data = build_hetero_data(
            exports_df=synthetic_exports,
            country_features=synthetic_country_features,
            product_features=synthetic_product_features,
            year=2022,
        )

        # EXPORTS edges should exist
        assert ("country", "exports", "product") in data.edge_types
        ei = data["country", "exports", "product"].edge_index
        assert ei.shape[0] == 2
        assert ei.shape[1] > 0

        # Reverse edges
        assert ("product", "rev_exports", "country") in data.edge_types

    def test_proximity_edges(self, synthetic_exports, synthetic_country_features,
                             synthetic_product_features, synthetic_proximity):
        from mktrade.graph.pyg_data import build_hetero_data

        data = build_hetero_data(
            exports_df=synthetic_exports,
            country_features=synthetic_country_features,
            product_features=synthetic_product_features,
            proximity_df=synthetic_proximity,
            year=2022,
        )

        assert ("product", "proximity", "product") in data.edge_types
        ei = data["product", "proximity", "product"].edge_index
        assert ei.shape[1] == 4  # 4 directed proximity edges

    def test_bilateral_edges(self, synthetic_exports, synthetic_country_features,
                             synthetic_product_features, synthetic_bilateral):
        from mktrade.graph.pyg_data import build_hetero_data

        data = build_hetero_data(
            exports_df=synthetic_exports,
            country_features=synthetic_country_features,
            product_features=synthetic_product_features,
            bilateral_df=synthetic_bilateral,
            year=2022,
        )

        assert ("country", "trades_with", "country") in data.edge_types

    def test_section_onehot_valid(self, synthetic_exports, synthetic_country_features,
                                  synthetic_product_features):
        """Product features should have exactly one section one-hot bit set."""
        from mktrade.graph.pyg_data import build_hetero_data

        data = build_hetero_data(
            exports_df=synthetic_exports,
            country_features=synthetic_country_features,
            product_features=synthetic_product_features,
            year=2022,
        )

        # Section one-hot starts at index 2
        section_onehot = data["product"].x[:, 2:]
        # Each product should have exactly one section bit
        assert (section_onehot.sum(dim=1) == 1.0).all()

    def test_no_nan_in_features(self, synthetic_exports, synthetic_country_features,
                                synthetic_product_features):
        from mktrade.graph.pyg_data import build_hetero_data

        data = build_hetero_data(
            exports_df=synthetic_exports,
            country_features=synthetic_country_features,
            product_features=synthetic_product_features,
            year=2022,
        )

        assert not data["country"].x.isnan().any()
        assert not data["product"].x.isnan().any()

    def test_save_load_roundtrip(self, tmp_path, synthetic_exports, synthetic_country_features,
                                 synthetic_product_features):
        from mktrade.graph.pyg_data import build_hetero_data, load_hetero_data, save_hetero_data

        data = build_hetero_data(
            exports_df=synthetic_exports,
            country_features=synthetic_country_features,
            product_features=synthetic_product_features,
            year=2022,
        )

        path = tmp_path / "test_data.pt"
        save_hetero_data(data, path)
        loaded = load_hetero_data(path)

        assert loaded["country"].num_nodes == data["country"].num_nodes
        assert loaded["product"].num_nodes == data["product"].num_nodes
        assert (loaded["country"].x == data["country"].x).all()

    def test_enrich_country_features(self, synthetic_exports, synthetic_country_features,
                                     synthetic_product_features):
        from mktrade.graph.pyg_data import build_hetero_data, enrich_country_features

        data = build_hetero_data(
            exports_df=synthetic_exports,
            country_features=synthetic_country_features,
            product_features=synthetic_product_features,
            year=2022,
        )

        wdi = pd.DataFrame([
            {"iso3": "AAA", "year": 2022, "gdp": 1e10, "gdp_pc": 5000, "population": 2e6},
            {"iso3": "BBB", "year": 2022, "gdp": 5e10, "gdp_pc": 15000, "population": 5e6},
        ])

        data = enrich_country_features(data, wdi, data["country"].iso3, year=2022)
        # AAA should have non-zero GDP features
        aaa_idx = data["country"].iso3.index("AAA")
        assert data["country"].x[aaa_idx, 2].item() > 0  # log GDP
        assert data["country"].x[aaa_idx, 3].item() > 0  # log GDP/cap
        assert data["country"].x[aaa_idx, 4].item() > 0  # log population


# ── Neo4j Graceful Skip ──

class TestNeo4jGraceful:
    def test_neo4j_loader_unavailable(self):
        """Neo4jLoader should gracefully handle missing Neo4j."""
        from mktrade.config import Neo4jConfig
        from mktrade.graph.build_neo4j import Neo4jLoader

        cfg = Neo4jConfig(uri="bolt://localhost:99999", password="wrong")
        loader = Neo4jLoader(cfg)
        assert not loader.available

        # All methods should be no-ops
        loader.load_countries(pd.DataFrame({"iso3": ["AAA"]}))
        loader.load_products(pd.DataFrame({"hs4": ["0101"]}))
        loader.close()
