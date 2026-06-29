"""Build an in-memory NetworkX graph for prototyping and classical heuristics.

This serves as the fallback when Neo4j is not available and also as the
input for computing Adamic-Adar, Jaccard, and common-neighbour baselines.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_bipartite_graph(
    exports_df: pd.DataFrame,
    country_attrs: pd.DataFrame | None = None,
    product_attrs: pd.DataFrame | None = None,
) -> nx.Graph:
    """Build a bipartite Country–Product graph from export data.

    Parameters
    ----------
    exports_df : columns [iso3, hs4, export_value, rca, year].
    country_attrs : optional country-level features to attach.
    product_attrs : optional product-level features to attach.
    """
    raise NotImplementedError


def build_full_hetero_graph(
    exports_df: pd.DataFrame,
    bilateral_df: pd.DataFrame | None = None,
    gravity_df: pd.DataFrame | None = None,
    proximity_df: pd.DataFrame | None = None,
) -> nx.MultiDiGraph:
    """Build the full heterogeneous multi-relational graph.

    Node types: Country, Product, ProductSection.
    Edge types: EXPORTS, IMPORTS, TRADES_WITH, SHIPS, PROXIMITY,
    IN_SECTION, NEIGHBOR_OF.
    """
    raise NotImplementedError


def classical_link_baselines(
    G: nx.Graph,
    candidate_edges: list[tuple[str, str]],
) -> pd.DataFrame:
    """Compute Adamic-Adar, Jaccard, and common-neighbour scores.

    Returns a DataFrame with columns: node_u, node_v, adamic_adar, jaccard, cn.
    """
    raise NotImplementedError
