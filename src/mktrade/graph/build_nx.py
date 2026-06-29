"""Build in-memory NetworkX graphs for prototyping and classical heuristics.

Two graph variants:
1. Bipartite Country-Product graph (for Task A baselines + product-space viz).
2. Full heterogeneous MultiDiGraph (Country, Product, ProductSection nodes;
   EXPORTS, IMPORTS, TRADES_WITH, PROXIMITY, IN_SECTION, NEIGHBOR_OF edges).

Also provides classical link-prediction baselines: Adamic-Adar, Jaccard,
common neighbours, and preferential attachment.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger


# ── HS section names (2-digit chapter → section) ─────────
# Simplified mapping: first 2 digits of HS code
HS_SECTION_NAMES: dict[str, str] = {
    "01": "Live Animals", "02": "Meat", "03": "Fish", "04": "Dairy",
    "05": "Animal Products", "06": "Live Trees", "07": "Vegetables",
    "08": "Fruits", "09": "Coffee/Tea", "10": "Cereals",
    "11": "Milling Products", "12": "Oil Seeds", "13": "Lac/Gums",
    "14": "Vegetable Plaiting", "15": "Fats/Oils", "16": "Prepared Meat/Fish",
    "17": "Sugars", "18": "Cocoa", "19": "Cereal Preparations",
    "20": "Vegetable Preparations", "21": "Misc. Food", "22": "Beverages",
    "23": "Food Residues", "24": "Tobacco", "25": "Salt/Stone",
    "26": "Ores", "27": "Mineral Fuels", "28": "Inorganic Chemicals",
    "29": "Organic Chemicals", "30": "Pharmaceuticals", "31": "Fertilizers",
    "32": "Dyes/Paints", "33": "Cosmetics", "34": "Soap/Wax",
    "35": "Albuminoidal", "36": "Explosives", "37": "Photographic",
    "38": "Misc. Chemicals", "39": "Plastics", "40": "Rubber",
    "41": "Raw Hides", "42": "Leather Articles", "43": "Fur",
    "44": "Wood", "45": "Cork", "46": "Straw/Plaiting",
    "47": "Wood Pulp", "48": "Paper", "49": "Printed Products",
    "50": "Silk", "51": "Wool", "52": "Cotton", "53": "Vegetable Fibers",
    "54": "Man-Made Filaments", "55": "Man-Made Staple", "56": "Wadding/Felt",
    "57": "Carpets", "58": "Special Woven", "59": "Coated Textiles",
    "60": "Knitted Fabrics", "61": "Knitted Apparel", "62": "Woven Apparel",
    "63": "Other Textiles", "64": "Footwear", "65": "Headgear",
    "66": "Umbrellas", "67": "Feathers", "68": "Stone/Cement",
    "69": "Ceramics", "70": "Glass", "71": "Precious Metals/Stones",
    "72": "Iron/Steel", "73": "Iron/Steel Articles", "74": "Copper",
    "75": "Nickel", "76": "Aluminium", "78": "Lead", "79": "Zinc",
    "80": "Tin", "81": "Other Base Metals", "82": "Tools",
    "83": "Misc. Base Metal", "84": "Machinery", "85": "Electrical Equipment",
    "86": "Railway", "87": "Vehicles", "88": "Aircraft", "89": "Ships",
    "90": "Optical/Medical", "91": "Clocks", "92": "Musical Instruments",
    "93": "Arms", "94": "Furniture", "95": "Toys", "96": "Misc. Manufactured",
    "97": "Art/Antiques",
}


def _hs4_to_section(hs4: str) -> str:
    """Map HS4 code to its 2-digit chapter code."""
    return hs4[:2]


def build_bipartite_graph(
    exports_df: pd.DataFrame,
    country_attrs: pd.DataFrame | None = None,
    product_attrs: pd.DataFrame | None = None,
    year: int | None = None,
) -> nx.Graph:
    """Build a bipartite Country-Product graph from export data.

    Parameters
    ----------
    exports_df : columns [year, iso3, hs4, export_value] + optional [export_rca, density].
    country_attrs : optional country-level features (iso3, eci, diversity, ...).
    product_attrs : optional product-level features (hs4, pci, ubiquity, ...).
    year : if given, filter to this year.

    Returns
    -------
    nx.Graph with bipartite node sets 'country' and 'product'.
    Country nodes prefixed as-is (e.g. 'MKD'), products as 'P_0101'.
    """
    df = exports_df.copy()
    if year is not None:
        df = df[df["year"] == year]

    # Only keep positive exports (optionally with RCA > 1)
    df = df[df["export_value"] > 0]

    G = nx.Graph()

    # Add country nodes
    countries = df["iso3"].unique()
    for c in countries:
        attrs = {"bipartite": 0, "node_type": "country"}
        if country_attrs is not None:
            row = country_attrs[country_attrs["iso3"] == c]
            if not row.empty:
                for col in row.columns:
                    if col != "iso3":
                        val = row[col].iloc[0]
                        if pd.notna(val):
                            attrs[col] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else val
        G.add_node(c, **attrs)

    # Add product nodes
    products = df["hs4"].unique()
    for p in products:
        pnode = f"P_{p}"
        attrs = {"bipartite": 1, "node_type": "product", "hs4": p, "section": _hs4_to_section(p)}
        if product_attrs is not None:
            row = product_attrs[product_attrs["hs4"] == p]
            if not row.empty:
                for col in row.columns:
                    if col != "hs4":
                        val = row[col].iloc[0]
                        if pd.notna(val):
                            attrs[col] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else val
        G.add_node(pnode, **attrs)

    # Add edges
    for _, row in df.iterrows():
        edge_attrs = {"export_value": float(row["export_value"])}
        if "export_rca" in row and pd.notna(row.get("export_rca")):
            edge_attrs["rca"] = float(row["export_rca"])
        if "density" in row and pd.notna(row.get("density")):
            edge_attrs["density"] = float(row["density"])
        G.add_edge(row["iso3"], f"P_{row['hs4']}", **edge_attrs)

    logger.info(
        f"Bipartite graph: {G.number_of_nodes()} nodes "
        f"({len(countries)} countries, {len(products)} products), "
        f"{G.number_of_edges()} edges"
    )
    return G


def build_full_hetero_graph(
    exports_df: pd.DataFrame,
    bilateral_df: pd.DataFrame | None = None,
    gravity_df: pd.DataFrame | None = None,
    proximity_df: pd.DataFrame | None = None,
    country_attrs: pd.DataFrame | None = None,
    product_attrs: pd.DataFrame | None = None,
    year: int | None = None,
) -> nx.MultiDiGraph:
    """Build the full heterogeneous multi-relational graph.

    Node types: Country, Product, ProductSection.
    Edge types: EXPORTS, TRADES_WITH, PROXIMITY, IN_SECTION, NEIGHBOR_OF.
    """
    G = nx.MultiDiGraph()

    df = exports_df.copy()
    if year is not None:
        df = df[df["year"] == year]

    # ── Country nodes ──
    countries = set(df["iso3"].unique())
    if bilateral_df is not None:
        bil = bilateral_df if year is None else bilateral_df[bilateral_df["year"] == year]
        countries |= set(bil["reporter_iso3"].unique()) | set(bil["partner_iso3"].unique())

    for c in countries:
        attrs = {"node_type": "country"}
        if country_attrs is not None:
            row = country_attrs[country_attrs["iso3"] == c]
            if not row.empty:
                for col in row.columns:
                    if col not in ("iso3", "year"):
                        val = row[col].iloc[0]
                        if pd.notna(val):
                            attrs[col] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else val
        G.add_node(f"C_{c}", **attrs)

    # ── Product nodes + ProductSection nodes + IN_SECTION edges ──
    products = df["hs4"].unique()
    sections_added: set[str] = set()

    for p in products:
        sec = _hs4_to_section(p)
        attrs = {"node_type": "product", "hs4": p, "section": sec}
        if product_attrs is not None:
            row = product_attrs[product_attrs["hs4"] == p]
            if not row.empty:
                for col in row.columns:
                    if col not in ("hs4", "year"):
                        val = row[col].iloc[0]
                        if pd.notna(val):
                            attrs[col] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else val
        G.add_node(f"P_{p}", **attrs)

        # ProductSection
        if sec not in sections_added:
            sec_name = HS_SECTION_NAMES.get(sec, f"Section {sec}")
            G.add_node(f"S_{sec}", node_type="product_section", code=sec, name=sec_name)
            sections_added.add(sec)
        G.add_edge(f"P_{p}", f"S_{sec}", key="IN_SECTION", edge_type="IN_SECTION")

    # ── EXPORTS edges ──
    df_pos = df[df["export_value"] > 0]
    for _, row in df_pos.iterrows():
        edge_attrs = {"edge_type": "EXPORTS", "value": float(row["export_value"])}
        if "export_rca" in row and pd.notna(row.get("export_rca")):
            edge_attrs["rca"] = float(row["export_rca"])
        G.add_edge(f"C_{row['iso3']}", f"P_{row['hs4']}", key="EXPORTS", **edge_attrs)

    # ── TRADES_WITH edges (bilateral aggregated by partner) ──
    if bilateral_df is not None:
        bil = bilateral_df if year is None else bilateral_df[bilateral_df["year"] == year]
        if not bil.empty:
            # Aggregate by reporter-partner pair
            agg = bil.groupby(["reporter_iso3", "partner_iso3", "flow"])["value"].sum().reset_index()
            exports_agg = agg[agg["flow"] == "X"].rename(columns={"value": "export_value"})
            imports_agg = agg[agg["flow"] == "M"].rename(columns={"value": "import_value"})

            merged = exports_agg.merge(
                imports_agg[["reporter_iso3", "partner_iso3", "import_value"]],
                on=["reporter_iso3", "partner_iso3"], how="outer",
            ).fillna(0)

            for _, row in merged.iterrows():
                r, p = f"C_{row['reporter_iso3']}", f"C_{row['partner_iso3']}"
                if r != p and G.has_node(r) and G.has_node(p):
                    G.add_edge(r, p, key="TRADES_WITH", edge_type="TRADES_WITH",
                               export_value=float(row.get("export_value", 0)),
                               import_value=float(row.get("import_value", 0)))

    # ── PROXIMITY edges (product-product) ──
    if proximity_df is not None and not proximity_df.empty:
        prox = proximity_df.copy()
        # Filter to significant proximities only (keep top edges)
        prox = prox[(prox["proximity"] > 0) & (prox["hs4_1"] != prox["hs4_2"])]
        for _, row in prox.iterrows():
            p1, p2 = f"P_{row['hs4_1']}", f"P_{row['hs4_2']}"
            if G.has_node(p1) and G.has_node(p2):
                G.add_edge(p1, p2, key="PROXIMITY", edge_type="PROXIMITY",
                           weight=float(row["proximity"]))

    # ── NEIGHBOR_OF edges (gravity country pairs) ──
    if gravity_df is not None and not gravity_df.empty:
        grav = gravity_df.copy()
        if "year" in grav.columns:
            if year is not None and year in grav["year"].values:
                grav = grav[grav["year"] == year]
            else:
                latest = grav["year"].max()
                grav = grav[grav["year"] == latest]
                if year is not None:
                    logger.info(f"Gravity: year {year} unavailable, using {latest}")

        for _, row in grav.iterrows():
            o, d = f"C_{row['iso3_o']}", f"C_{row['iso3_d']}"
            if G.has_node(o) and G.has_node(d) and o != d:
                _safe = lambda v, default=0: default if pd.isna(v) else v
                G.add_edge(o, d, key="NEIGHBOR_OF", edge_type="NEIGHBOR_OF",
                           distance=float(_safe(row.get("dist", 0))),
                           contiguous=int(_safe(row.get("contig", 0))),
                           common_lang=int(_safe(row.get("comlang_off", 0))),
                           fta=int(_safe(row.get("fta_wto", 0))))

    n_countries = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "country")
    n_products = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "product")
    n_sections = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "product_section")

    logger.info(
        f"Full hetero graph: {G.number_of_nodes()} nodes "
        f"({n_countries} countries, {n_products} products, {n_sections} sections), "
        f"{G.number_of_edges()} edges"
    )
    return G


def classical_link_baselines(
    G: nx.Graph,
    candidate_edges: list[tuple[str, str]],
) -> pd.DataFrame:
    """Compute Adamic-Adar, Jaccard, common-neighbour, and preferential attachment scores.

    Works on an undirected (bipartite) graph. For directed graphs,
    converts to undirected first.

    Returns a DataFrame with columns: node_u, node_v, adamic_adar, jaccard, cn, pref_attach.
    """
    if G.is_directed():
        G = G.to_undirected()

    # Filter to candidates where both nodes exist
    valid = [(u, v) for u, v in candidate_edges if G.has_node(u) and G.has_node(v)]
    if not valid:
        return pd.DataFrame(columns=["node_u", "node_v", "adamic_adar", "jaccard", "cn", "pref_attach"])

    # Adamic-Adar (yields (u, v, score) triples)
    aa_scores = {(u, v): s for u, v, s in nx.adamic_adar_index(G, valid)}

    # Jaccard
    jc_scores = {(u, v): s for u, v, s in nx.jaccard_coefficient(G, valid)}

    # Common neighbours
    cn_scores = {}
    for u, v in valid:
        cn_scores[(u, v)] = len(set(G.neighbors(u)) & set(G.neighbors(v)))

    # Preferential attachment
    pa_scores = {(u, v): s for u, v, s in nx.preferential_attachment(G, valid)}

    rows = []
    for u, v in valid:
        rows.append({
            "node_u": u,
            "node_v": v,
            "adamic_adar": aa_scores.get((u, v), 0.0),
            "jaccard": jc_scores.get((u, v), 0.0),
            "cn": cn_scores.get((u, v), 0),
            "pref_attach": pa_scores.get((u, v), 0),
        })

    result = pd.DataFrame(rows)
    logger.info(f"Classical baselines computed for {len(result)} candidate edges")
    return result
