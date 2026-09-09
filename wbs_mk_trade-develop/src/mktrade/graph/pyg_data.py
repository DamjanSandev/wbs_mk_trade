"""Convert the knowledge graph to PyTorch Geometric HeteroData objects.

Builds node feature tensors, edge-index tensors, and edge attributes
for the heterogeneous graph consumed by the GNN models.

Node types and features:
    'country': [eci, diversity, gdp_log, gdp_pc_log, pop_log, landlocked, eu_member,
                cefta_member, region_onehot(6)]  → 14-dim
    'product': [pci, ubiquity, section_onehot(~22)]  → 24-dim
    'product_section': [one-hot identity(~22)]  → 22-dim

Edge types:
    ('country', 'exports', 'product')        — from atlas/complexity
    ('country', 'trades_with', 'country')    — from bilateral, aggregated
    ('product', 'proximity', 'product')      — from proximity matrix
    ('country', 'neighbor_of', 'country')    — from gravity
    ('product', 'in_section', 'product_section') — from HS code prefix

Reverse edges are added automatically for message-passing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import HeteroData

from mktrade.config import DataConfig

# Broad HS sections for one-hot encoding (2-digit chapter ranges → section index)
# We use the first 2 digits of HS4 and group into ~22 standard sections.
_HS_CHAPTER_TO_SECTION: dict[int, int] = {}
_SECTION_RANGES = [
    (1, 5, 0), (6, 14, 1), (15, 15, 2), (16, 24, 3), (25, 27, 4),
    (28, 38, 5), (39, 40, 6), (41, 43, 7), (44, 46, 8), (47, 49, 9),
    (50, 63, 10), (64, 67, 11), (68, 70, 12), (71, 71, 13), (72, 83, 14),
    (84, 85, 15), (86, 89, 16), (90, 92, 17), (93, 93, 18), (94, 96, 19),
    (97, 97, 20),
]
NUM_HS_SECTIONS = 21

for lo, hi, sec_idx in _SECTION_RANGES:
    for ch in range(lo, hi + 1):
        _HS_CHAPTER_TO_SECTION[ch] = sec_idx


def _hs4_to_section_idx(hs4: str) -> int:
    """Map HS4 code string to section index (0-20)."""
    try:
        chapter = int(hs4[:2])
    except (ValueError, IndexError):
        return 0
    return _HS_CHAPTER_TO_SECTION.get(chapter, 0)


# Region labels for one-hot
_REGION_LABELS = ["Europe", "Asia", "Africa", "Americas", "Oceania", "Other"]
NUM_REGIONS = len(_REGION_LABELS)


def _assign_region(iso3: str) -> int:
    """Rough region assignment for countries. Returns index into _REGION_LABELS."""
    # European countries (common ISO3 codes)
    europe = {
        "ALB", "AND", "AUT", "BLR", "BEL", "BIH", "BGR", "HRV", "CYP", "CZE",
        "DNK", "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ITA",
        "XKX", "LVA", "LIE", "LTU", "LUX", "MLT", "MDA", "MNE", "NLD", "MKD",
        "NOR", "POL", "PRT", "ROU", "RUS", "SMR", "SRB", "SVK", "SVN", "ESP",
        "SWE", "CHE", "UKR", "GBR", "VAT",
    }
    asia = {
        "AFG", "ARM", "AZE", "BHR", "BGD", "BTN", "BRN", "KHM", "CHN", "GEO",
        "IND", "IDN", "IRN", "IRQ", "ISR", "JPN", "JOR", "KAZ", "KWT", "KGZ",
        "LAO", "LBN", "MYS", "MDV", "MNG", "MMR", "NPL", "OMN", "PAK", "PHL",
        "QAT", "SAU", "SGP", "KOR", "LKA", "SYR", "TWN", "TJK", "THA", "TLS",
        "TUR", "TKM", "ARE", "UZB", "VNM", "YEM", "PRK",
    }
    africa = {
        "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CMR", "CPV", "CAF", "TCD",
        "COM", "COD", "COG", "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH",
        "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG",
        "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
        "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO",
        "TUN", "UGA", "ZMB", "ZWE",
    }
    americas = {
        "ATG", "ARG", "BHS", "BRB", "BLZ", "BOL", "BRA", "CAN", "CHL", "COL",
        "CRI", "CUB", "DMA", "DOM", "ECU", "SLV", "GRD", "GTM", "GUY", "HTI",
        "HND", "JAM", "MEX", "NIC", "PAN", "PRY", "PER", "KNA", "LCA", "VCT",
        "SUR", "TTO", "USA", "URY", "VEN",
    }
    oceania = {"AUS", "FJI", "KIR", "MHL", "FSM", "NRU", "NZL", "PLW", "PNG", "WSM", "SLB", "TON", "TUV", "VUT"}

    if iso3 in europe:
        return 0
    elif iso3 in asia:
        return 1
    elif iso3 in africa:
        return 2
    elif iso3 in americas:
        return 3
    elif iso3 in oceania:
        return 4
    return 5


# EU members for binary feature
_EU_MEMBERS = {
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE",
}

# Landlocked countries (subset)
_LANDLOCKED = {
    "AFG", "AND", "ARM", "AUT", "AZE", "BLR", "BTN", "BOL", "BWA", "BFA",
    "BDI", "CAF", "TCD", "CZE", "ETH", "HUN", "KAZ", "KGZ", "LAO", "LSO",
    "LIE", "LUX", "MKD", "MWI", "MLI", "MDA", "MNG", "NPL", "NER", "PRY",
    "RWA", "SRB", "SVK", "SSD", "SWZ", "CHE", "TJK", "TKM", "UGA", "UZB",
    "ZMB", "ZWE", "XKX",
}


def build_hetero_data(
    exports_df: pd.DataFrame,
    country_features: pd.DataFrame,
    product_features: pd.DataFrame,
    bilateral_df: pd.DataFrame | None = None,
    gravity_df: pd.DataFrame | None = None,
    proximity_df: pd.DataFrame | None = None,
    year: int | None = None,
    cefta_members: set[str] | None = None,
    proximity_top_k: int | None = None,
) -> HeteroData:
    """Assemble a PyG HeteroData object from processed DataFrames.

    Parameters
    ----------
    exports_df : columns [year, iso3, hs4, export_value] + optional [export_rca, density].
    country_features : columns [iso3, eci, diversity] (from country_complexity).
    product_features : columns [hs4, pci, ubiquity] (from product_complexity).
    bilateral_df : columns [year, reporter_iso3, partner_iso3, hs4, flow, value].
    gravity_df : columns [year, iso3_o, iso3_d, dist, contig, comlang_off, ...].
    proximity_df : columns [year, hs4_1, hs4_2, proximity].
    year : snapshot year (filters all DataFrames).
    cefta_members : set of ISO3 codes for CEFTA membership feature.
    proximity_top_k : if set, keep only top-K nearest products per product
        by proximity score (limits memory for attention-heavy models like HGT).

    Returns
    -------
    HeteroData with node features, edge indices, and edge attributes.
    """
    if cefta_members is None:
        cefta_members = {"ALB", "BIH", "MKD", "MDA", "MNE", "SRB", "XKX"}

    # ── Filter to snapshot year ──
    exp = exports_df.copy()
    if year is not None:
        exp = exp[exp["year"] == year]
    exp = exp[exp["export_value"] > 0]

    # ── Build country and product ID mappings ──
    countries = sorted(exp["iso3"].unique())
    products = sorted(exp["hs4"].unique())

    country2idx = {c: i for i, c in enumerate(countries)}
    product2idx = {p: i for i, p in enumerate(products)}

    n_countries = len(countries)
    n_products = len(products)

    logger.info(f"PyG snapshot year={year}: {n_countries} countries, {n_products} products")

    # ── Country node features ──
    # [eci, diversity, gdp_log, gdp_pc_log, pop_log, landlocked, eu, cefta, region_onehot(6)]
    country_feat = torch.zeros(n_countries, 8 + NUM_REGIONS, dtype=torch.float32)

    cf = country_features.copy()
    if "year" in cf.columns and year is not None:
        cf = cf[cf["year"] == year]

    cf_lookup = cf.set_index("iso3") if not cf.empty else pd.DataFrame()

    for iso3, idx in country2idx.items():
        if iso3 in cf_lookup.index:
            row = cf_lookup.loc[iso3]
            eci_val = float(row.get("eci", 0)) if pd.notna(row.get("eci")) else 0.0
            div_val = float(row.get("diversity", 0)) if pd.notna(row.get("diversity")) else 0.0
            country_feat[idx, 0] = eci_val
            country_feat[idx, 1] = div_val

        # gdp, gdp_pc, population will be filled below from WDI if available
        # Binary features
        country_feat[idx, 5] = 1.0 if iso3 in _LANDLOCKED else 0.0
        country_feat[idx, 6] = 1.0 if iso3 in _EU_MEMBERS else 0.0
        country_feat[idx, 7] = 1.0 if iso3 in cefta_members else 0.0

        # Region one-hot
        region_idx = _assign_region(iso3)
        country_feat[idx, 8 + region_idx] = 1.0

    # ── Product node features ──
    # [pci, ubiquity, section_onehot(NUM_HS_SECTIONS)]
    product_feat = torch.zeros(n_products, 2 + NUM_HS_SECTIONS, dtype=torch.float32)

    pf = product_features.copy()
    if "year" in pf.columns and year is not None:
        pf = pf[pf["year"] == year]

    pf_lookup = pf.set_index("hs4") if not pf.empty else pd.DataFrame()

    for hs4, idx in product2idx.items():
        if hs4 in pf_lookup.index:
            row = pf_lookup.loc[hs4]
            pci_val = float(row.get("pci", 0)) if pd.notna(row.get("pci")) else 0.0
            ubi_val = float(row.get("ubiquity", 0)) if pd.notna(row.get("ubiquity")) else 0.0
            product_feat[idx, 0] = pci_val
            product_feat[idx, 1] = ubi_val

        sec_idx = _hs4_to_section_idx(hs4)
        product_feat[idx, 2 + sec_idx] = 1.0

    # ── Product section nodes ──
    sections_in_data = sorted(set(_hs4_to_section_idx(p) for p in products))
    section2idx = {s: i for i, s in enumerate(sections_in_data)}
    n_sections = len(sections_in_data)

    # Section features: one-hot identity
    section_feat = torch.zeros(n_sections, NUM_HS_SECTIONS, dtype=torch.float32)
    for sec, idx in section2idx.items():
        section_feat[idx, sec] = 1.0

    # ── Build HeteroData ──
    data = HeteroData()

    data["country"].x = country_feat
    data["country"].iso3 = countries
    data["country"].num_nodes = n_countries

    data["product"].x = product_feat
    data["product"].hs4 = products
    data["product"].num_nodes = n_products

    data["product_section"].x = section_feat
    data["product_section"].num_nodes = n_sections

    # ── EXPORTS edges (country → product) — vectorized ──
    exp_mapped = exp.copy()
    exp_mapped["c_idx"] = exp_mapped["iso3"].map(country2idx)
    exp_mapped["p_idx"] = exp_mapped["hs4"].map(product2idx)
    exp_mapped = exp_mapped.dropna(subset=["c_idx", "p_idx"])

    if not exp_mapped.empty:
        src = torch.tensor(exp_mapped["c_idx"].astype(int).values, dtype=torch.long)
        dst = torch.tensor(exp_mapped["p_idx"].astype(int).values, dtype=torch.long)
        edge_index = torch.stack([src, dst])
        data["country", "exports", "product"].edge_index = edge_index

        val = torch.tensor(exp_mapped["export_value"].values, dtype=torch.float32)
        rca_col = exp_mapped["export_rca"].fillna(0).values if "export_rca" in exp_mapped.columns else np.zeros(len(exp_mapped))
        rca = torch.tensor(rca_col, dtype=torch.float32)
        data["country", "exports", "product"].edge_attr = torch.stack([val, rca], dim=1)

        # Add reverse edges for message passing
        data["product", "rev_exports", "country"].edge_index = torch.stack([dst, src])
        data["product", "rev_exports", "country"].edge_attr = data["country", "exports", "product"].edge_attr.clone()

    # ── IN_SECTION edges (product → product_section) ──
    sec_src, sec_dst = [], []
    for hs4, p_idx in product2idx.items():
        sec = _hs4_to_section_idx(hs4)
        if sec in section2idx:
            sec_src.append(p_idx)
            sec_dst.append(section2idx[sec])

    if sec_src:
        edge_index = torch.tensor([sec_src, sec_dst], dtype=torch.long)
        data["product", "in_section", "product_section"].edge_index = edge_index
        data["product_section", "rev_in_section", "product"].edge_index = torch.stack(
            [edge_index[1], edge_index[0]]
        )

    # ── PROXIMITY edges (product ↔ product) — vectorized ──
    if proximity_df is not None and not proximity_df.empty:
        prox = proximity_df.copy()
        if "year" in prox.columns:
            if year is not None and year in prox["year"].values:
                prox = prox[prox["year"] == year]
            else:
                eligible = prox if year is None else prox[prox["year"] <= year]
                if eligible.empty:
                    logger.warning(
                        f"  Proximity: no leakage-safe matrix is available at or before {year}; "
                        "skipping proximity edges"
                    )
                    prox = eligible
                else:
                    latest = eligible["year"].max()
                    prox = eligible[eligible["year"] == latest]
                    if year is not None:
                        logger.info(f"  Proximity: year {year} unavailable, using prior year {latest}")
        prox = prox[(prox["proximity"] > 0) & (prox["hs4_1"] != prox["hs4_2"])]

        prox["i"] = prox["hs4_1"].map(product2idx)
        prox["j"] = prox["hs4_2"].map(product2idx)
        prox = prox.dropna(subset=["i", "j"])

        # Optionally sparsify: keep top-K nearest neighbors per product
        # (full matrix can be 1.45M+ edges, causing OOM for attention-based models)
        if proximity_top_k is not None and not prox.empty and len(prox) > proximity_top_k * n_products:
            before = len(prox)
            prox = (
                prox.sort_values("proximity", ascending=False)
                .groupby("i", sort=False)
                .head(proximity_top_k)
            )
            logger.info(f"  Proximity sparsified: {before} -> {len(prox)} "
                        f"(top-{proximity_top_k} per product)")

        if not prox.empty:
            p_src = torch.tensor(prox["i"].astype(int).values, dtype=torch.long)
            p_dst = torch.tensor(prox["j"].astype(int).values, dtype=torch.long)
            p_w = torch.tensor(prox["proximity"].values, dtype=torch.float32)
            data["product", "proximity", "product"].edge_index = torch.stack([p_src, p_dst])
            data["product", "proximity", "product"].edge_attr = p_w.unsqueeze(1)
            logger.info(f"  PROXIMITY edges: {len(prox)}")

    # ── TRADES_WITH edges (country → country) — vectorized ──
    if bilateral_df is not None and not bilateral_df.empty:
        bil = bilateral_df.copy()
        if year is not None:
            bil = bil[bil["year"] == year]

        if not bil.empty:
            agg = bil.groupby(["reporter_iso3", "partner_iso3", "flow"])["value"].sum().reset_index()
            exports_agg = agg[agg["flow"] == "X"].rename(columns={"value": "export_value"})
            imports_agg = agg[agg["flow"] == "M"].rename(columns={"value": "import_value"})

            merged = exports_agg.merge(
                imports_agg[["reporter_iso3", "partner_iso3", "import_value"]],
                on=["reporter_iso3", "partner_iso3"], how="outer",
            ).fillna(0)

            merged["r_idx"] = merged["reporter_iso3"].map(country2idx)
            merged["p_idx"] = merged["partner_iso3"].map(country2idx)
            merged = merged.dropna(subset=["r_idx", "p_idx"])
            merged = merged[merged["r_idx"] != merged["p_idx"]]

            if not merged.empty:
                tw_src = torch.tensor(merged["r_idx"].astype(int).values, dtype=torch.long)
                tw_dst = torch.tensor(merged["p_idx"].astype(int).values, dtype=torch.long)
                tw_ev = torch.tensor(merged["export_value"].values, dtype=torch.float32)
                tw_iv = torch.tensor(merged["import_value"].values, dtype=torch.float32)
                data["country", "trades_with", "country"].edge_index = torch.stack([tw_src, tw_dst])
                data["country", "trades_with", "country"].edge_attr = torch.stack([tw_ev, tw_iv], dim=1)
                logger.info(f"  TRADES_WITH edges: {len(merged)}")

    # ── NEIGHBOR_OF edges (country → country, from gravity) — vectorized ──
    if gravity_df is not None and not gravity_df.empty:
        grav = gravity_df.copy()
        if "year" in grav.columns:
            if year is not None and year in grav["year"].values:
                grav = grav[grav["year"] == year]
            else:
                # Gravity data may lag — use latest available year
                latest = grav["year"].max()
                grav = grav[grav["year"] == latest]
                if year is not None:
                    logger.info(f"  Gravity: year {year} unavailable, using {latest}")

        grav["o_idx"] = grav["iso3_o"].map(country2idx)
        grav["d_idx"] = grav["iso3_d"].map(country2idx)
        grav = grav.dropna(subset=["o_idx", "d_idx"])
        grav = grav[grav["o_idx"] != grav["d_idx"]]

        if not grav.empty:
            gn_src = torch.tensor(grav["o_idx"].astype(int).values, dtype=torch.long)
            gn_dst = torch.tensor(grav["d_idx"].astype(int).values, dtype=torch.long)
            gn_dist = torch.tensor(grav["dist"].fillna(0).values, dtype=torch.float32)
            gn_contig = torch.tensor(grav["contig"].fillna(0).values, dtype=torch.float32)
            gn_lang = torch.tensor(grav["comlang_off"].fillna(0).values, dtype=torch.float32)
            gn_fta = torch.tensor(grav["fta_wto"].fillna(0).values, dtype=torch.float32)
            data["country", "neighbor_of", "country"].edge_index = torch.stack([gn_src, gn_dst])
            data["country", "neighbor_of", "country"].edge_attr = torch.stack(
                [gn_dist, gn_contig, gn_lang, gn_fta], dim=1
            )
            logger.info(f"  NEIGHBOR_OF edges: {len(grav)}")

    _log_hetero_summary(data)
    return data


def enrich_country_features(
    data: HeteroData,
    wdi_df: pd.DataFrame,
    countries: list[str],
    year: int | None = None,
) -> HeteroData:
    """Add WDI economic indicators (GDP, GDP/cap, population) to country node features.

    Fills columns 2-4 of country.x with log-transformed values.
    """
    wdi = wdi_df.copy()
    if year is not None and "year" in wdi.columns:
        wdi = wdi[wdi["year"] == year]
    elif "year" in wdi.columns:
        wdi = wdi[wdi["year"] == wdi["year"].max()]

    wdi_lookup = wdi.set_index("iso3") if not wdi.empty else pd.DataFrame()

    for i, iso3 in enumerate(countries):
        if iso3 in wdi_lookup.index:
            row = wdi_lookup.loc[iso3]
            gdp = float(row.get("gdp", 0)) if pd.notna(row.get("gdp")) else 0.0
            gdp_pc = float(row.get("gdp_pc", 0)) if pd.notna(row.get("gdp_pc")) else 0.0
            pop = float(row.get("population", 0)) if pd.notna(row.get("population")) else 0.0

            data["country"].x[i, 2] = np.log1p(gdp) if gdp > 0 else 0.0
            data["country"].x[i, 3] = np.log1p(gdp_pc) if gdp_pc > 0 else 0.0
            data["country"].x[i, 4] = np.log1p(pop) if pop > 0 else 0.0

    logger.info(f"Enriched country features with WDI data ({wdi_lookup.shape[0]} entries)")
    return data


def add_structural_embeddings(
    data: HeteroData,
    country_emb: torch.Tensor | None = None,
    product_emb: torch.Tensor | None = None,
) -> HeteroData:
    """Concatenate structural embeddings (e.g. FastRP, Node2Vec) to node feature matrices."""
    if country_emb is not None:
        assert country_emb.shape[0] == data["country"].num_nodes
        data["country"].x = torch.cat([data["country"].x, country_emb], dim=1)
        logger.info(f"Added {country_emb.shape[1]}-dim structural embeddings to country nodes")

    if product_emb is not None:
        assert product_emb.shape[0] == data["product"].num_nodes
        data["product"].x = torch.cat([data["product"].x, product_emb], dim=1)
        logger.info(f"Added {product_emb.shape[1]}-dim structural embeddings to product nodes")

    return data


def save_hetero_data(data: HeteroData, path: Path) -> None:
    """Save HeteroData to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    logger.info(f"Saved HeteroData to {path}")


def load_hetero_data(path: Path) -> HeteroData:
    """Load HeteroData from disk."""
    data = torch.load(path, weights_only=False)
    logger.info(f"Loaded HeteroData from {path}")
    _log_hetero_summary(data)
    return data


def _log_hetero_summary(data: HeteroData) -> None:
    """Log a summary of the HeteroData object."""
    logger.info(f"HeteroData summary:")
    for node_type in data.node_types:
        store = data[node_type]
        n = store.num_nodes
        feat_dim = store.x.shape[1] if hasattr(store, "x") and store.x is not None else 0
        logger.info(f"  {node_type}: {n} nodes, {feat_dim}-dim features")
    for edge_type in data.edge_types:
        store = data[edge_type]
        n = store.edge_index.shape[1] if hasattr(store, "edge_index") else 0
        logger.info(f"  {edge_type}: {n} edges")
