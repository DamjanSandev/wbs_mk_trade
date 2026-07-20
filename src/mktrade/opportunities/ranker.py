"""Rank predicted export opportunities for North Macedonia.

Task A: top-K new products MKD could export (country-product links).
Task B: top-K new markets for MKD's existing products (country-product-country links).
"""

from __future__ import annotations

import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import HeteroData

from mktrade.models.link_predictor import LinkPredictor


def rank_product_opportunities(
    model: LinkPredictor,
    data: HeteroData,
    country: str = "MKD",
    top_k: int = 50,
    complexity_df: pd.DataFrame | None = None,
    msg_data: HeteroData | None = None,
) -> pd.DataFrame:
    """Score and rank all non-exported products for a country (Task A).

    Parameters
    ----------
    model : Trained LinkPredictor.
    data : HeteroData with all graph structure (latest-year graph for node IDs).
    country : ISO3 code of target country.
    top_k : Number of top opportunities to return.
    complexity_df : Optional DataFrame with columns [iso3, hs4, density, pci]
        for enriching results.
    msg_data : Optional HeteroData used for message passing (encoding).
        If None, uses ``data``. Use this when the model was trained on a
        different graph topology (e.g. temporal split training graph).

    Returns
    -------
    DataFrame with columns: hs4, score, rank, density, pci, section.
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    encode_data = (msg_data.to(device) if msg_data is not None else data)

    countries = data["country"].iso3
    products = data["product"].hs4

    if country not in countries:
        raise ValueError(f"Country {country} not found in graph ({len(countries)} countries)")

    country_idx = countries.index(country)

    # Find products already exported by this country
    ei = data["country", "exports", "product"].edge_index
    exported_mask = ei[0] == country_idx
    exported_products = set(ei[1, exported_mask].cpu().tolist())

    # Candidate products: all products NOT currently exported
    all_product_indices = list(range(len(products)))
    candidate_indices = [i for i in all_product_indices if i not in exported_products]

    if not candidate_indices:
        logger.warning(f"No candidate products for {country} (exports all {len(products)} products)")
        return pd.DataFrame(columns=["hs4", "score", "rank", "density", "pci", "section"])

    logger.info(f"Scoring {len(candidate_indices)} candidate products for {country} "
                f"(already exports {len(exported_products)})")

    # Score all candidates at once
    src = torch.full((len(candidate_indices),), country_idx, dtype=torch.long, device=device)
    dst = torch.tensor(candidate_indices, dtype=torch.long, device=device)
    edge_label_index = torch.stack([src, dst])

    with torch.no_grad():
        z_dict = model.encode(encode_data)
        logits = model.decode(z_dict, edge_label_index, "country", "product")
        scores = torch.sigmoid(logits).cpu().numpy()

    # Build result DataFrame
    rows = []
    for i, prod_idx in enumerate(candidate_indices):
        hs4 = products[prod_idx]
        rows.append({"hs4": hs4, "score": float(scores[i])})

    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Enrich with complexity data
    if complexity_df is not None:
        cx = complexity_df[complexity_df["iso3"] == country].copy()
        if not cx.empty:
            cx_lookup = cx.set_index("hs4")
            df["density"] = df["hs4"].map(cx_lookup.get("density", pd.Series(dtype=float)))
            df["pci"] = df["hs4"].map(cx_lookup.get("pci", pd.Series(dtype=float)))
        else:
            df["density"] = None
            df["pci"] = None
    else:
        df["density"] = None
        df["pci"] = None

    # Add HS section
    df["section"] = df["hs4"].apply(lambda x: x[:2])

    result = df.head(top_k)
    logger.info(f"Top {len(result)} product opportunities for {country}: "
                f"score range [{result['score'].min():.4f}, {result['score'].max():.4f}]")

    return result


def rank_market_opportunities(
    model: LinkPredictor,
    data: HeteroData,
    country: str = "MKD",
    top_k: int = 50,
    bilateral_df: pd.DataFrame | None = None,
    msg_data: HeteroData | None = None,
) -> pd.DataFrame:
    """Score and rank new (product, destination) pairs for a country (Task B).

    For each product MKD already exports, find countries it does NOT yet
    export that product to and score each (product, destination) pair.

    Parameters
    ----------
    model : Trained LinkPredictor.
    data : HeteroData with all graph structure.
    country : ISO3 code of target country.
    top_k : Number of top opportunities to return.
    bilateral_df : Optional DataFrame for enriching with current trade volumes.
    msg_data : Optional HeteroData for message passing (encoding).

    Returns
    -------
    DataFrame with columns: hs4, partner_iso3, score, rank, section.
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    encode_data = (msg_data.to(device) if msg_data is not None else data)

    countries = data["country"].iso3
    products = data["product"].hs4

    if country not in countries:
        raise ValueError(f"Country {country} not found in graph")

    country_idx = countries.index(country)

    # Find products MKD already exports
    ei = data["country", "exports", "product"].edge_index
    exported_mask = ei[0] == country_idx
    mkd_products = set(ei[1, exported_mask].cpu().tolist())

    # Build the set of all existing (country, product) export edges
    existing_edges = set()
    for i in range(ei.shape[1]):
        existing_edges.add((ei[0, i].item(), ei[1, i].item()))

    # For each MKD product, find countries that don't export it
    candidates = []
    for prod_idx in mkd_products:
        for c_idx in range(len(countries)):
            if c_idx != country_idx and (c_idx, prod_idx) not in existing_edges:
                candidates.append((c_idx, prod_idx))

    if not candidates:
        logger.warning(f"No market expansion candidates for {country}")
        return pd.DataFrame(columns=["hs4", "partner_iso3", "score", "rank", "section"])

    logger.info(f"Scoring {len(candidates)} market expansion candidates for {country}")

    # Score in batches to avoid OOM
    batch_size = 50000
    all_scores = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        src = torch.tensor([c for c, _ in batch], dtype=torch.long, device=device)
        dst = torch.tensor([p for _, p in batch], dtype=torch.long, device=device)
        edge_label_index = torch.stack([src, dst])

        with torch.no_grad():
            z_dict = model.encode(encode_data)
            logits = model.decode(z_dict, edge_label_index, "country", "product")
            batch_scores = torch.sigmoid(logits).cpu().numpy()
        all_scores.extend(batch_scores.tolist())

    # Build result
    rows = []
    for i, (c_idx, p_idx) in enumerate(candidates):
        rows.append({
            "hs4": products[p_idx],
            "partner_iso3": countries[c_idx],
            "score": all_scores[i],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["section"] = df["hs4"].apply(lambda x: x[:2])

    result = df.head(top_k)
    logger.info(f"Top {len(result)} market expansion opportunities for {country}: "
                f"score range [{result['score'].min():.4f}, {result['score'].max():.4f}]")

    return result


def ensemble_rankings(
    gnn_df: pd.DataFrame,
    density_df: pd.DataFrame | None = None,
    classical_df: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Combine GNN scores with density and classical baselines via weighted average.

    Parameters
    ----------
    gnn_df : GNN-scored opportunities (must have 'hs4' and 'score' columns).
    density_df : Optional complexity density scores.
    classical_df : Optional classical baselines (adamic_adar, jaccard, etc.).
    weights : Dict of {method: weight}. Default: gnn=0.6, density=0.3, classical=0.1.

    Returns
    -------
    DataFrame with combined 'ensemble_score' and 'ensemble_rank'.
    """
    if weights is None:
        weights = {"gnn": 0.6, "density": 0.3, "classical": 0.1}

    result = gnn_df[["hs4", "score"]].copy()
    result = result.rename(columns={"score": "gnn_score"})

    # Normalize GNN scores to [0, 1]
    _min, _max = result["gnn_score"].min(), result["gnn_score"].max()
    if _max > _min:
        result["gnn_norm"] = (result["gnn_score"] - _min) / (_max - _min)
    else:
        result["gnn_norm"] = 0.5

    result["ensemble_score"] = result["gnn_norm"] * weights.get("gnn", 0.6)

    if density_df is not None and "density" in density_df.columns:
        merged = result.merge(density_df[["hs4", "density"]], on="hs4", how="left")
        merged["density"] = merged["density"].fillna(0)
        d_min, d_max = merged["density"].min(), merged["density"].max()
        if d_max > d_min:
            merged["density_norm"] = (merged["density"] - d_min) / (d_max - d_min)
        else:
            merged["density_norm"] = 0.5
        merged["ensemble_score"] += merged["density_norm"] * weights.get("density", 0.3)
        result = merged

    if classical_df is not None and "adamic_adar" in classical_df.columns:
        # Classical baselines use 'node_v' = 'P_{hs4}' format
        cl = classical_df.copy()
        cl["hs4"] = cl["node_v"].str.replace("P_", "", n=1)
        cl = cl[["hs4", "adamic_adar"]].copy()
        merged = result.merge(cl, on="hs4", how="left")
        merged["adamic_adar"] = merged["adamic_adar"].fillna(0)
        aa_min, aa_max = merged["adamic_adar"].min(), merged["adamic_adar"].max()
        if aa_max > aa_min:
            merged["aa_norm"] = (merged["adamic_adar"] - aa_min) / (aa_max - aa_min)
        else:
            merged["aa_norm"] = 0.5
        merged["ensemble_score"] += merged["aa_norm"] * weights.get("classical", 0.1)
        result = merged

    result = result.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    result["ensemble_rank"] = range(1, len(result) + 1)

    return result
