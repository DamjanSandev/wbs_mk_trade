"""Candidate generation and ranking for product and market opportunities.

Task A ranks every unseen country-product link. Task B ranks a genuine
``(origin, product, destination)`` relationship using destination demand,
origin supply, bilateral history, gravity, and macro-economic features.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from loguru import logger

from mktrade.opportunities.features import enrich_market_candidates
from mktrade.opportunities.reranker import OpportunityReranker

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from mktrade.models.link_predictor import LinkPredictor


def _candidate_complexity(
    complexity_df: pd.DataFrame | None,
    country: str,
    as_of_year: int | None,
) -> pd.DataFrame:
    if complexity_df is None or complexity_df.empty:
        return pd.DataFrame(columns=["hs4"])
    frame = complexity_df[complexity_df["iso3"].astype(str) == str(country)].copy()
    if as_of_year is not None and "year" in frame.columns:
        frame = frame[frame["year"] <= as_of_year]
        if not frame.empty:
            frame = frame[frame["year"] == frame["year"].max()]
    columns = [
        column
        for column in ("hs4", "density", "pci", "complexity_gain", "cog")
        if column in frame.columns
    ]
    if "hs4" not in columns:
        return pd.DataFrame(columns=["hs4"])
    frame["hs4"] = frame["hs4"].astype(str).str.zfill(4)
    return frame[columns].drop_duplicates("hs4", keep="last")


def rank_product_opportunities(
    model: LinkPredictor,
    data: HeteroData,
    country: str = "MKD",
    top_k: int | None = 50,
    complexity_df: pd.DataFrame | None = None,
    msg_data: HeteroData | None = None,
    as_of_year: int | None = None,
) -> pd.DataFrame:
    """Score all products without a sustained export link for a country.

    Set ``top_k=None`` during ensemble construction. This ensures density and
    other features can rescue candidates outside the GNN's initial top-K.
    """

    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    encode_data = msg_data.to(device) if msg_data is not None else data
    countries = data["country"].iso3
    products = data["product"].hs4

    if country not in countries:
        raise ValueError(f"Country {country} not found in graph ({len(countries)} countries)")
    country_idx = countries.index(country)
    edge_index = data["country", "exports", "product"].edge_index
    exported = set(edge_index[1, edge_index[0] == country_idx].cpu().tolist())
    candidate_indices = [index for index in range(len(products)) if index not in exported]
    if not candidate_indices:
        return pd.DataFrame(
            columns=["hs4", "score", "gnn_score", "rank", "density", "pci", "section"]
        )

    logger.info(
        f"Scoring all {len(candidate_indices)} product candidates for {country} "
        f"({len(exported)} sustained export links excluded)"
    )
    source = torch.full((len(candidate_indices),), country_idx, dtype=torch.long, device=device)
    destination = torch.tensor(candidate_indices, dtype=torch.long, device=device)
    with torch.no_grad():
        embeddings = model.encode(encode_data)
        logits = model.decode(
            embeddings, torch.stack([source, destination]), "country", "product"
        )
        raw_scores = logits.cpu().numpy()
        probabilities = torch.sigmoid(logits).cpu().numpy()

    result = pd.DataFrame(
        {
            "hs4": [str(products[index]).zfill(4) for index in candidate_indices],
            "gnn_logit": raw_scores.astype(float),
            "gnn_score": probabilities.astype(float),
        }
    )
    result["score"] = result["gnn_score"]
    complexity = _candidate_complexity(complexity_df, country, as_of_year)
    result = result.merge(complexity, on="hs4", how="left")
    for column in ("density", "pci", "complexity_gain"):
        if column not in result.columns:
            result[column] = np.nan
    result["section"] = result["hs4"].str[:2]
    result = result.sort_values("gnn_logit", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result if top_k is None else result.head(top_k).reset_index(drop=True)


def _existing_market_routes(
    bilateral_df: pd.DataFrame | None,
    country: str,
    as_of_year: int | None,
    minimum_value: float,
) -> set[tuple[str, str]]:
    if bilateral_df is None or bilateral_df.empty:
        return set()
    required = {"reporter_iso3", "partner_iso3", "hs4", "flow", "value", "year"}
    if not required.issubset(bilateral_df.columns):
        logger.warning("Bilateral data lacks route-level columns; no existing routes can be filtered")
        return set()
    frame = bilateral_df[
        (bilateral_df["reporter_iso3"].astype(str) == str(country))
        & (bilateral_df["flow"].astype(str).str.upper() == "X")
    ].copy()
    if as_of_year is not None:
        frame = frame[frame["year"] <= as_of_year]
    if frame.empty:
        return set()
    latest_year = int(frame["year"].max())
    frame = frame[frame["year"] == latest_year]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").fillna(0.0)
    active = frame.groupby(["hs4", "partner_iso3"], as_index=False)["value"].sum()
    active = active[active["value"] >= minimum_value]
    return {
        (str(row.hs4).zfill(4), str(row.partner_iso3))
        for row in active.itertuples(index=False)
    }


def rank_market_opportunities(
    model: LinkPredictor,
    data: HeteroData,
    country: str = "MKD",
    top_k: int | None = 50,
    bilateral_df: pd.DataFrame | None = None,
    msg_data: HeteroData | None = None,
    *,
    gravity_df: pd.DataFrame | None = None,
    tariff_df: pd.DataFrame | None = None,
    wdi_df: pd.DataFrame | None = None,
    exports_df: pd.DataFrame | None = None,
    as_of_year: int | None = None,
    reranker: OpportunityReranker | None = None,
    minimum_existing_route_value: float = 100_000.0,
) -> pd.DataFrame:
    """Rank origin-product-destination market-expansion candidates.

    The destination GNN score is only one feature. Bilateral imports and growth,
    current and historical MKD trade, supplier concentration, distance, FTA,
    language, GDP, population, and MKD supply are attached before reranking.
    """

    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    encode_data = msg_data.to(device) if msg_data is not None else data
    countries = data["country"].iso3
    products = data["product"].hs4
    if country not in countries:
        raise ValueError(f"Country {country} not found in graph")

    country_idx = countries.index(country)
    exports_edge = data["country", "exports", "product"].edge_index
    product_indices = sorted(
        set(exports_edge[1, exports_edge[0] == country_idx].cpu().tolist())
    )
    existing_routes = _existing_market_routes(
        bilateral_df, country, as_of_year, minimum_existing_route_value
    )
    candidates = [
        (destination_idx, product_idx)
        for product_idx in product_indices
        for destination_idx in range(len(countries))
        if destination_idx != country_idx
        and (str(products[product_idx]).zfill(4), str(countries[destination_idx]))
        not in existing_routes
    ]
    if not candidates:
        return pd.DataFrame(columns=["hs4", "partner_iso3", "score", "rank", "section"])

    logger.info(
        f"Scoring all {len(candidates)} ({country}, product, destination) candidates; "
        f"filtered {len(existing_routes)} active bilateral routes"
    )
    with torch.no_grad():
        embeddings = model.encode(encode_data)
        destination_logits: list[float] = []
        destination_scores: list[float] = []
        batch_size = 50_000
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            source = torch.tensor([item[0] for item in batch], dtype=torch.long, device=device)
            target = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
            logits = model.decode(
                embeddings, torch.stack([source, target]), "country", "product"
            )
            destination_logits.extend(logits.cpu().tolist())
            destination_scores.extend(torch.sigmoid(logits).cpu().tolist())

        product_tensor = torch.tensor(product_indices, dtype=torch.long, device=device)
        origin_tensor = torch.full_like(product_tensor, country_idx)
        supply_logits = model.decode(
            embeddings, torch.stack([origin_tensor, product_tensor]), "country", "product"
        )
        supply_logit_lookup = {
            product_idx: float(logit)
            for product_idx, logit in zip(
                product_indices,
                supply_logits.cpu().tolist(),
                strict=True,
            )
        }
        supply_lookup = {
            product_idx: float(score)
            for product_idx, score in zip(
                product_indices, torch.sigmoid(supply_logits).cpu().tolist(), strict=True
            )
        }

    result = pd.DataFrame(
        [
            {
                "origin_iso3": country,
                "hs4": str(products[product_idx]).zfill(4),
                "partner_iso3": str(countries[destination_idx]),
                "destination_gnn_logit": float(destination_logits[position]),
                "destination_gnn_score": float(destination_scores[position]),
                "supply_gnn_logit": supply_logit_lookup[product_idx],
                "supply_gnn_score": supply_lookup[product_idx],
            }
            for position, (destination_idx, product_idx) in enumerate(candidates)
        ]
    )
    result = enrich_market_candidates(
        result,
        country=country,
        bilateral_df=bilateral_df,
        gravity_df=gravity_df,
        tariff_df=tariff_df,
        wdi_df=wdi_df,
        exports_df=exports_df,
        as_of_year=as_of_year,
    )
    result["section"] = result["hs4"].str[:2]
    if reranker is not None:
        return reranker.rerank(result, top_k=top_k)

    result["score"] = result["destination_gnn_score"]
    result = result.sort_values("destination_gnn_logit", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result if top_k is None else result.head(top_k).reset_index(drop=True)


def assemble_candidate_signals(
    gnn_df: pd.DataFrame,
    density_df: pd.DataFrame | None,
    classical_df: pd.DataFrame | None,
) -> pd.DataFrame:
    result = gnn_df.copy()
    if "gnn_score" not in result.columns:
        result["gnn_score"] = result["score"]
    if density_df is not None and not density_df.empty and "density" in density_df.columns:
        density = density_df.copy()
        density["hs4"] = density["hs4"].astype(str).str.zfill(4)
        density = density.drop_duplicates("hs4", keep="last")
        additions = [column for column in ("hs4", "density", "pci", "complexity_gain") if column in density.columns]
        result = result.merge(density[additions], on="hs4", how="left", suffixes=("", "_extra"))
        for column in additions[1:]:
            extra = f"{column}_extra"
            if extra in result.columns:
                result[column] = result[column].combine_first(result[extra])
                result = result.drop(columns=extra)

    if classical_df is not None and not classical_df.empty:
        classical = classical_df.copy()
        if "hs4" not in classical.columns and "node_v" in classical.columns:
            classical["hs4"] = classical["node_v"].astype(str).str.replace(
                r"^P_", "", regex=True
            )
        if "hs4" in classical.columns:
            classical["hs4"] = classical["hs4"].astype(str).str.zfill(4)
            signal_columns = [
                column
                for column in ("adamic_adar", "jaccard", "cn", "pref_attach")
                if column in classical.columns
            ]
            result = result.merge(
                classical[["hs4", *signal_columns]].drop_duplicates("hs4", keep="last"),
                on="hs4",
                how="left",
            )
    return result


def ensemble_rankings(
    gnn_df: pd.DataFrame,
    density_df: pd.DataFrame | None = None,
    classical_df: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    *,
    reranker: OpportunityReranker | None = None,
    validation_df: pd.DataFrame | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Rerank the full candidate universe using validation outcomes.

    ``weights`` remains only to give callers an actionable migration error; the
    fixed 60/30/10 blend has deliberately been removed. If no fitted reranker or
    validation labels are supplied, a non-probabilistic equal-percentile fallback
    is used and constant signals such as all-zero Adamic-Adar are ignored.
    """

    if weights is not None:
        raise ValueError(
            "Fixed ensemble weights were removed. Pass a fitted OpportunityReranker "
            "or validation_df with historical labels."
        )
    result = assemble_candidate_signals(gnn_df, density_df, classical_df)
    if reranker is None and validation_df is not None:
        reranker = OpportunityReranker().fit(validation_df)
    if reranker is not None:
        return reranker.rerank(result, top_k=top_k)

    warnings.warn(
        "No validation labels were supplied; using equal percentile ranks, not calibrated probabilities.",
        RuntimeWarning,
        stacklevel=2,
    )
    preferred = [
        "gnn_logit", "gnn_score", "density", "complexity_gain", "global_demand_growth",
        "historical_export_persistence", "adamic_adar", "jaccard", "cn",
    ]
    usable = [
        column
        for column in preferred
        if column in result.columns
        and pd.to_numeric(result[column], errors="coerce").nunique(dropna=True) > 1
    ]
    if not usable:
        usable = ["gnn_score"]
    percentile_scores = []
    for column in usable:
        values = pd.to_numeric(result[column], errors="coerce")
        values = values.fillna(values.median() if values.notna().any() else 0.0)
        percentile_scores.append(values.rank(pct=True).to_numpy())
    result["ensemble_score"] = np.mean(percentile_scores, axis=0)
    result["score"] = result["ensemble_score"]
    result = result.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    result["ensemble_rank"] = np.arange(1, len(result) + 1)
    result["rank"] = result["ensemble_rank"]
    result["ensemble_method"] = "equal_percentile_fallback"
    return result if top_k is None else result.head(top_k).reset_index(drop=True)
