"""Model explanations for predicted opportunities.

Uses feature-importance approximation and attention-weight extraction
to explain why a particular link was predicted.

Note: PyG's GNNExplainer has limited support for heterogeneous graphs,
so we use a gradient-based feature importance method that works universally.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch_geometric.data import HeteroData

from mktrade.models.link_predictor import LinkPredictor
from mktrade.models.encoders import GATEncoder, HGTEncoder


# Country feature names (must match pyg_data.py ordering)
COUNTRY_FEATURE_NAMES = [
    "eci", "diversity", "gdp_log", "gdp_pc_log", "pop_log",
    "landlocked", "eu_member", "cefta_member",
    "region_europe", "region_asia", "region_africa",
    "region_americas", "region_oceania", "region_other",
]

# Product feature names (first 2 scalar + 21 section one-hot)
PRODUCT_FEATURE_NAMES = ["pci", "ubiquity"] + [f"section_{i}" for i in range(21)]


def explain_link(
    model: LinkPredictor,
    data: HeteroData,
    src_idx: int,
    dst_idx: int,
    src_type: str = "country",
    dst_type: str = "product",
) -> dict[str, Any]:
    """Explain a single predicted link using gradient-based feature importance.

    Computes the gradient of the link score with respect to the source and
    destination node features, giving a per-feature importance measure.

    Returns a dict with:
        - score: predicted link probability.
        - src_importance: dict of {feature_name: importance_score} for source node.
        - dst_importance: dict of {feature_name: importance_score} for destination node.
        - src_features: raw feature values for source node.
        - dst_features: raw feature values for destination node.
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    # Enable gradient computation on node features
    src_x = data[src_type].x.clone().detach().requires_grad_(True)
    dst_x = data[dst_type].x.clone().detach().requires_grad_(True)

    # Replace features in data
    original_src_x = data[src_type].x
    original_dst_x = data[dst_type].x
    data[src_type].x = src_x
    data[dst_type].x = dst_x

    # Forward pass for the specific edge
    edge_label_index = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long, device=device)

    try:
        z_dict = model.encode(data)
        logit = model.decode(z_dict, edge_label_index, src_type, dst_type)
        score = torch.sigmoid(logit).item()

        # Backward to get gradients
        logit.backward()

        # Feature importance = |feature_value * gradient|
        src_grad = src_x.grad[src_idx].cpu().detach()
        dst_grad = dst_x.grad[dst_idx].cpu().detach()

        src_vals = src_x[src_idx].cpu().detach()
        dst_vals = dst_x[dst_idx].cpu().detach()

        src_importance_raw = (src_vals * src_grad).abs()
        dst_importance_raw = (dst_vals * dst_grad).abs()

        # Normalize importance
        src_total = src_importance_raw.sum()
        dst_total = dst_importance_raw.sum()

        src_imp_norm = (src_importance_raw / src_total).numpy() if src_total > 0 else src_importance_raw.numpy()
        dst_imp_norm = (dst_importance_raw / dst_total).numpy() if dst_total > 0 else dst_importance_raw.numpy()

        # Map to feature names
        src_names = COUNTRY_FEATURE_NAMES if src_type == "country" else PRODUCT_FEATURE_NAMES
        dst_names = PRODUCT_FEATURE_NAMES if dst_type == "product" else COUNTRY_FEATURE_NAMES

        src_importance = {
            name: float(src_imp_norm[i])
            for i, name in enumerate(src_names[:len(src_imp_norm)])
        }
        dst_importance = {
            name: float(dst_imp_norm[i])
            for i, name in enumerate(dst_names[:len(dst_imp_norm)])
        }

        src_features = {
            name: float(src_vals[i])
            for i, name in enumerate(src_names[:len(src_vals)])
        }
        dst_features = {
            name: float(dst_vals[i])
            for i, name in enumerate(dst_names[:len(dst_vals)])
        }

        return {
            "score": score,
            "src_importance": src_importance,
            "dst_importance": dst_importance,
            "src_features": src_features,
            "dst_features": dst_features,
        }

    finally:
        # Restore original features
        data[src_type].x = original_src_x
        data[dst_type].x = original_dst_x


def explain_top_opportunities(
    model: LinkPredictor,
    data: HeteroData,
    opportunities_df,
    country: str = "MKD",
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Explain the top-N predicted opportunities.

    Parameters
    ----------
    model : Trained LinkPredictor.
    data : HeteroData with graph structure.
    opportunities_df : DataFrame with 'hs4' and 'score' columns.
    country : ISO3 code.
    top_n : Number of top opportunities to explain.

    Returns
    -------
    List of explanation dicts, one per opportunity.
    """
    countries = data["country"].iso3
    products = data["product"].hs4

    if country not in countries:
        return []

    country_idx = countries.index(country)
    product2idx = {p: i for i, p in enumerate(products)}

    explanations = []
    for _, row in opportunities_df.head(top_n).iterrows():
        hs4 = row["hs4"]
        if hs4 not in product2idx:
            continue

        prod_idx = product2idx[hs4]
        try:
            explanation = explain_link(model, data, country_idx, prod_idx)
            explanation["hs4"] = hs4
            explanation["rank"] = int(row.get("rank", 0))
            explanations.append(explanation)
        except Exception as e:
            logger.warning(f"Failed to explain {country}->{hs4}: {e}")

    logger.info(f"Explained {len(explanations)}/{top_n} top opportunities")
    return explanations


def extract_attention_weights(
    model: LinkPredictor,
    data: HeteroData,
) -> dict[str, torch.Tensor]:
    """Extract attention weights from GAT/HGT layers (if applicable).

    Returns dict mapping edge type string to attention tensor,
    or empty dict if model doesn't use attention.
    """
    encoder = model.encoder

    if not isinstance(encoder, (GATEncoder, HGTEncoder)):
        logger.info("Model does not use attention-based encoder, skipping attention extraction")
        return {}

    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)

    attention_weights = {}

    if isinstance(encoder, GATEncoder):
        # GATConv stores attention weights when return_attention_weights=True
        # For now, collect the conv module references
        for layer_idx, layer_dict in enumerate(encoder.convs):
            for et_key, conv in layer_dict.items():
                attention_weights[f"layer{layer_idx}_{et_key}"] = {
                    "type": "GAT",
                    "heads": conv.heads if hasattr(conv, "heads") else 1,
                }

    elif isinstance(encoder, HGTEncoder):
        for layer_idx, conv in enumerate(encoder.convs):
            attention_weights[f"hgt_layer{layer_idx}"] = {
                "type": "HGT",
                "heads": conv.heads if hasattr(conv, "heads") else 1,
            }

    logger.info(f"Attention structure: {list(attention_weights.keys())}")
    return attention_weights
