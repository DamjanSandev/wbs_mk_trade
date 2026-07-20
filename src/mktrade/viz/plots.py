"""Reusable Plotly chart builders for the dashboard and reports."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# HS section code -> broad name (for labelling)
_SECTION_NAMES = {
    "01": "Animals", "02": "Meat", "03": "Fish", "04": "Dairy",
    "05": "Animal Prod.", "06": "Trees", "07": "Vegetables", "08": "Fruits",
    "09": "Coffee/Tea", "10": "Cereals", "11": "Milling", "12": "Oil Seeds",
    "13": "Lac/Gums", "14": "Veg. Plaiting", "15": "Fats/Oils",
    "16": "Prep. Meat", "17": "Sugars", "18": "Cocoa", "19": "Cereal Prep.",
    "20": "Veg. Prep.", "21": "Misc. Food", "22": "Beverages",
    "23": "Residues", "24": "Tobacco", "25": "Salt/Stone", "26": "Ores",
    "27": "Fuels", "28": "Inorg. Chem.", "29": "Org. Chem.",
    "30": "Pharma", "31": "Fertilizers", "32": "Dyes", "33": "Cosmetics",
    "34": "Soap/Wax", "35": "Albuminoidal", "36": "Explosives",
    "37": "Photo", "38": "Misc. Chem.", "39": "Plastics", "40": "Rubber",
    "41": "Hides", "42": "Leather", "43": "Fur", "44": "Wood",
    "45": "Cork", "46": "Straw", "47": "Pulp", "48": "Paper",
    "49": "Printed", "50": "Silk", "51": "Wool", "52": "Cotton",
    "53": "Veg. Fibers", "54": "Filaments", "55": "Staple Fibers",
    "56": "Wadding", "57": "Carpets", "58": "Special Woven",
    "59": "Coated Text.", "60": "Knit Fabrics", "61": "Knit Apparel",
    "62": "Woven Apparel", "63": "Other Text.", "64": "Footwear",
    "65": "Headgear", "66": "Umbrellas", "67": "Feathers",
    "68": "Stone/Cement", "69": "Ceramics", "70": "Glass",
    "71": "Precious Met.", "72": "Iron/Steel", "73": "Iron Art.",
    "74": "Copper", "75": "Nickel", "76": "Aluminium", "78": "Lead",
    "79": "Zinc", "80": "Tin", "81": "Other Metals", "82": "Tools",
    "83": "Misc. Metal", "84": "Machinery", "85": "Electrical",
    "86": "Railway", "87": "Vehicles", "88": "Aircraft", "89": "Ships",
    "90": "Optical/Med.", "91": "Clocks", "92": "Musical",
    "93": "Arms", "94": "Furniture", "95": "Toys", "96": "Misc. Mfg.",
    "97": "Art/Antiques",
}


def _section_label(code: str) -> str:
    return _SECTION_NAMES.get(code, f"Ch.{code}")


def opportunity_bar_chart(
    opportunities: pd.DataFrame,
    top_k: int = 20,
    title: str = "Top Predicted Product Opportunities for MKD",
) -> go.Figure:
    """Horizontal bar chart of top-K opportunity scores, coloured by section."""
    df = opportunities.head(top_k).copy()
    df["section_name"] = df["section"].apply(_section_label)
    df["label"] = df["hs4"] + " (" + df["section_name"] + ")"
    df = df.sort_values("score", ascending=True)  # ascending for horizontal bar

    fig = px.bar(
        df, x="score", y="label", orientation="h",
        color="section_name",
        title=title,
        labels={"score": "GNN Link Score", "label": "Product (HS4)", "section_name": "Section"},
    )
    fig.update_layout(
        height=max(400, top_k * 25),
        yaxis_title="",
        showlegend=True,
        legend_title="HS Section",
    )
    return fig


def model_comparison_heatmap(comparison_df: pd.DataFrame) -> go.Figure:
    """Heatmap of models (rows) x metrics (columns)."""
    display_cols = [c for c in ["roc_auc", "avg_precision", "mrr",
                                "precision@50", "recall@50", "hits@50"]
                    if c in comparison_df.columns]

    df = comparison_df[display_cols].copy()

    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=[c.replace("_", " ").title() for c in df.columns],
        y=df.index.tolist(),
        colorscale="RdYlGn",
        text=[[f"{v:.4f}" for v in row] for row in df.values],
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    fig.update_layout(
        title="Model Comparison",
        height=max(300, len(df) * 60 + 100),
        xaxis_title="Metric",
        yaxis_title="Model",
    )
    return fig


def model_comparison_bars(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing models on key metrics."""
    metrics = ["roc_auc", "avg_precision"]
    available = [m for m in metrics if m in comparison_df.columns]

    df = comparison_df[available].copy()
    df = df.reset_index()
    df = df.rename(columns={"index": "model"})
    if "model" not in df.columns and df.index.name == "model":
        df = df.reset_index()

    melted = df.melt(id_vars=["model"], var_name="metric", value_name="value")
    melted["metric_label"] = melted["metric"].str.replace("_", " ").str.title()

    fig = px.bar(
        melted, x="model", y="value", color="metric_label",
        barmode="group", title="Model Performance Comparison",
        labels={"value": "Score", "model": "Model", "metric_label": "Metric"},
    )
    fig.update_layout(height=400)
    return fig


def feature_importance_chart(
    explanation: dict,
    node_type: str = "country",
    top_n: int = 8,
) -> go.Figure:
    """Bar chart showing feature importance for a link explanation."""
    key = f"{node_type.split('_')[0]}_importance"
    if key not in explanation:
        key = "src_importance" if node_type == "country" else "dst_importance"

    importance = explanation.get(key, {})
    if not importance:
        return go.Figure().update_layout(title="No importance data available")

    df = pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in importance.items()
    ])
    df = df.sort_values("importance", ascending=False).head(top_n)
    df = df.sort_values("importance", ascending=True)

    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        title=f"Feature Importance ({node_type.title()})",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(height=max(250, top_n * 35))
    return fig


def section_distribution_chart(opportunities: pd.DataFrame) -> go.Figure:
    """Pie/treemap of opportunity distribution across HS sections."""
    df = opportunities.copy()
    df["section_name"] = df["section"].apply(_section_label)
    section_counts = df["section_name"].value_counts().reset_index()
    section_counts.columns = ["section", "count"]

    fig = px.pie(
        section_counts, values="count", names="section",
        title="Opportunities by HS Section",
    )
    fig.update_layout(height=400)
    return fig


def temporal_validation_chart(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> go.Figure:
    """Show how many top-K predictions appeared in held-out years.

    Parameters
    ----------
    predictions : DataFrame with 'hs4' and 'score' columns (ranked).
    actuals : DataFrame with 'hs4' column (products that actually appeared).
    """
    actual_set = set(actuals["hs4"].unique())
    ks = [5, 10, 20, 50, 100, 200]
    hit_rates = []

    for k in ks:
        if k > len(predictions):
            break
        top_k = set(predictions.head(k)["hs4"])
        hits = len(top_k & actual_set)
        hit_rates.append({"k": k, "hits": hits, "hit_rate": hits / k if k > 0 else 0})

    df = pd.DataFrame(hit_rates)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df["k"], y=df["hits"], name="Hits (count)"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["k"], y=df["hit_rate"], name="Hit rate", mode="lines+markers"),
        secondary_y=True,
    )
    fig.update_layout(title="Temporal Validation: Predictions vs. Future Exports")
    fig.update_xaxes(title_text="Top-K")
    fig.update_yaxes(title_text="Number of hits", secondary_y=False)
    fig.update_yaxes(title_text="Hit rate", secondary_y=True)

    return fig


def product_space_network(
    proximity_df: pd.DataFrame,
    highlight_products: list[str] | None = None,
    top_edges: int = 2000,
) -> go.Figure:
    """Simplified product-space network as a Plotly scatter (spring layout).

    For the full interactive version, see the Streamlit dashboard which uses pyvis.
    """
    import networkx as nx

    prox = proximity_df.copy()
    prox = prox[prox["proximity"] > 0].nlargest(top_edges, "proximity")

    G = nx.Graph()
    for _, row in prox.iterrows():
        G.add_edge(row["hs4_1"], row["hs4_2"], weight=row["proximity"])

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.3, color="#ccc"), hoverinfo="none",
    ))

    # Nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())
    node_color = []
    for n in G.nodes():
        if highlight_products and n in highlight_products:
            node_color.append("red")
        else:
            section = n[:2] if len(n) >= 2 else "00"
            node_color.append(int(section) % 20)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=4, color=node_color, colorscale="Viridis"),
        text=node_text, hoverinfo="text",
    ))

    fig.update_layout(
        title="Product Space Network (top proximity edges)",
        showlegend=False, height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig
