"""Reusable Plotly chart builders for the dashboard and reports."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mktrade.viz.i18n import t


# The redesigned held-out test contains 890 sustained-success links among
# 235,019 eligible candidates. Positive prevalence is the expected AP of a
# random ranking, so it is the meaningful reference point for AP lift.
_RANDOM_AP_BASELINE = 890 / 235_019


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
    title: str | None = None,
    lang: str = "en",
) -> go.Figure:
    """Horizontal bar chart of top-K opportunity scores, coloured by section."""
    from mktrade.viz.hs_names import hs4_display

    if title is None:
        title = t("chart_top_opportunities", lang)

    df = opportunities.head(top_k).copy()
    df["section_name"] = df["section"].apply(_section_label)
    if "product" in df.columns:
        df["label"] = df["product"]
    else:
        df["label"] = df["hs4"].apply(lambda x: hs4_display(x, lang))
    df = df.sort_values("score", ascending=True)

    fig = px.bar(
        df, x="score", y="label", orientation="h",
        color="section_name",
        title=title,
        labels={
            "score": t("chart_gnn_link_score", lang),
            "label": t("chart_product", lang),
            "section_name": t("chart_chapter", lang),
        },
    )
    fig.update_layout(
        height=max(400, top_k * 28),
        yaxis_title="",
        showlegend=True,
        legend_title=t("chart_hs_chapter", lang),
    )
    return fig


def model_comparison_heatmap(comparison_df: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Heatmap of models (rows) x metrics (columns)."""
    display_cols = [c for c in ["roc_auc", "avg_precision", "mrr",
                                "precision@50", "recall@50", "hits@50"]
                    if c in comparison_df.columns]

    df = comparison_df[display_cols].copy()

    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=[c.replace("_", " ").title() for c in df.columns],
        y=[m.upper() for m in df.index.tolist()],
        colorscale=[[0, "#fee2e2"], [0.5, "#fef3c7"], [1, "#d1fae5"]],
        text=[[f"{v:.4f}" for v in row] for row in df.values],
        texttemplate="%{text}",
        textfont={"size": 13, "color": "#0f172a"},
    ))
    fig.update_layout(
        title=t("chart_model_comparison", lang),
        height=max(300, len(df) * 60 + 100),
        xaxis_title=t("chart_metric", lang),
        yaxis_title=t("chart_model", lang),
    )
    return fig


def model_comparison_bars(comparison_df: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Grouped bar chart comparing ROC-AUC and Average Precision."""
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
        barmode="group",
        title=t("chart_model_perf", lang),
        labels={
            "value": t("chart_score", lang),
            "model": t("chart_model", lang),
            "metric_label": t("chart_metric", lang),
        },
    )
    fig.update_layout(height=400)
    return fig


def ranking_quality_chart(
    comparison_df: pd.DataFrame,
    lang: str = "en",
    random_ap_baseline: float = _RANDOM_AP_BASELINE,
) -> go.Figure:
    """Show top-10 ranking quality and Average Precision lift over random.

    Average Precision uses a different scale from ROC-AUC in this highly
    imbalanced candidate universe. Expressing it as lift over positive
    prevalence makes the result interpretable while retaining raw AP in the
    hover information.
    """
    if random_ap_baseline <= 0:
        raise ValueError("random_ap_baseline must be positive")

    models = [str(model).upper() for model in comparison_df.index]
    top_k_metrics = [
        ("precision@10", "Precision@10", "#2563eb"),
        ("map@10", "MAP@10", "#7c3aed"),
        ("ndcg@10", "NDCG@10", "#0d9488"),
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.12,
        subplot_titles=(t("chart_top10_quality", lang), t("chart_ap_lift", lang)),
    )

    for column, label, color in top_k_metrics:
        if column not in comparison_df.columns:
            continue
        values = comparison_df[column].astype(float)
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=label,
                marker_color=color,
                hovertemplate=(
                    f"{t('chart_model', lang)}: %{{x}}<br>"
                    f"{label}: %{{y:.2%}}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    if "avg_precision" in comparison_df.columns:
        ap_values = comparison_df["avg_precision"].astype(float)
        lift_values = ap_values / random_ap_baseline
        fig.add_trace(
            go.Bar(
                x=models,
                y=lift_values,
                name=t("chart_ap_lift_short", lang),
                marker_color="#f97316",
                customdata=ap_values,
                text=[f"{value:.1f}×" for value in lift_values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    f"{t('chart_model', lang)}: %{{x}}<br>"
                    f"{t('chart_avg_precision', lang)}: %{{customdata:.3%}}<br>"
                    f"{t('chart_ap_lift_short', lang)}: %{{y:.2f}}×<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#64748b",
            annotation_text=t("chart_random_baseline", lang),
            annotation_position="top left",
            row=1,
            col=2,
        )

    fig.update_layout(
        title=t("chart_ranking_quality", lang),
        barmode="group",
        height=470,
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
        margin=dict(t=120),
    )
    fig.update_xaxes(title_text=t("chart_model", lang), row=1, col=1)
    fig.update_xaxes(title_text=t("chart_model", lang), row=1, col=2)
    fig.update_yaxes(title_text=t("chart_score", lang), tickformat=".1%", row=1, col=1)
    fig.update_yaxes(title_text=t("chart_lift", lang), ticksuffix="×", rangemode="tozero", row=1, col=2)
    return fig


def feature_importance_chart(
    explanation: dict,
    node_type: str = "country",
    top_n: int = 8,
    lang: str = "en",
) -> go.Figure:
    """Bar chart showing feature importance for a link explanation."""
    key = f"{node_type.split('_')[0]}_importance"
    if key not in explanation:
        key = "src_importance" if node_type == "country" else "dst_importance"

    importance = explanation.get(key, {})
    if not importance:
        return go.Figure().update_layout(title=t("chart_no_importance", lang))

    df = pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in importance.items()
    ])
    df = df.sort_values("importance", ascending=False).head(top_n)
    df = df.sort_values("importance", ascending=True)

    node_label = t("chart_country", lang) if node_type == "country" else t("chart_product_node", lang)
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        title=f"{t('chart_feature_importance', lang)} ({node_label})",
        labels={
            "importance": t("chart_importance", lang),
            "feature": t("chart_feature", lang),
        },
    )
    fig.update_layout(height=max(250, top_n * 35))
    return fig


def section_distribution_chart(opportunities: pd.DataFrame, lang: str = "en") -> go.Figure:
    """Pie/treemap of opportunity distribution across HS chapters."""
    from mktrade.viz.hs_names import HS_CHAPTERS, HS_CHAPTERS_MK, chapter_display

    df = opportunities.copy()
    df["chapter_num"] = df["hs4"].apply(lambda x: int(x) // 100)
    df["chapter_name"] = df["chapter_num"].apply(lambda c: chapter_display(c, lang))
    chapter_counts = df["chapter_name"].value_counts().reset_index()
    chapter_counts.columns = ["chapter", "count"]

    fig = px.pie(
        chapter_counts, values="count", names="chapter",
        title=t("chart_opportunities_by_chapter", lang),
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=400, showlegend=False)
    return fig


def temporal_validation_chart(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> go.Figure:
    """Show how many top-K predictions appeared in held-out years."""
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
    """Simplified product-space network as a Plotly scatter (spring layout)."""
    import networkx as nx

    prox = proximity_df.copy()
    prox = prox[prox["proximity"] > 0].nlargest(top_edges, "proximity")

    G = nx.Graph()
    for _, row in prox.iterrows():
        G.add_edge(row["hs4_1"], row["hs4_2"], weight=row["proximity"])

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

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
