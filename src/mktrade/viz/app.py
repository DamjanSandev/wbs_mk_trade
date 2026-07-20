"""Streamlit dashboard for exploring MKD trade opportunities.

Panels:
1. Model comparison — metrics table & charts.
2. Task A rankings — new products for MKD.
3. Task B rankings — new markets per product.
4. Feature importance / explanation viewer.
5. Product-space network visualization.

Usage:
    streamlit run src/mktrade/viz/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mktrade.config import PROJECT_ROOT


# ── Data loading (cached) ───────────────────────────────────

@st.cache_data
def load_model_comparison() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "reports" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_task_a_results() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "reports" / "task_a_opportunities.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_task_b_results() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "reports" / "task_b_opportunities.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_explanations() -> list[dict] | None:
    import json
    path = PROJECT_ROOT / "reports" / "explanations.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_ensemble_results() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "reports" / "ensemble_rankings.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ── Dashboard layout ────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="MKD Trade Opportunities",
        page_icon="🇲🇰",
        layout="wide",
    )

    st.title("North Macedonia Export Opportunity Explorer")
    st.markdown(
        "Knowledge Graph + GNN link prediction for discovering "
        "export-expansion opportunities for North Macedonia."
    )

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select page:",
        ["Overview", "Task A: Products", "Task B: Markets",
         "Model Comparison", "Explanations"],
    )

    if page == "Overview":
        _page_overview()
    elif page == "Task A: Products":
        _page_task_a()
    elif page == "Task B: Markets":
        _page_task_b()
    elif page == "Model Comparison":
        _page_model_comparison()
    elif page == "Explanations":
        _page_explanations()


def _page_overview():
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Focus Country", "North Macedonia (MKD)")
    with col2:
        comp = load_model_comparison()
        if comp is not None:
            best = comp.index[0]
            best_auc = comp.iloc[0].get("roc_auc", 0)
            st.metric("Best Model", best, f"AUC: {best_auc:.4f}")
        else:
            st.metric("Best Model", "N/A")
    with col3:
        task_a = load_task_a_results()
        if task_a is not None:
            st.metric("Product Opportunities", len(task_a))
        else:
            st.metric("Product Opportunities", "Run Phase 5 first")

    st.markdown("---")

    st.subheader("Pipeline Summary")
    st.markdown("""
    | Phase | Description | Status |
    |-------|-------------|--------|
    | 1 | Data acquisition & cleaning | Done |
    | 2 | Economic complexity metrics | Done |
    | 3 | Knowledge graph construction | Done |
    | 4 | GNN training & evaluation | Done |
    | 5 | Opportunity ranking & dashboard | Done |
    """)

    # Show model comparison if available
    comp = load_model_comparison()
    if comp is not None:
        st.subheader("Model Performance Summary")
        from mktrade.viz.plots import model_comparison_bars
        fig = model_comparison_bars(comp)
        st.plotly_chart(fig, use_container_width=True)


def _page_task_a():
    st.header("Task A: Product Diversification")
    st.markdown("New products North Macedonia could develop comparative advantage in.")

    task_a = load_task_a_results()
    if task_a is None:
        st.warning("No Task A results found. Run `python scripts/05_generate_opportunities.py` first.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of products to show", 10, min(200, len(task_a)), 30)
    with col2:
        sections = sorted(task_a["section"].unique())
        selected_sections = st.multiselect("Filter by HS section", sections, default=sections)

    filtered = task_a[task_a["section"].isin(selected_sections)].head(top_k)

    # Bar chart
    from mktrade.viz.plots import opportunity_bar_chart, section_distribution_chart
    fig = opportunity_bar_chart(filtered, top_k=top_k)
    st.plotly_chart(fig, use_container_width=True)

    # Section distribution
    col1, col2 = st.columns(2)
    with col1:
        fig_sec = section_distribution_chart(filtered)
        st.plotly_chart(fig_sec, use_container_width=True)
    with col2:
        st.subheader("Opportunity Details")
        display_cols = [c for c in ["rank", "hs4", "score", "density", "pci", "section"]
                        if c in filtered.columns]
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    # Ensemble results
    ensemble = load_ensemble_results()
    if ensemble is not None:
        st.subheader("Ensemble Rankings (GNN + Density + Classical)")
        st.dataframe(ensemble.head(top_k), use_container_width=True, hide_index=True)


def _page_task_b():
    st.header("Task B: Market Expansion")
    st.markdown("New export markets for North Macedonia's existing products.")

    task_b = load_task_b_results()
    if task_b is None:
        st.warning("No Task B results found. Run `python scripts/05_generate_opportunities.py` first.")
        return

    top_k = st.slider("Number of entries to show", 10, min(200, len(task_b)), 30)

    # Group by product or by market
    view = st.radio("Group by:", ["Product", "Market"], horizontal=True)

    if view == "Product":
        products = sorted(task_b["hs4"].unique())
        selected = st.selectbox("Select product (HS4):", products)
        subset = task_b[task_b["hs4"] == selected].head(top_k)
        st.subheader(f"Top markets for {selected}")
    else:
        markets = sorted(task_b["partner_iso3"].unique())
        selected = st.selectbox("Select market:", markets)
        subset = task_b[task_b["partner_iso3"] == selected].head(top_k)
        st.subheader(f"Top products for market {selected}")

    display_cols = [c for c in ["rank", "hs4", "partner_iso3", "score", "section"]
                    if c in subset.columns]
    st.dataframe(subset[display_cols], use_container_width=True, hide_index=True)

    # Full table
    with st.expander("Show full Task B results"):
        st.dataframe(task_b.head(200), use_container_width=True, hide_index=True)


def _page_model_comparison():
    st.header("Model Comparison")

    comp = load_model_comparison()
    if comp is None:
        st.warning("No model comparison results found. Run Phase 4 first.")
        return

    from mktrade.viz.plots import model_comparison_heatmap, model_comparison_bars

    # Heatmap
    fig_hm = model_comparison_heatmap(comp)
    st.plotly_chart(fig_hm, use_container_width=True)

    # Bar chart
    fig_bar = model_comparison_bars(comp)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Raw table
    st.subheader("Detailed Metrics")
    st.dataframe(comp.style.format("{:.4f}"), use_container_width=True)


def _page_explanations():
    st.header("Link Explanations")
    st.markdown("Feature importance for top predicted opportunities.")

    explanations = load_explanations()
    if not explanations:
        st.warning("No explanations found. Run `python scripts/05_generate_opportunities.py` first.")
        return

    from mktrade.viz.plots import feature_importance_chart

    # Select explanation
    options = [f"#{e.get('rank', i+1)}: HS {e['hs4']} (score={e['score']:.4f})"
               for i, e in enumerate(explanations)]
    selected_idx = st.selectbox("Select opportunity:", range(len(options)),
                                format_func=lambda i: options[i])

    exp = explanations[selected_idx]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Country Features (MKD)")
        fig_src = feature_importance_chart(exp, "country")
        st.plotly_chart(fig_src, use_container_width=True)

        st.markdown("**Raw feature values:**")
        if "src_features" in exp:
            st.json(exp["src_features"])

    with col2:
        st.subheader(f"Product Features (HS {exp['hs4']})")
        fig_dst = feature_importance_chart(exp, "product")
        st.plotly_chart(fig_dst, use_container_width=True)

        if "dst_features" in exp:
            st.markdown("**Raw feature values:**")
            st.json(exp["dst_features"])


if __name__ == "__main__":
    main()
