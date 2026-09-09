"""Dash dashboard for exploring MKD trade opportunities.

Usage:
    python src/mktrade/viz/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback, no_update, dash_table
import dash_bootstrap_components as dbc

from mktrade.config import PROJECT_ROOT
from mktrade.viz.hs_names import (
    HS_CHAPTERS,
    HS_CHAPTERS_MK,
    HS_SECTIONS,
    HS_SECTIONS_MK,
    hs4_display,
    chapter_display,
    iso3_display,
)
from mktrade.viz.i18n import t

# ── Data loading ────────────────────────────────────────────

def _load_csv(name: str) -> pd.DataFrame | None:
    p = PROJECT_ROOT / "reports" / name
    return pd.read_csv(p, index_col=0) if p.exists() else None

def _load_csv_no_idx(name: str) -> pd.DataFrame | None:
    p = PROJECT_ROOT / "reports" / name
    return pd.read_csv(p) if p.exists() else None

def _load_json(name: str):
    p = PROJECT_ROOT / "reports" / name
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

COMP = _load_csv("model_comparison.csv")
TASK_A = _load_csv_no_idx("task_a_opportunities.csv")
TASK_B = _load_csv_no_idx("task_b_opportunities.csv")
ENSEMBLE = _load_csv_no_idx("ensemble_rankings.csv")
EXPLANATIONS = _load_json("explanations.json")


# ── Helpers ─────────────────────────────────────────────────

def _fmt_hs4(code) -> str:
    return f"{int(code):04d}"

def _enrich_task_a(df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    out = df.copy()
    out["hs4_code"] = out["hs4"].apply(_fmt_hs4)
    out["product"] = out["hs4"].apply(lambda x: hs4_display(x, lang))
    out["chapter"] = out["hs4"].apply(lambda x: chapter_display(int(x) // 100, lang))
    out["chapter_num"] = out["hs4"].apply(lambda x: int(x) // 100)
    return out

def _enrich_task_b(df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    out = df.copy()
    out["hs4_code"] = out["hs4"].apply(_fmt_hs4)
    out["product"] = out["hs4"].apply(lambda x: hs4_display(x, lang))
    out["market"] = out["partner_iso3"].apply(lambda x: iso3_display(x, lang))
    return out


# ── Plotly styling ──────────────────────────────────────────

_ACCENT = "#e63946"
_PRIMARY = "#0f172a"
_TEXT_MUTED = "#64748b"
_BORDER = "#e2e8f0"

_PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, sans-serif", size=13, color=_PRIMARY),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=[
        _ACCENT, "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6",
        "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
    ],
)

def _style_fig(fig):
    fig.update_layout(**_PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0",
                     title_font=dict(size=12, color=_TEXT_MUTED),
                     tickfont=dict(size=11, color=_TEXT_MUTED))
    fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0",
                     title_font=dict(size=12, color=_TEXT_MUTED),
                     tickfont=dict(size=11, color=_TEXT_MUTED))
    return fig


# ── App ─────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="MKD Trade Explorer",
)

# ── Navigation structure ────────────────────────────────────

PAGE_IDS = ["overview", "task-a", "task-b", "models", "explanations", "glossary"]
PAGE_ICONS = ["bi-house", "bi-box-seam", "bi-globe", "bi-bar-chart-line", "bi-search", "bi-book"]


def _nav_items(lang):
    keys = ["page_overview", "page_task_a", "page_task_b",
            "page_models", "page_explanations", "page_glossary"]
    items = []
    for pid, icon, key in zip(PAGE_IDS, PAGE_ICONS, keys):
        items.append(
            dbc.NavLink(
                [html.I(className=f"bi {icon} me-2"), t(key, lang)],
                href=f"/{pid}",
                active="exact",
                className="sidebar-link",
            )
        )
    return items


# ── Layout ──────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="lang-store", data="en"),

    # Sidebar
    html.Div([
        # Logo area
        html.Div([
            html.Div("\U0001F1F2\U0001F1F0", style={"fontSize": "2rem"}),
            html.Div("MKD Trade Explorer",
                      style={"fontWeight": "700", "fontSize": "1rem",
                             "color": "#e2e8f0", "marginTop": "4px"}),
        ], style={"textAlign": "center", "padding": "24px 16px 20px"}),

        html.Hr(style={"borderColor": "rgba(255,255,255,0.08)", "margin": "0 16px"}),

        # Nav
        html.Div(id="sidebar-nav", style={"padding": "12px 8px"}),

        html.Hr(style={"borderColor": "rgba(255,255,255,0.08)", "margin": "12px 16px"}),

        # Language toggle
        html.Div([
            dbc.ButtonGroup([
                dbc.Button("EN \U0001F1EC\U0001F1E7", id="btn-en", color="light",
                           outline=True, size="sm", className="lang-btn"),
                dbc.Button("MK \U0001F1F2\U0001F1F0", id="btn-mk", color="light",
                           outline=True, size="sm", className="lang-btn"),
            ], style={"width": "100%"}),
        ], style={"padding": "8px 16px"}),

        # Info
        html.Div([
            html.P([t("sidebar_info", "en")],
                   id="sidebar-info",
                   style={"color": "#94a3b8", "fontSize": "0.78rem",
                          "margin": "0", "lineHeight": "1.5"}),
        ], style={"padding": "16px", "margin": "8px 12px",
                  "background": "rgba(255,255,255,0.04)", "borderRadius": "12px",
                  "border": "1px solid rgba(255,255,255,0.06)"}),

    ], id="sidebar", style={
        "position": "fixed", "left": 0, "top": 0, "bottom": 0,
        "width": "260px",
        "background": f"linear-gradient(180deg, {_PRIMARY} 0%, #1e293b 100%)",
        "overflowY": "auto", "zIndex": 1000,
    }),

    # Main content
    html.Div([
        html.Div(id="page-content"),
    ], style={
        "marginLeft": "260px", "padding": "32px 40px",
        "minHeight": "100vh", "background": "#ffffff",
    }),

], style={"fontFamily": "Inter, -apple-system, sans-serif"})


# ── Custom CSS injected ────────────────────────────────────

app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
        body { margin: 0; background: #fff; }
        .sidebar-link {
            color: #94a3b8 !important;
            border-radius: 10px !important;
            padding: 10px 16px !important;
            margin: 2px 0;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.15s ease;
        }
        .sidebar-link:hover {
            background: rgba(255,255,255,0.08) !important;
            color: #e2e8f0 !important;
        }
        .sidebar-link.active {
            background: rgba(255,255,255,0.12) !important;
            color: #ffffff !important;
            font-weight: 600;
        }
        .lang-btn {
            font-size: 0.8rem !important;
            padding: 6px 12px !important;
            border-color: rgba(255,255,255,0.15) !important;
            color: #94a3b8 !important;
        }
        .lang-btn:hover, .lang-btn.active-lang {
            background: rgba(255,255,255,0.12) !important;
            color: #fff !important;
        }
        .metric-card {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 20px 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: all 0.2s ease;
            height: 100%;
        }
        .metric-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transform: translateY(-2px);
        }
        .metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f172a;
        }
        .metric-delta {
            font-size: 0.82rem;
            color: #10b981;
            font-weight: 500;
        }
        .accent-bar {
            width: 100%; height: 4px;
            background: linear-gradient(90deg, #e63946, #f97316);
            border-radius: 2px;
            margin: 12px 0 24px;
        }
        .section-divider {
            height: 1px;
            background: #e2e8f0;
            margin: 2rem 0;
        }
        .page-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 4px;
        }
        .page-subtitle {
            font-size: 0.92rem;
            color: #64748b;
            line-height: 1.5;
        }
        .pipeline-step {
            display: flex; align-items: center;
            padding: 12px 16px;
            border-left: 3px solid #10b981;
            background: #ecfdf5;
            border-radius: 0 10px 10px 0;
            margin: 6px 0;
        }
        .step-num {
            background: #10b981; color: white;
            width: 28px; height: 28px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 0.82rem; margin-right: 14px; flex-shrink: 0;
        }
        .step-text { font-size: 0.88rem; color: #0f172a; }
        .badge-score {
            display: inline-block; padding: 5px 14px;
            border-radius: 20px; font-weight: 600; font-size: 0.85rem;
        }
        .badge-green { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .badge-amber { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
        .badge-red { background: #fce4e6; color: #991b1b; border: 1px solid #fca5a5; }

        /* Dash DataTable styling */
        .dash-spreadsheet-container .dash-spreadsheet-inner th {
            background-color: #0f172a !important;
            color: white !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 12px 16px !important;
        }
        .dash-spreadsheet-container .dash-spreadsheet-inner td {
            font-size: 0.85rem !important;
            padding: 10px 16px !important;
            color: #0f172a !important;
        }
        .dash-spreadsheet-container .dash-spreadsheet-inner tr:nth-child(even) td {
            background: #f8fafc !important;
        }

        .app-footer {
            text-align: center; padding: 32px 0 16px;
            color: #64748b; font-size: 0.78rem;
            border-top: 1px solid #e2e8f0; margin-top: 3rem;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''


# ── Callbacks ───────────────────────────────────────────────

@callback(
    Output("lang-store", "data"),
    Output("btn-en", "className"),
    Output("btn-mk", "className"),
    Input("btn-en", "n_clicks"),
    Input("btn-mk", "n_clicks"),
    State("lang-store", "data"),
    prevent_initial_call=True,
)
def toggle_language(n_en, n_mk, current):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn_id == "btn-en":
        return "en", "lang-btn active-lang", "lang-btn"
    return "mk", "lang-btn", "lang-btn active-lang"


@callback(
    Output("sidebar-nav", "children"),
    Output("sidebar-info", "children"),
    Output("btn-en", "className", allow_duplicate=True),
    Output("btn-mk", "className", allow_duplicate=True),
    Input("lang-store", "data"),
    prevent_initial_call="initial",
)
def update_sidebar(lang):
    en_cls = "lang-btn active-lang" if lang == "en" else "lang-btn"
    mk_cls = "lang-btn active-lang" if lang == "mk" else "lang-btn"
    return (
        dbc.Nav(_nav_items(lang), vertical=True, pills=True),
        t("sidebar_info", lang),
        en_cls,
        mk_cls,
    )


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("lang-store", "data"),
)
def render_page(pathname, lang):
    lang = lang or "en"
    p = (pathname or "/overview").strip("/")
    if p in ("", "overview"):
        return page_overview(lang)
    elif p == "task-a":
        return page_task_a(lang)
    elif p == "task-b":
        return page_task_b(lang)
    elif p == "models":
        return page_models(lang)
    elif p == "explanations":
        return page_explanations(lang)
    elif p == "glossary":
        return page_glossary(lang)
    return page_overview(lang)


# ── Reusable components ─────────────────────────────────────

def _metric_card(label: str, value: str, delta: str = ""):
    children = [
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
    ]
    if delta:
        children.append(html.Div(delta, className="metric-delta"))
    return dbc.Col(html.Div(children, className="metric-card"), md=3)


def _page_header(title: str, subtitle: str = ""):
    items = [
        html.H1(title, className="page-title"),
    ]
    if subtitle:
        items.append(html.P(subtitle, className="page-subtitle"))
    items.append(html.Div(className="accent-bar"))
    return html.Div(items)


def _divider():
    return html.Div(className="section-divider")


def _score_badge(score: float, label: str = ""):
    cls = "badge-green" if score >= 0.8 else "badge-amber" if score >= 0.5 else "badge-red"
    text = f"{label}: {score:.4f}" if label else f"{score:.4f}"
    return html.Span(text, className=f"badge-score {cls}")


def _make_table(df: pd.DataFrame, page_size: int = 20):
    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "borderRadius": "12px",
                     "border": f"1px solid {_BORDER}"},
        style_header={"backgroundColor": _PRIMARY, "color": "white",
                      "fontWeight": "600", "fontSize": "0.85rem",
                      "padding": "12px 16px"},
        style_cell={"textAlign": "left", "padding": "10px 16px",
                    "fontSize": "0.85rem", "color": _PRIMARY,
                    "border": "none", "borderBottom": f"1px solid {_BORDER}"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
        ],
    )


# ── Page: Overview ──────────────────────────────────────────

def page_overview(lang):
    metrics_row = []

    metrics_row.append(_metric_card(
        t("focus_country", lang), t("mkd_name", lang),
    ))

    if COMP is not None:
        best = COMP.index[0]
        best_auc = COMP.iloc[0].get("roc_auc", 0)
        metrics_row.append(_metric_card(
            t("best_model", lang), best.upper(), f"AUC {best_auc:.4f}",
        ))
    else:
        metrics_row.append(_metric_card(t("best_model", lang), "N/A"))

    if TASK_A is not None:
        metrics_row.append(_metric_card(
            t("product_opportunities", lang), str(len(TASK_A)),
        ))
    else:
        metrics_row.append(_metric_card(t("product_opportunities", lang), "—"))

    if TASK_B is not None:
        n_markets = TASK_B["partner_iso3"].nunique() if "partner_iso3" in TASK_B.columns else 0
        metrics_row.append(_metric_card(t("target_markets", lang), str(n_markets)))

    # Pipeline steps
    pipeline_labels = {
        "en": [
            "Data acquisition & cleaning",
            "Economic complexity metrics",
            "Knowledge graph construction",
            "GNN training & evaluation",
            "Opportunity ranking & dashboard",
        ],
        "mk": [
            "\u041F\u0440\u0438\u0431\u0438\u0440\u0430\u045A\u0435 \u0438 \u0447\u0438\u0441\u0442\u0435\u045A\u0435 \u043D\u0430 \u043F\u043E\u0434\u0430\u0442\u043E\u0446\u0438",
            "\u041C\u0435\u0442\u0440\u0438\u043A\u0438 \u0437\u0430 \u0435\u043A\u043E\u043D\u043E\u043C\u0441\u043A\u0430 \u043A\u043E\u043C\u043F\u043B\u0435\u043A\u0441\u043D\u043E\u0441\u0442",
            "\u041A\u043E\u043D\u0441\u0442\u0440\u0443\u043A\u0446\u0438\u0458\u0430 \u043D\u0430 \u0433\u0440\u0430\u0444 \u043D\u0430 \u0437\u043D\u0430\u0435\u045A\u0435",
            "\u041E\u0431\u0443\u043A\u0430 \u0438 \u0435\u0432\u0430\u043B\u0443\u0430\u0446\u0438\u0458\u0430 \u043D\u0430 GNN",
            "\u0420\u0430\u043D\u0433\u0438\u0440\u0430\u045A\u0435 \u043D\u0430 \u043C\u043E\u0436\u043D\u043E\u0441\u0442\u0438 \u0438 \u0442\u0430\u0431\u043B\u0430",
        ],
    }
    steps = pipeline_labels.get(lang, pipeline_labels["en"])
    pipeline = [
        html.Div([
            html.Div(str(i + 1), className="step-num"),
            html.Div(s, className="step-text"),
        ], className="pipeline-step")
        for i, s in enumerate(steps)
    ]

    # Model perf chart
    chart_section = []
    if COMP is not None:
        from mktrade.viz.plots import model_comparison_bars
        fig = model_comparison_bars(COMP, lang=lang)
        _style_fig(fig)
        chart_section = [
            _divider(),
            html.H2(t("model_perf_summary", lang)),
            html.P(t("model_perf_caption", lang), className="page-subtitle"),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ]

    return html.Div([
        _page_header(t("overview_header", lang)),
        dbc.Row(metrics_row, className="g-3 mb-4"),
        _divider(),
        dbc.Row([
            dbc.Col([
                html.H2(t("overview_what", lang)),
                dcc.Markdown(t("overview_what_body", lang),
                             style={"color": _PRIMARY, "lineHeight": "1.7"}),
            ], md=7),
            dbc.Col([
                html.H2(t("pipeline_summary", lang)),
                html.Div(pipeline),
            ], md=5),
        ]),
        *chart_section,
        html.Div(className="app-footer",
                 children="MKD Trade Opportunity Explorer \u00b7 MSc Thesis \u00b7 Knowledge Graph + GNN"),
    ])


# ── Page: Task A ────────────────────────────────────────────

def page_task_a(lang):
    if TASK_A is None:
        return html.Div([_page_header(t("task_a_header", lang), t("task_a_desc", lang)),
                         dbc.Alert(t("no_data_warning", lang), color="warning")])

    enriched = _enrich_task_a(TASK_A, lang)
    chapters = sorted(enriched["chapter_num"].unique())
    chapter_opts = [{"label": chapter_display(ch, lang), "value": ch} for ch in chapters]

    # Bar chart (top 30)
    from mktrade.viz.plots import opportunity_bar_chart, section_distribution_chart
    filtered = enriched.head(30)
    fig_bar = opportunity_bar_chart(filtered, top_k=30, lang=lang)
    _style_fig(fig_bar)
    fig_bar.update_layout(height=max(400, 30 * 26))

    # Pie chart
    fig_pie = section_distribution_chart(filtered, lang=lang)
    _style_fig(fig_pie)
    fig_pie.update_layout(height=380, margin=dict(t=30))

    # Detail table
    display_df = filtered[
        ["product", "score", "gnn_score", "density", "pci", "chapter"]
    ].copy()
    display_df.columns = [
        t("col_product", lang), t("col_final_score", lang), t("col_gnn_score", lang),
        t("col_density", lang), t("col_pci", lang), t("col_chapter", lang),
    ]
    for c in [
        t("col_final_score", lang), t("col_gnn_score", lang),
        t("col_density", lang), t("col_pci", lang),
    ]:
        if c in display_df.columns:
            display_df[c] = display_df[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    # Ensemble section
    ens_section = []
    if ENSEMBLE is not None:
        ens = ENSEMBLE.head(30).copy()
        ens["product"] = ens["hs4"].apply(lambda x: hs4_display(x, lang))
        ens_col_map = {
            "product": t("col_product", lang),
            "ensemble_score": t("col_ensemble_score", lang),
            "gnn_score": t("col_gnn_score_short", lang),
            "density": t("col_density", lang),
            "classical_score": t("col_classical_score", lang),
        }
        show_cols = [c for c in ens_col_map if c in ens.columns]
        ens_show = ens[show_cols].copy()
        ens_show.columns = [ens_col_map[c] for c in show_cols]
        for c in ens_show.columns:
            if c != t("col_product", lang):
                ens_show[c] = ens_show[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        ens_section = [
            _divider(),
            html.H2(t("ensemble_header", lang)),
            html.P(t("ensemble_caption", lang), className="page-subtitle"),
            _make_table(ens_show),
        ]

    return html.Div([
        _page_header(t("task_a_header", lang), t("task_a_desc", lang)),
        dcc.Graph(figure=fig_bar, config={"displayModeBar": False}),
        _divider(),
        dbc.Row([
            dbc.Col([
                html.H2(t("opportunity_details", lang)),
                _make_table(display_df),
            ], md=7),
            dbc.Col([
                dcc.Graph(figure=fig_pie, config={"displayModeBar": False}),
            ], md=5),
        ]),
        html.Details([
            html.Summary(t("what_columns_mean", lang),
                         style={"cursor": "pointer", "fontWeight": "500",
                                "padding": "12px 0", "color": _PRIMARY}),
            dcc.Markdown(t("task_a_columns_help", lang),
                         style={"color": _PRIMARY, "padding": "0 16px 16px"}),
        ], style={"border": f"1px solid {_BORDER}", "borderRadius": "12px",
                  "padding": "4px 16px", "marginTop": "16px", "background": "#fff"}),
        *ens_section,
    ])


# ── Page: Task B ────────────────────────────────────────────

def page_task_b(lang):
    if TASK_B is None:
        return html.Div([_page_header(t("task_b_header", lang), t("task_b_desc", lang)),
                         dbc.Alert(t("no_data_warning", lang), color="warning")])

    enriched = _enrich_task_b(TASK_B, lang)

    # Product-grouped view (default): first product
    products = sorted(enriched["hs4"].unique())
    first_product = products[0] if products else None

    product_opts = [{"label": hs4_display(p, lang), "value": p} for p in products]
    markets = sorted(enriched["partner_iso3"].unique())
    market_opts = [{"label": iso3_display(m, lang), "value": m} for m in markets]

    gnn_col = t("col_gnn_score", lang)
    market_col = t("col_market", lang)
    product_col = t("col_product", lang)

    # Show top 30 for first product
    if first_product is not None:
        subset = enriched[enriched["hs4"] == first_product].head(30)
        tbl = subset[["market", "score"]].copy()
        tbl.columns = [market_col, gnn_col]
        tbl[gnn_col] = tbl[gnn_col].apply(lambda x: f"{x:.4f}")
        default_title = f"{t('top_markets_for', lang)} {hs4_display(first_product, lang)}"
    else:
        tbl = pd.DataFrame()
        default_title = ""

    return html.Div([
        _page_header(t("task_b_header", lang), t("task_b_desc", lang)),
        dbc.Row([
            dbc.Col([
                html.Label(t("select_product", lang),
                           style={"fontWeight": "500", "fontSize": "0.88rem", "color": _TEXT_MUTED}),
                dcc.Dropdown(
                    id="task-b-product",
                    options=product_opts,
                    value=first_product,
                    style={"borderRadius": "10px"},
                ),
            ], md=8),
        ], className="mb-4"),
        html.H2(default_title, id="task-b-title"),
        html.Div(id="task-b-table", children=_make_table(tbl) if not tbl.empty else ""),
        _divider(),
        html.Details([
            html.Summary(t("what_columns_mean", lang),
                         style={"cursor": "pointer", "fontWeight": "500",
                                "padding": "12px 0", "color": _PRIMARY}),
            dcc.Markdown(t("task_b_columns_help", lang),
                         style={"color": _PRIMARY, "padding": "0 16px 16px"}),
        ], style={"border": f"1px solid {_BORDER}", "borderRadius": "12px",
                  "padding": "4px 16px", "background": "#fff"}),
    ])


@callback(
    Output("task-b-title", "children"),
    Output("task-b-table", "children"),
    Input("task-b-product", "value"),
    State("lang-store", "data"),
)
def update_task_b(product, lang):
    if product is None or TASK_B is None:
        return no_update, no_update
    lang = lang or "en"
    enriched = _enrich_task_b(TASK_B, lang)
    gnn_col = t("col_gnn_score", lang)
    market_col = t("col_market", lang)
    subset = enriched[enriched["hs4"] == product].head(30)
    tbl = subset[["market", "score"]].copy()
    tbl.columns = [market_col, gnn_col]
    tbl[gnn_col] = tbl[gnn_col].apply(lambda x: f"{x:.4f}")
    title = f"{t('top_markets_for', lang)} {hs4_display(product, lang)}"
    return title, _make_table(tbl)


# ── Page: Models ────────────────────────────────────────────

def page_models(lang):
    if COMP is None:
        return html.Div([_page_header(t("models_header", lang), t("models_desc", lang)),
                         dbc.Alert(t("no_data_warning", lang), color="warning")])

    from mktrade.viz.plots import (
        model_comparison_bars,
        model_comparison_heatmap,
        ranking_quality_chart,
    )

    # Metric cards for each model
    metric_cards = []
    if "roc_auc" in COMP.columns:
        for model_name, row in COMP.head(6).iterrows():
            auc_val = row.get("roc_auc", 0)
            ap_val = row.get("avg_precision", 0)
            metric_cards.append(dbc.Col(
                html.Div([
                    html.Div(model_name.upper(), className="metric-label"),
                    html.Div(f"{auc_val:.4f}", className="metric-value"),
                    html.Div(f"AP: {ap_val:.4f}", className="metric-delta"),
                ], className="metric-card"),
                md=2,
            ))

    fig_hm = model_comparison_heatmap(COMP, lang=lang)
    _style_fig(fig_hm)
    fig_hm.update_layout(height=max(300, len(COMP) * 70 + 100))

    fig_bar = model_comparison_bars(COMP, lang=lang)
    _style_fig(fig_bar)

    fig_ranking = ranking_quality_chart(COMP, lang=lang)
    _style_fig(fig_ranking)

    # Full metrics table
    comp_display = COMP.copy()
    comp_display.index = [m.upper() for m in comp_display.index]
    for c in comp_display.columns:
        comp_display[c] = comp_display[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    comp_display = comp_display.reset_index().rename(columns={"index": t("chart_model", lang)})

    return html.Div([
        _page_header(t("models_header", lang), t("models_desc", lang)),
        dbc.Row(metric_cards, className="g-3 mb-4") if metric_cards else html.Div(),
        _divider(),
        dbc.Tabs([
            dbc.Tab(dcc.Graph(figure=fig_hm, config={"displayModeBar": False}),
                    label=t("tab_heatmap", lang)),
            dbc.Tab(dcc.Graph(figure=fig_bar, config={"displayModeBar": False}),
                    label=t("tab_bar_chart", lang)),
            dbc.Tab([
                dcc.Graph(figure=fig_ranking, config={"displayModeBar": False}),
                html.P(t("ranking_quality_caption", lang),
                       className="text-muted px-3 pb-2"),
            ], label=t("tab_ranking_quality", lang)),
        ], className="mb-4"),
        _divider(),
        html.H2(t("detailed_metrics", lang)),
        _make_table(comp_display, page_size=10),
        html.Details([
            html.Summary(t("what_metrics_mean", lang),
                         style={"cursor": "pointer", "fontWeight": "500",
                                "padding": "12px 0", "color": _PRIMARY}),
            dcc.Markdown(t("metrics_help", lang),
                         style={"color": _PRIMARY, "padding": "0 16px 16px"}),
        ], style={"border": f"1px solid {_BORDER}", "borderRadius": "12px",
                  "padding": "4px 16px", "marginTop": "16px", "background": "#fff"}),
    ])


# ── Page: Explanations ──────────────────────────────────────

def page_explanations(lang):
    if not EXPLANATIONS:
        return html.Div([_page_header(t("explain_header", lang), t("explain_desc", lang)),
                         dbc.Alert(t("no_data_warning", lang), color="warning")])

    from mktrade.viz.plots import feature_importance_chart

    score_word = t("score_label", lang)
    exp_opts = [
        {"label": f"#{e.get('rank', i+1)}: {hs4_display(e['hs4'], lang)} ({score_word}={e['score']:.4f})",
         "value": i}
        for i, e in enumerate(EXPLANATIONS)
    ]

    # Default to first
    exp = EXPLANATIONS[0]
    score = exp.get("score", 0)
    fig_src = feature_importance_chart(exp, "country", lang=lang)
    _style_fig(fig_src)
    fig_dst = feature_importance_chart(exp, "product", lang=lang)
    _style_fig(fig_dst)

    return html.Div([
        _page_header(t("explain_header", lang), t("explain_desc", lang)),
        dbc.Row([
            dbc.Col([
                html.Label(t("select_opportunity", lang),
                           style={"fontWeight": "500", "fontSize": "0.88rem", "color": _TEXT_MUTED}),
                dcc.Dropdown(id="explain-select", options=exp_opts, value=0,
                             style={"borderRadius": "10px"}),
            ], md=8),
        ], className="mb-3"),
        html.Div(id="explain-badge", children=_score_badge(score, score_word)),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.H2(t("country_features", lang)),
                html.P(t("country_features_caption", lang), className="page-subtitle"),
                dcc.Graph(id="explain-fig-country", figure=fig_src,
                          config={"displayModeBar": False}),
            ], md=6),
            dbc.Col([
                html.H2(id="explain-product-title",
                         children=f"{t('product_features', lang)} ({hs4_display(exp['hs4'], lang)})"),
                html.P(t("product_features_caption", lang), className="page-subtitle"),
                dcc.Graph(id="explain-fig-product", figure=fig_dst,
                          config={"displayModeBar": False}),
            ], md=6),
        ]),
        html.Details([
            html.Summary(t("country_features_help_title", lang),
                         style={"cursor": "pointer", "fontWeight": "500",
                                "padding": "12px 0", "color": _PRIMARY}),
            dcc.Markdown(t("country_features_help", lang),
                         style={"color": _PRIMARY, "padding": "0 16px 16px"}),
        ], style={"border": f"1px solid {_BORDER}", "borderRadius": "12px",
                  "padding": "4px 16px", "marginTop": "16px", "background": "#fff"}),
        html.Details([
            html.Summary(t("product_features_help_title", lang),
                         style={"cursor": "pointer", "fontWeight": "500",
                                "padding": "12px 0", "color": _PRIMARY}),
            dcc.Markdown(t("product_features_help", lang),
                         style={"color": _PRIMARY, "padding": "0 16px 16px"}),
        ], style={"border": f"1px solid {_BORDER}", "borderRadius": "12px",
                  "padding": "4px 16px", "marginTop": "8px", "background": "#fff"}),
    ])


@callback(
    Output("explain-badge", "children"),
    Output("explain-fig-country", "figure"),
    Output("explain-fig-product", "figure"),
    Output("explain-product-title", "children"),
    Input("explain-select", "value"),
    State("lang-store", "data"),
)
def update_explanation(idx, lang):
    if idx is None or not EXPLANATIONS:
        return no_update, no_update, no_update, no_update
    lang = lang or "en"
    from mktrade.viz.plots import feature_importance_chart
    exp = EXPLANATIONS[idx]
    score = exp.get("score", 0)
    score_word = t("score_label", lang)
    fig_src = feature_importance_chart(exp, "country", lang=lang)
    _style_fig(fig_src)
    fig_dst = feature_importance_chart(exp, "product", lang=lang)
    _style_fig(fig_dst)
    title = f"{t('product_features', lang)} ({hs4_display(exp['hs4'], lang)})"
    return _score_badge(score, score_word), fig_src, fig_dst, title


# ── Page: Glossary ──────────────────────────────────────────

def page_glossary(lang):
    ch_dict = HS_CHAPTERS_MK if lang == "mk" else HS_CHAPTERS
    ch_df = pd.DataFrame([
        {t("col_chapter_num", lang): f"{ch:02d}", t("col_name", lang): name}
        for ch, name in sorted(ch_dict.items())
    ])

    sec_dict = HS_SECTIONS_MK if lang == "mk" else HS_SECTIONS
    sec_df = pd.DataFrame([
        {t("col_section_num", lang): num, t("col_description", lang): desc}
        for num, desc in sec_dict.items()
    ])

    return html.Div([
        _page_header(t("glossary_header", lang), t("glossary_intro", lang)),

        html.H2(t("hs_code_title", lang)),
        dcc.Markdown(t("hs_code_body", lang), style={"color": _PRIMARY, "lineHeight": "1.7"}),

        _divider(),
        html.H2(t("hs_chapters_title", lang)),
        html.Details([
            html.Summary(t("show_all_chapters", lang),
                         style={"cursor": "pointer", "fontWeight": "500",
                                "padding": "12px 0", "color": _PRIMARY}),
            html.Div(_make_table(ch_df, page_size=15), style={"padding": "0 0 16px"}),
        ], style={"border": f"1px solid {_BORDER}", "borderRadius": "12px",
                  "padding": "4px 16px", "background": "#fff"}),

        html.H2(t("hs_sections_title", lang), style={"marginTop": "24px"}),
        html.Details([
            html.Summary(t("show_all_sections", lang),
                         style={"cursor": "pointer", "fontWeight": "500",
                                "padding": "12px 0", "color": _PRIMARY}),
            html.Div(_make_table(sec_df, page_size=21), style={"padding": "0 0 16px"}),
        ], style={"border": f"1px solid {_BORDER}", "borderRadius": "12px",
                  "padding": "4px 16px", "background": "#fff"}),

        _divider(),
        html.H2(t("country_codes_title", lang)),
        dcc.Markdown(t("country_codes_body", lang), style={"color": _PRIMARY}),

        _divider(),
        html.H2(t("trade_metrics_title", lang)),
        dcc.Markdown(t("trade_metrics_body", lang), style={"color": _PRIMARY}),

        _divider(),
        html.H2(t("ml_terms_title", lang)),
        dcc.Markdown(t("ml_terms_body", lang), style={"color": _PRIMARY}),

        _divider(),
        html.H2(t("kg_structure_title", lang)),
        dcc.Markdown(t("kg_structure_body", lang), style={"color": _PRIMARY}),
    ])


# ── Run ─────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=8050)
