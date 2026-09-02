# MKD Trade Opportunity Explorer

**Knowledge Graph + GNN Link Prediction for North Macedonia Export Opportunities**

MSc Thesis project that predicts new export opportunities for North Macedonia using Graph Neural Networks trained on international trade data.

- **Task A (Product Diversification):** Which new products could MKD start exporting competitively?
- **Task B (Market Expansion):** For products MKD already exports, which new countries could it sell to?

## Results

Latest regenerated single-split results under the redesigned target and
complete-candidate evaluation protocol:

| Model | ROC-AUC | Average precision | Precision@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| Density baseline | 0.7036 | **0.0117** | **0.0311** | **0.0718** |
| HGT | **0.7115** | 0.0085 | 0.0176 | 0.0498 |
| GCN | 0.6512 | 0.0064 | 0.0124 | 0.0222 |
| GraphSAGE | 0.6429 | 0.0060 | 0.0062 | 0.0118 |
| GAT | 0.5684 | 0.0047 | 0.0073 | 0.0110 |

Average precision is measured over a candidate universe with only 0.379%
positives; its random-ranking baseline is approximately 0.00379. See
[Results after the redesign](#results-after-the-redesign) for the full
before-and-after comparison and interpretation.

## Quick Start

### Prerequisites

- Python 3.11+
- Git

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/wbs-mk-trade.git
cd wbs-mk-trade
pip install -e .
```

### 2. Environment setup

```bash
cp .env.example .env
# Edit .env and add your UN Comtrade API key (free at https://comtradeplus.un.org/)
# Neo4j is optional — the pipeline falls back to NetworkX automatically
```

### 3. Run the dashboard (quick — uses pre-computed results)

The `reports/` directory contains pre-computed results, so the dashboard works immediately:

```bash
python src/mktrade/viz/app.py
# Open http://localhost:8050
```

### 4. Reproduce from scratch (full pipeline)

To rebuild everything from raw data:

```bash
# Phase 1: Download and clean data (needs internet + Comtrade API key)
python scripts/00_download_data.py

# Phase 2: Compute economic complexity metrics (RCA, ECI, PCI, proximity)
python scripts/01_build_complexity.py

# Phase 3: Build knowledge graph (NetworkX + PyG HeteroData)
python scripts/02_build_graph.py

# Phase 4: Train GNN models (GAT, HGT, GraphSAGE, GCN) + baselines
python scripts/04_train_models.py

# Optional robust evaluation: rolling cutoffs x configured random seeds
python scripts/04_train_models.py --model gat --rolling-backtest

# Phase 5: Generate opportunity rankings, ensemble scores, explanations
python scripts/05_generate_opportunities.py

# Launch dashboard
python src/mktrade/viz/app.py
```

## Project Structure

```
wbs-mk-trade/
├── configs/                    # YAML configuration files
│   ├── data.yaml              # Data source paths and parameters
│   ├── graph.yaml             # Graph construction settings
│   ├── train.yaml             # Training hyperparameters
│   └── model/                 # Per-model configs (GAT, HGT, GCN, etc.)
├── data/
│   ├── raw/                   # Raw downloads (gitignored)
│   ├── processed/             # Cleaned parquets, PyG data (gitignored)
│   └── external/              # CEPII gravity, WDI (gitignored)
├── models/                    # Saved model checkpoints (gitignored)
├── reports/                   # Generated results (tracked in git)
│   ├── task_a_opportunities.csv
│   ├── task_b_opportunities.csv
│   ├── ensemble_rankings.csv
│   ├── model_comparison.csv
│   └── explanations.json
├── scripts/                   # Pipeline scripts (run in order)
│   ├── 00_download_data.py    # Download + clean all data sources
│   ├── 01_build_complexity.py # Economic complexity metrics
│   ├── 02_build_graph.py      # Knowledge graph construction
│   ├── 04_train_models.py     # GNN training + baselines
│   └── 05_generate_opportunities.py  # Ranking + explanations
├── src/mktrade/               # Main package
│   ├── config.py              # Pydantic-settings config loader
│   ├── data/                  # Data loading, cleaning, ISO/M49 mapping
│   ├── complexity/            # Economic complexity (RCA, ECI, PCI)
│   ├── graph/                 # NetworkX, Neo4j, PyG HeteroData builders
│   ├── models/                # GNN encoders (GAT, HGT, SAGE, GCN)
│   ├── train/                 # Training loop, temporal splits
│   ├── eval/                  # Evaluation metrics, gravity baseline
│   ├── opportunities/         # Ranking, ensemble, explanations
│   └── viz/                   # Dashboard (Dash/Plotly), i18n, HS names
├── tests/                     # Unit tests (pytest)
├── pyproject.toml             # Dependencies and tool config
└── .env.example               # Environment variable template
```

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| [Harvard Atlas](https://dataverse.harvard.edu/) | HS4 exports (country × product × year) | Free download |
| [UN Comtrade](https://comtradeplus.un.org/) | Bilateral trade flows | Free API key required |
| [CEPII Gravity](http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8) | Distance, contiguity, language, FTA | Free download |
| [World Bank WDI](https://databank.worldbank.org/) | GDP, GDP per capita, population | Free API |

All data is downloaded automatically by `scripts/00_download_data.py`. The only manual requirement is a Comtrade API key in `.env`.
Task B demand coverage is controlled by `comtrade.market_reporters` in
`configs/data.yaml`; the default list stays within the configured daily API-call
budget but takes longer than the earlier MKD-only download.

## Two Prediction Tasks

| Task | Link Type | Question |
|------|-----------|----------|
| **A** (Product Diversification) | Country → Product | What new products could MKD export? |
| **B** (Market Expansion) | Country → Product → Country | Which new markets for existing exports? |

Task B is implemented as an origin-product-destination ranking problem. The
destination GNN compatibility score is combined with destination import value
and growth, supplier concentration, existing MKD route history, total bilateral
trade, distance, language, FTA status, GDP, population, MKD supply capacity,
and an optional tariff table (`data/external/tariffs.parquet`). Missing market
data is represented explicitly rather than silently interpreted as zero demand.

## Knowledge Graph

The graph has **3 node types** and **7 edge types**:

- **Nodes:** Country (230), Product (1,241), HS Section (21)
- **Edges:** `exports`, `proximity`, `in_section`, `trades_with`, `neighbor_of` + reverse edges
- **Country features (14-dim):** ECI, diversity, GDP, population, landlocked, EU, CEFTA, region
- **Product features (23-dim):** PCI, ubiquity, section one-hot

## Models & Baselines

**GNN encoders:** GAT, HGT, GraphSAGE, GCN

**Baselines:** Economic complexity density, PPML gravity model, Adamic-Adar, Jaccard coefficient, Common Neighbors

**Evaluation:** Temporal holdout plus optional rolling-origin backtests. Every
unseen product is ranked separately for each country. Headline metrics are
query-based MRR, Precision@10, Recall@10, MAP@10, and NDCG@10; ROC-AUC and
global average precision remain diagnostic metrics.

## Ranking redesign

The production ranking path now follows these rules:

1. A successful export must have annual value of at least USD 100,000, RCA >= 1,
   and remain successful for at least two consecutive years. These thresholds
   are configurable in `configs/train.yaml`.
2. Validation and test sets contain the complete filtered candidate universe,
   not one sampled negative per positive. Validation positives are removed from
   later test candidates.
3. Every candidate is scored before top-K selection. GNN, density, PCI,
   complexity gain, demand, persistence, gravity, and market feasibility signals
   can therefore change the final order.
4. A regularised logistic reranker learns feature weights from validation-era
   outcomes. Its imputation and scaling statistics are fitted on validation
   data, and its output is a success probability. The former 60/30/10 manual
   blend has been removed.
5. Bootstrap reranker fits report `probability_lower`, `probability_upper`,
   `average_rank`, `rank_std`, and `rank_stability` for every recommendation.
6. GAT consumes standardised edge attributes for exports/RCA, proximity,
   bilateral trade, and gravity relations.

If validation data contains only one class, report generation emits a warning
and uses an equal-percentile fallback over non-constant signals. It does not
label that fallback as a calibrated probability.

### Results after the redesign

The following table compares the archived implementation with the regenerated
single-split results in `reports/model_comparison.csv`.

| Model | Previous ROC-AUC | Redesigned ROC-AUC | Previous average precision | Redesigned average precision | Redesigned Precision@10 | Redesigned NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Density baseline | 0.6500 | 0.7036 | 0.6532 | **0.0117** | **0.0311** | **0.0718** |
| HGT | 0.6800 | **0.7115** | 0.6692 | 0.0085 | 0.0176 | 0.0498 |
| GCN | 0.6165 | 0.6512 | 0.5929 | 0.0064 | 0.0124 | 0.0222 |
| GraphSAGE | 0.6177 | 0.6429 | 0.5855 | 0.0060 | 0.0062 | 0.0118 |
| GAT | **0.7803** | 0.5684 | **0.7616** | 0.0047 | 0.0073 | 0.0110 |

These average-precision columns are not directly comparable. The previous test
sample contained 27,247 positive links and one randomly sampled negative per
positive, giving a 50% positive rate. The redesigned test ranks all 235,019
eligible links, of which only 890 (0.379%) meet the sustained-success target.
A random ranking therefore has redesigned average precision of approximately
0.00379. Relative to that baseline, Density reaches about 3.1x random performance,
HGT 2.2x, GCN 1.7x, GraphSAGE 1.6x, and GAT 1.2x.

The redesigned top-K metrics are macro-averaged across 193 country queries,
whereas the previous implementation ranked all country-product edges in one
global list. Under the deployment-aligned evaluation, HGT has the best overall
ROC-AUC, while Density is strongest at the top of the recommendation list. GAT's
former lead does not persist under the stricter target and complete candidate
universe.

## Dashboard

Interactive dashboard built with Dash (Plotly) supporting **English and Macedonian**:

- Product diversification rankings with HS chapter filters
- Market expansion grouped by product or country
- Model comparison (heatmap + grouped bar charts)
- Gradient-based feature importance explanations
- Glossary of HS codes, trade metrics, and ML terminology

```bash
python src/mktrade/viz/app.py
# → http://localhost:8050
```

## Key Technical Notes

- **Success target:** RCA >= 1, value >= USD 100,000, active for 2 consecutive years
- **Temporal split:** Train ≤ 2019, validation onset 2020, test 2021–2022;
  validation relationships are filtered from test
- **Candidate evaluation:** all unseen products per country, with macro-averaged query metrics
- **Model selection:** validation NDCG@10
- **Robustness:** optional rolling cutoffs `[2016, 2017, 2018, 2019]` and seeds `[13, 42, 73]`
- **Proximity sparsification:** HGT uses top-50 neighbors per product (prevents OOM); GAT/SAGE/GCN use full proximity edges
- **Graceful degradation:** Neo4j optional (NetworkX fallback), Comtrade key optional (cached data fallback)
- **Reranker:** validation-trained logistic model with probability calibration and bootstrap stability
- **Market demand coverage:** Comtrade downloads include configured destination reporters in addition to MKD
- **Explanations:** Gradient-based attribution (input × gradient) for feature importance

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
