# MKD Trade Opportunity Explorer

**Knowledge Graph + GNN Link Prediction for North Macedonia Export Opportunities**

MSc Thesis project that predicts new export opportunities for North Macedonia using Graph Neural Networks trained on international trade data.

- **Task A (Product Diversification):** Which new products could MKD start exporting competitively?
- **Task B (Market Expansion):** For products MKD already exports, which new countries could it sell to?

## Results

| Model | ROC AUC | Avg Precision |
|-------|---------|---------------|
| GAT | 0.7803 | 0.7616 |
| HGT | 0.6800 | 0.6692 |
| Density Baseline | 0.6500 | 0.6532 |
| GraphSAGE | 0.6295 | 0.5932 |
| GCN | 0.5978 | 0.5730 |

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
# Phase 1: Download and clean data (~15 min, needs internet + Comtrade API key)
python scripts/00_download_data.py

# Phase 2: Compute economic complexity metrics (RCA, ECI, PCI, proximity)
python scripts/01_build_complexity.py

# Phase 3: Build knowledge graph (NetworkX + PyG HeteroData)
python scripts/02_build_graph.py

# Phase 4: Train GNN models (GAT, HGT, GraphSAGE, GCN) + baselines
python scripts/04_train_models.py

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

## Two Prediction Tasks

| Task | Link Type | Question |
|------|-----------|----------|
| **A** (Product Diversification) | Country → Product | What new products could MKD export? |
| **B** (Market Expansion) | Country → Product → Country | Which new markets for existing exports? |

## Knowledge Graph

The graph has **3 node types** and **7 edge types**:

- **Nodes:** Country (230), Product (1,241), HS Section (21)
- **Edges:** `exports`, `proximity`, `in_section`, `trades_with`, `neighbor_of` + reverse edges
- **Country features (14-dim):** ECI, diversity, GDP, population, landlocked, EU, CEFTA, region
- **Product features (23-dim):** PCI, ubiquity, section one-hot

## Models & Baselines

**GNN encoders:** GAT, HGT, GraphSAGE, GCN

**Baselines:** Economic complexity density, PPML gravity model, Adamic-Adar, Jaccard coefficient, Common Neighbors

**Evaluation:** Temporal holdout (train ≤ 2019, test 2021-2022). Metrics: ROC-AUC, Average Precision, MRR, Precision@K, Recall@K, Hits@K.

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

- **Temporal split:** Train ≤ 2019, validation 2020, test 2021–2022
- **Proximity sparsification:** HGT uses top-50 neighbors per product (prevents OOM); GAT/SAGE/GCN use full proximity edges
- **Graceful degradation:** Neo4j optional (NetworkX fallback), Comtrade key optional (cached data fallback)
- **Ensemble:** 60% GNN + 30% density + 10% classical baselines
- **Explanations:** Gradient-based attribution (input × gradient) for feature importance

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
