# MKD Trade Opportunity Explorer

**Knowledge Graph + GNN Link Prediction for North Macedonia Export Opportunities**

MSc Thesis project that ranks new export opportunities for North Macedonia by
combining graph neural networks, economic-complexity signals, gravity and market
features, and a validation-trained reranker.

- **Task A (Product Diversification):** Which new products could MKD start exporting competitively?
- **Task B (Market Expansion):** For products MKD already exports, which new countries could it sell to?

## Results

These are the current single-split results in `reports/model_comparison.csv`.
Models train on information available through 2018, validate against sustained
export outcomes in 2019-2020, and test on new sustained outcomes in 2021-2022.

| Model | ROC-AUC | Average precision | MRR | Precision@10 | Recall@10 | MAP@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Density baseline** | **0.7289** | **0.0187** | **0.1935** | **0.0550** | **0.1047** | **0.0417** | **0.0890** |
| HGT | 0.7204 | 0.0152 | 0.1231 | 0.0292 | 0.0514 | 0.0258 | 0.0500 |
| Learned reranker | 0.5979 | 0.0122 | 0.1258 | 0.0426 | 0.0650 | 0.0225 | 0.0559 |
| Gravity baseline | 0.6504 | 0.0109 | 0.0423 | 0.0091 | 0.0188 | 0.0068 | 0.0155 |
| GraphSAGE | 0.6178 | 0.0095 | 0.0593 | 0.0139 | 0.0163 | 0.0056 | 0.0166 |
| GAT | 0.5735 | 0.0075 | 0.0547 | 0.0139 | 0.0371 | 0.0090 | 0.0226 |
| GCN | 0.5210 | 0.0067 | 0.0353 | 0.0072 | 0.0134 | 0.0030 | 0.0091 |

The held-out candidate universe has a positive rate of approximately **0.64%**,
so random-ranking average precision is approximately `0.0064`. Density reaches
about 2.9x that baseline, HGT 2.4x, and the learned reranker 1.9x. The density
baseline remains the strongest method overall. The learned reranker is
competitive near the top of the list—second on Precision@10 and NDCG@10—but
does not yet outperform density.

These values cover 209 country queries. Because positive prevalence and query
coverage changed from the preceding version, interpret raw average-precision
changes together with AP lift and query-based top-K metrics. See
[Comparison with the preceding implementation](#comparison-with-the-preceding-implementation).

## Quick Start

### Prerequisites

- Python 3.11 (recommended on Windows)
- Git

### 1. Clone and install

```bash
git clone https://github.com/DamjanSandev/wbs_mk_trade.git
cd wbs_mk_trade/wbs_mk_trade-develop
python -m pip install -e .
```

For an isolated PowerShell environment on Windows:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows is intentionally pinned to PyTorch `<2.9` because newer releases can
fail while loading `torch\lib\c10.dll` in this dependency combination. If that
DLL error appears, confirm that PyCharm and the terminal both use
`.venv311\Scripts\python.exe`, then reinstall the project dependencies in that
environment.

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
# 1. Download and clean data (needs internet + Comtrade API key)
python scripts/00_download_data.py

# 2. Compute complexity metrics and leakage-safe cutoff proximities
python scripts/01_build_complexity.py

# 3. Build the exploratory graph and classical baselines
python scripts/02_build_graph.py

# 4. Train GNNs and evaluate GNNs, density, gravity, and the reranker
python scripts/04_train_models.py

# Optional robust evaluation: rolling cutoffs x configured random seeds
python scripts/04_train_models.py --rolling-backtest

# 5. Generate Task A/Task B rankings, calibrated scores, and explanations
python scripts/05_generate_opportunities.py

# Launch dashboard
python src/mktrade/viz/app.py
```

Phase 4 and Phase 5 construct their leakage-safe historical graph snapshots
directly from the processed tables. You do **not** need to rerun
`02_build_graph.py` after changing only model, target, reranker, or dashboard
code. See [What needs to be rerun?](#what-needs-to-be-rerun).

## Project Structure

```
wbs_mk_trade-develop/
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
│   ├── rolling_backtests.csv
│   ├── rolling_backtest_summary.csv
│   ├── task_a_reranker_validation_all_countries.csv
│   ├── task_a_reranker_test_all_countries.csv
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

The older `01_clean_data.py` and `02_compute_complexity.py` files are unfinished
placeholders, while `03_build_graph.py` is a deprecated compatibility message.
Use the five commands shown in [Reproduce from scratch](#4-reproduce-from-scratch-full-pipeline).

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

For Task A, `gnn_score` and the final `score` have deliberately different
meanings:

| Field | Meaning |
|---|---|
| `gnn_logit` | Raw GNN link score used for ranking without sigmoid saturation |
| `gnn_score` | Sigmoid-transformed GNN signal; an input to the reranker, not a calibrated success probability |
| `ranking_score` | Unbounded pairwise-reranker decision value used to order candidates |
| `score` / `ensemble_score` | Validation-calibrated probability shown by the Task A dashboard |
| `probability_lower` / `probability_upper` | 5th and 95th percentiles across bootstrap reranker fits |
| `average_rank` / `rank_std` / `rank_stability` | Recommendation stability across bootstrap fits |

The old fixed `60% GNN + 30% density + 10% classical` formula is no longer
used. The dashboard now labels the final probability separately from the raw
GNN signal.

Task B is implemented as an origin-product-destination ranking problem. The
destination GNN compatibility score is combined with destination import value
and growth, supplier concentration, existing MKD route history, total bilateral
trade, distance, language, FTA status, GDP, population, MKD supply capacity,
and an optional tariff table (`data/external/tariffs.parquet`). Missing market
data is represented explicitly rather than silently interpreted as zero demand.

## Knowledge Graph

The graph has **3 node types** and **7 edge types**:

- **Nodes:** Country (230), Product (1,242), HS Section (21)
- **Edges:** `exports`, `proximity`, `in_section`, `trades_with`, `neighbor_of` + reverse edges
- **Country features (14-dim):** ECI, diversity, GDP, population, landlocked, EU, CEFTA, region
- **Product features (23-dim):** PCI, ubiquity, section one-hot

## Models & Baselines

**GNN configurations:** GAT, HGT, GraphSAGE, GCN

The current heterogeneous `GCN` compatibility configuration uses the same
SAGEConv-based encoder family as GraphSAGE because PyG `GCNConv` does not support
the project’s bipartite edge types. Treat its row as an experimental
configuration, not as an independent native GCN implementation.

**Baselines:** Economic complexity density, PPML gravity model, Adamic-Adar, Jaccard coefficient, Common Neighbors

**Second stage:** Pairwise logistic reranker using GNN, complexity, demand,
persistence, supply, gravity, and feasibility features when available

**Evaluation:** Temporal holdout plus optional rolling-origin backtests. Every
unseen product is ranked separately for each country. Headline metrics are
query-based MRR, Precision@10, Recall@10, MAP@10, and NDCG@10; ROC-AUC and
global average precision remain diagnostic metrics.

## Ranking redesign

The production ranking path now follows these rules:

1. A successful export must have annual value of at least USD 100,000, RCA >= 1,
   and remain successful for at least two consecutive years. These thresholds
   are configurable in `configs/train.yaml`.
2. Training examples are historical transitions: candidates are absent at a
   cutoff and positive only if they become sustained successes in the following
   outcome window. Each positive is trained against a configurable mixture of
   high-density hard negatives and random negatives.
3. Validation (2019-2020) and test (2021-2022) outcome windows do not overlap.
   Both contain the complete filtered candidate universe, and validation
   positives are removed from later test candidates.
4. The GNN uses a weighted binary-classification plus RankNet pairwise loss.
   Checkpoint selection uses the geometric composite of Average Precision, MRR,
   and NDCG@10. Raw logits are retained for ranking to avoid saturated ties.
5. Every candidate is scored before top-K selection. GNN, density, PCI,
   complexity gain, demand, persistence, gravity, and market feasibility signals
   can therefore change the final order.
6. A pairwise reranker is fitted across every country query in Phase 4, rather
   than only the five MKD positives available in one report-generation cohort.
   It uses transformed economic features and within-country percentile ranks.
   The former 60/30/10 manual blend has been removed.
7. Because a pairwise decision value is a relative score rather than an
   item-level probability, a separate logistic calibration layer maps each
   within-query score percentile to validation-era success probability.
   Bootstrap fits report `probability_lower`, `probability_upper`,
   `average_rank`, `rank_std`, and `rank_stability` for every recommendation.
8. Product proximity matrices are generated at historical cutoffs and the graph
   is forbidden from falling forward to a future proximity matrix. GAT consumes
   standardised edge attributes and now uses an MLP decoder instead of the
   saturating dot-product decoder.

If validation data contains only one class, report generation emits a warning
and uses an equal-percentile fallback over non-constant signals. It does not
label that fallback as a calibrated probability. Phase 5 also detects a saved
pairwise reranker created before probability calibration was introduced and
refits it from the cached all-country validation candidates when available.

### Comparison with the preceding implementation

The preceding implementation already used the sustained-success target and
complete candidate universe, but it predated the current non-overlapping split,
ranking-aware training, all-country pairwise reranker, gravity baseline, GAT
decoder update, and calibrated Task A score.

| Model | ROC-AUC, previous → current | AP, previous → current | MRR, previous → current | Precision@10, previous → current | NDCG@10, previous → current |
|---|---:|---:|---:|---:|---:|
| Density | 0.7036 → **0.7289** | 0.0117 → **0.0187** | 0.1361 → **0.1935** | 0.0311 → **0.0550** | 0.0718 → **0.0890** |
| HGT | 0.7115 → **0.7204** | 0.0085 → **0.0152** | 0.0882 → **0.1231** | 0.0176 → **0.0292** | 0.0498 → **0.0500** |
| GraphSAGE | **0.6429** → 0.6178 | 0.0060 → **0.0095** | 0.0284 → **0.0593** | 0.0062 → **0.0139** | 0.0118 → **0.0166** |
| GAT | 0.5684 → **0.5735** | 0.0047 → **0.0075** | 0.0338 → **0.0547** | 0.0073 → **0.0139** | 0.0110 → **0.0226** |
| GCN | **0.6512** → 0.5210 | 0.0064 → **0.0067** | **0.0524** → 0.0353 | **0.0124** → 0.0072 | **0.0222** → 0.0091 |
| Learned reranker | — → 0.5979 | — → 0.0122 | — → 0.1258 | — → 0.0426 | — → 0.0559 |
| Gravity baseline | — → 0.6504 | — → 0.0109 | — → 0.0423 | — → 0.0091 | — → 0.0155 |

Raw AP increased for every comparable model, but the positive rate also rose
from approximately 0.379% to 0.64%, and the number of evaluated queries rose
from 193 to 209. The increase in raw AP is therefore not entirely attributable
to better modelling. Relative AP lift improved modestly for HGT, while density,
GraphSAGE, and GAT are slightly lower relative to their new random baseline.

The clearer improvement is at the top of the ranking: density, HGT, GraphSAGE,
and GAT all improve MRR and Precision@10. This is the expected direction from
hard-negative training and the RankNet component, which optimise early ranking
quality more directly than global ROC-AUC. GCN is the main regression and should
not be selected without further investigation. The learned reranker is promising
near the top but currently remains below density.

For historical context, the original exploratory implementation evaluated one
random negative for every positive, producing a roughly balanced test set and
very large AP values. Those values are not comparable with either full-candidate
version because real deployment has far fewer successful new exports than
unsuccessful candidates.

### Rolling-origin robustness

The optional robustness run uses training cutoffs `[2016, 2017, 2018, 2019]`
and seeds `[13, 42, 73]`. With the two-year persistence target, the windows are:

| Training data | Validation outcome | Test outcome |
|---|---|---|
| Through 2016 | 2017-2018 | 2019-2020 |
| Through 2017 | 2018-2019 | 2020-2021 |
| Through 2018 | 2019-2020 | 2021-2022 |
| Through 2019 | 2020-2021 | 2022-2023 |

The checked-in rolling report currently contains **GAT only**: 4 windows × 3
seeds = 12 runs.

| GAT metric | Mean | Standard deviation |
|---|---:|---:|
| ROC-AUC | 0.5272 | 0.0553 |
| Average precision | 0.0041 | 0.0007 |
| MRR | 0.0321 | 0.0069 |
| Precision@10 | 0.0067 | 0.0025 |
| NDCG@10 | 0.0115 | 0.0034 |

Run `python scripts/04_train_models.py --rolling-backtest` without `--model` to
repeat the configured windows for all four GNNs. This is computationally
expensive. Density and gravity are deterministic for a given window and do not
need repeated random seeds; a complete robustness comparison should still
recompute them for every time window. The current rolling helper covers the GNN
models, so the checked-in rolling summary should not be read as a complete
seven-model comparison.

## Dashboard

Interactive dashboard built with Dash (Plotly) supporting **English and Macedonian**:

- Product diversification chart using the calibrated final score
- Task A detail table showing final probability, GNN signal, density, and PCI separately
- Learned ensemble rankings with corrected score definitions
- Market expansion grouped by product or country
- Model comparison with ROC-AUC, Average Precision, top-K ranking metrics, and AP lift over random
- Gradient-based feature importance explanations
- Glossary of HS codes, trade metrics, and ML terminology

```bash
python src/mktrade/viz/app.py
# → http://localhost:8050
```

## Key Technical Notes

- **Success target:** RCA >= 1, value >= USD 100,000, active for 2 consecutive years
- **Temporal split:** Feature cutoff 2018, validation outcomes 2019–2020, test
  outcomes 2021–2022; the outcome windows do not overlap
- **Training target:** future sustained link formation, with 10 negatives per
  positive and 70% hard-negative mining
- **Candidate evaluation:** all unseen products per country, with macro-averaged query metrics
- **Training loss:** 50% weighted BCE and 50% pairwise RankNet loss
- **Model selection:** geometric composite of validation AP, MRR, and NDCG@10
- **Robustness:** optional rolling cutoffs `[2016, 2017, 2018, 2019]` and seeds `[13, 42, 73]`
- **Proximity sparsification:** HGT uses top-50 neighbors per product (prevents OOM); GAT/SAGE/GCN use full proximity edges
- **Graceful degradation:** Neo4j optional (NetworkX fallback), Comtrade key optional (cached data fallback)
- **Reranker:** all-country pairwise model with log transforms, query-relative
  ranks, hard pairs, candidate-level probability calibration, and bootstrap stability
- **Market demand coverage:** Comtrade downloads include configured destination reporters in addition to MKD
- **Explanations:** Gradient-based attribution (input × gradient) for feature importance

## What needs to be rerun?

| Change | Required action |
|---|---|
| Dashboard layout or labels only | Restart `python src/mktrade/viz/app.py` |
| Ranking/reranker code only, with compatible checkpoints | Run `python scripts/05_generate_opportunities.py` |
| Model architecture, loss, success target, split, or training configuration | Run Phase 4, then Phase 5 |
| Complexity calculation or source export data | Run complexity, training, and report generation again |
| Raw data sources or Comtrade market coverage | Run the full pipeline |
| Neo4j/exploratory graph or classical-baseline construction | Run `python scripts/02_build_graph.py` |

Phase 5 detects pairwise reranker files created before candidate-level
probability calibration. When the cached all-country validation candidates are
available, it refits and saves the reranker instead of emitting an all-zero
ensemble score. Otherwise it attempts the smaller MKD validation fallback and,
if that is not learnable, uses the explicitly labelled percentile fallback.

After regenerating a CSV report, restart the dashboard because report files are
loaded when the Dash process starts.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
