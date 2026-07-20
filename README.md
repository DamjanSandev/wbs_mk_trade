# MK Trade KG-GNN

Knowledge-graph and Graph Neural Network link prediction for discovering
export-expansion opportunities for North Macedonia.

## Quick start

```bash
# 1. Clone & install
git clone <repo-url> && cd wbs_mk_trade
uv pip install -e ".[dev]"
cp .env.example .env          # edit with your Comtrade key + Neo4j password

# 2. Start infrastructure
make neo4j-up                 # Neo4j 5.x + GDS (Docker required)

# 3. Run the pipeline
make data                     # Phase 1: download + clean
make complexity               # Phase 2: ecomplexity metrics
make graph                    # Phase 3: build KG (Neo4j + PyG)
make train                    # Phase 4: train GNNs + baselines
make opportunities            # Phase 5: generate MKD report

# 4. Explore results
make dashboard                # Streamlit app on localhost:8501
```

## Project structure

```
wbs_mk_trade/
├── configs/                  # YAML configuration
│   ├── data.yaml             #   data sources, paths, year ranges
│   ├── graph.yaml            #   KG schema, Neo4j, embeddings
│   ├── train.yaml            #   training hyper-parameters
│   └── model/                #   per-architecture configs
│       ├── graphsage.yaml
│       ├── gat.yaml
│       ├── gcn.yaml
│       ├── rgcn.yaml
│       ├── hgt.yaml
│       └── vgae.yaml
├── data/
│   ├── raw/                  # cached downloads (git-ignored)
│   ├── interim/              # cleaned parquets
│   ├── processed/            # final tensors / HeteroData
│   └── external/             # CEPII, WDI
├── src/mktrade/              # main Python package
│   ├── config.py             # pydantic-settings config loader
│   ├── data/                 # download, clean, ISO/M49 mapping
│   ├── complexity/           # ecomplexity wrapper + baselines
│   ├── graph/                # NetworkX, Neo4j, PyG HeteroData
│   ├── models/               # GNN encoders + decoders
│   ├── train/                # splits, training loop
│   ├── eval/                 # metrics, comparison, gravity baseline
│   ├── opportunities/        # ranking + GNNExplainer
│   └── viz/                  # Streamlit app + Plotly charts
├── scripts/                  # thin CLI entry points (00–05)
├── tests/                    # pytest suite
├── notebooks/                # EDA / prototyping
├── reports/                  # figures + thesis drafts
├── models/                   # saved checkpoints
├── pyproject.toml            # deps, ruff, black, mypy, pytest
├── Makefile                  # pipeline orchestration
├── docker-compose.yml        # Neo4j + GDS + MLflow
├── .env.example              # template environment vars
└── .pre-commit-config.yaml   # ruff + black + mypy hooks
```

## Two prediction tasks

| Task | Link type | Question |
|------|-----------|----------|
| **A** (product diversification) | Country → Product | What new products could MKD export? |
| **B** (market expansion) | Country → Product → Country | Which new markets for MKD's existing products? |

## Models & baselines

**GNN encoders:** GraphSAGE, GAT, GCN, R-GCN, HGT (+ GAE/VGAE variants)
**Decoders:** dot-product, DistMult, MLP
**Baselines:** complexity density/COG, PPML gravity, Adamic-Adar/Jaccard/CN, Neo4j GDS LP

## Evaluation

Metrics: ROC-AUC, AP, Precision@K, Recall@K, MRR, Hits@K
Headline experiment: **temporal holdout** — do top-ranked predictions appear in future years?

## Phase roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold repo, configs, stubs | Done |
| 1 | Data acquisition & cleaning (Atlas, Comtrade, CEPII, WDI) | Done |
| 2 | Economic complexity (ecomplexity pipeline) | Done |
| 3 | Knowledge graph (Neo4j + NetworkX + PyG HeteroData) | Done |
| 4 | GNN training, baselines, evaluation, ablations | Done |
| 5 | Opportunity report, explanations, Streamlit dashboard | Done |

## Requirements

- Python 3.11+
- Docker (for Neo4j + MLflow)
- UN Comtrade API key (free tier)
