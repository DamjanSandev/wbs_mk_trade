.PHONY: help install lint test neo4j-up neo4j-down data complexity graph train eval opportunities dashboard clean

PYTHON ?= python
UV     ?= uv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────
install:  ## Install project + dev deps with uv
	$(UV) pip install -e ".[dev]"

lint:  ## Run ruff + black + mypy
	$(UV) run ruff check src/ tests/ scripts/
	$(UV) run black --check src/ tests/ scripts/
	$(UV) run mypy src/

test:  ## Run pytest
	$(UV) run pytest tests/ -v --tb=short

# ── Infrastructure ───────────────────────────────────────
neo4j-up:  ## Start Neo4j + GDS via docker-compose
	docker compose up -d neo4j

neo4j-down:  ## Stop Neo4j
	docker compose down

# ── Pipeline steps ───────────────────────────────────────
data:  ## Phase 1 — download & clean data
	$(PYTHON) scripts/00_download_data.py
	$(PYTHON) scripts/01_clean_data.py

complexity:  ## Phase 2 — compute economic complexity
	$(PYTHON) scripts/02_compute_complexity.py

graph:  ## Phase 3 — build knowledge graph
	$(PYTHON) scripts/03_build_graph.py

train:  ## Phase 4 — train GNN models
	$(PYTHON) scripts/04_train_models.py

eval:  ## Phase 4b — evaluate & compare
	$(PYTHON) scripts/04_train_models.py --eval-only

opportunities:  ## Phase 5 — generate opportunity report
	$(PYTHON) scripts/05_generate_opportunities.py

dashboard:  ## Launch Streamlit dashboard
	$(UV) run streamlit run src/mktrade/viz/app.py

# ── Housekeeping ─────────────────────────────────────────
clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
