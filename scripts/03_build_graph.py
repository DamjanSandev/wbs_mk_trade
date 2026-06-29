"""Phase 3 — Build the knowledge graph.

1. Builds a NetworkX in-memory graph (always).
2. Loads into Neo4j + runs GDS embeddings (if Neo4j is available).
3. Converts to PyG HeteroData and saves to data/processed/.

Usage:
    python scripts/03_build_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mktrade.config import load_data_config, load_neo4j_config


def main() -> None:
    data_cfg = load_data_config()
    neo4j_cfg = load_neo4j_config()
    # TODO: implement in Phase 3
    raise NotImplementedError("Phase 3: implement graph construction")


if __name__ == "__main__":
    main()
