"""Deprecated — use scripts/02_build_graph.py instead."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    print("This script has been replaced by scripts/02_build_graph.py")
    print("Usage: python scripts/02_build_graph.py [--year 2022] [--skip-neo4j]")


if __name__ == "__main__":
    main()
