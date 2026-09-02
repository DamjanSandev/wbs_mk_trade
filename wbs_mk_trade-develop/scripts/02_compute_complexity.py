"""Phase 2 — Compute economic-complexity metrics.

Runs the ecomplexity pipeline on cleaned Atlas data to produce
RCA, ECI, PCI, proximity, density, COI, COG. Writes results to
data/processed/.

Usage:
    python scripts/02_compute_complexity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mktrade.config import load_data_config


def main() -> None:
    cfg = load_data_config()
    # TODO: implement in Phase 2
    raise NotImplementedError("Phase 2: implement complexity computation")


if __name__ == "__main__":
    main()
