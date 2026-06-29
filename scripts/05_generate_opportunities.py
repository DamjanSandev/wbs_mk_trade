"""Phase 5 — Generate the opportunity report for North Macedonia.

Loads the best model checkpoint, scores all candidate links, ranks them,
runs GNNExplainer, and writes the final report to reports/.

Usage:
    python scripts/05_generate_opportunities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mktrade.config import load_data_config, load_train_config


def main() -> None:
    data_cfg = load_data_config()
    train_cfg = load_train_config()
    # TODO: implement in Phase 5
    raise NotImplementedError("Phase 5: implement opportunity generation")


if __name__ == "__main__":
    main()
