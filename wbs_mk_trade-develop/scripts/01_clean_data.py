"""Phase 1b — Clean and harmonise raw data.

Reads raw downloads, standardises country codes (ISO3/M49), filters
HS level and year range, and writes clean parquet files to data/interim/.

Usage:
    python scripts/01_clean_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mktrade.config import load_data_config


def main() -> None:
    cfg = load_data_config()
    # TODO: implement in Phase 1
    raise NotImplementedError("Phase 1: implement data cleaning")


if __name__ == "__main__":
    main()
