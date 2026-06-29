"""Tests for configuration loading."""

from __future__ import annotations

from mktrade.config import load_data_config, load_neo4j_config, load_train_config


def test_load_data_config() -> None:
    cfg = load_data_config()
    assert cfg.hs_level in (2, 4, 6)
    assert cfg.year_start < cfg.year_end
    assert cfg.focus_country == "MKD"


def test_load_neo4j_config() -> None:
    cfg = load_neo4j_config()
    assert cfg.database == "neo4j"
    assert cfg.import_batch_size > 0


def test_load_train_config() -> None:
    cfg = load_train_config()
    assert cfg.seed == 42
    assert cfg.split_strategy in ("temporal", "random")
    assert cfg.train_end_year < cfg.val_year
