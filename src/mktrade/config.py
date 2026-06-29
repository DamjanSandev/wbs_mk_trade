"""Centralised configuration loaded from YAML + environment variables.

Uses pydantic-settings so every value can be overridden by an env var.
All paths are resolved relative to PROJECT_ROOT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/wbs_mk_trade

# Load .env from project root (if present)
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml(name: str) -> dict[str, Any]:
    """Load a YAML config file from the configs/ directory."""
    path = PROJECT_ROOT / "configs" / name
    with open(path) as f:
        return yaml.safe_load(f) or {}


class DataConfig(BaseSettings):
    """Data-acquisition settings (mirrors configs/data.yaml)."""

    hs_level: int = 4
    year_start: int = 2000
    year_end: int = 2022
    focus_country: str = "MKD"
    comtrade_api_key: str = Field(default="", alias="COMTRADE_KEY")
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    external_dir: Path = PROJECT_ROOT / "data" / "external"

    # Comtrade-specific
    comtrade_year_start: int = 2015
    comtrade_year_end: int = 2022
    comtrade_reporters_m49: list[str] = Field(default_factory=lambda: ["807"])
    comtrade_flow_codes: list[str] = Field(default_factory=lambda: ["X", "M"])
    comtrade_max_records: int = 250000

    # Country sets
    cefta_members: list[str] = Field(
        default_factory=lambda: ["ALB", "BIH", "MKD", "MDA", "MNE", "SRB", "XKX"]
    )
    western_balkans: list[str] = Field(
        default_factory=lambda: ["ALB", "BIH", "MKD", "MNE", "SRB", "XKX"]
    )

    model_config = {"env_prefix": "", "extra": "ignore", "populate_by_name": True}


class Neo4jConfig(BaseSettings):
    """Neo4j connection settings."""

    uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    user: str = Field(default="neo4j", alias="NEO4J_USER")
    password: str = Field(default="changeme", alias="NEO4J_PASSWORD")
    database: str = "neo4j"
    import_batch_size: int = 5000

    model_config = {"env_prefix": "", "extra": "ignore", "populate_by_name": True}


class TrainConfig(BaseSettings):
    """Training hyper-parameters (mirrors configs/train.yaml)."""

    seed: int = 42
    device: str = "auto"
    task: str = "task_a"
    epochs: int = 200
    patience: int = 20
    lr: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 1024
    split_strategy: str = "temporal"
    train_end_year: int = 2019
    val_year: int = 2020
    test_years: list[int] = Field(default_factory=lambda: [2021, 2022])
    mlflow_experiment: str = "mk-trade-link-pred"
    mlflow_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")

    model_config = {"env_prefix": "", "extra": "ignore", "populate_by_name": True}


def load_data_config() -> DataConfig:
    """Build DataConfig from YAML defaults + env overrides."""
    raw = _load_yaml("data.yaml")
    atlas = raw.get("atlas", {})
    comtrade = raw.get("comtrade", {})
    ct_years = comtrade.get("year_range", [2015, 2022])
    return DataConfig(
        hs_level=atlas.get("hs_level", 4),
        year_start=atlas.get("year_range", [2000])[0],
        year_end=atlas.get("year_range", [0, 2022])[1],
        focus_country=raw.get("focus_country", "MKD"),
        comtrade_year_start=ct_years[0],
        comtrade_year_end=ct_years[1],
        comtrade_reporters_m49=comtrade.get("reporters", ["807"]),
        comtrade_flow_codes=comtrade.get("flow_codes", ["X", "M"]),
        comtrade_max_records=comtrade.get("max_records_per_call", 250000),
        cefta_members=raw.get("cefta_members", []),
        western_balkans=raw.get("western_balkans", []),
    )


def load_neo4j_config() -> Neo4jConfig:
    """Build Neo4jConfig from YAML defaults + env overrides."""
    raw = _load_yaml("graph.yaml").get("neo4j", {})
    return Neo4jConfig(
        database=raw.get("database", "neo4j"),
        import_batch_size=raw.get("import_batch_size", 5000),
    )


def load_train_config() -> TrainConfig:
    """Build TrainConfig from YAML defaults + env overrides."""
    raw = _load_yaml("train.yaml")
    split = raw.get("split", {})
    temporal = split.get("temporal", {})
    return TrainConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        task=raw.get("task", "task_a"),
        epochs=raw.get("epochs", 200),
        patience=raw.get("patience", 20),
        lr=raw.get("optimizer", {}).get("lr", 0.001),
        weight_decay=raw.get("optimizer", {}).get("weight_decay", 0.0001),
        batch_size=raw.get("loader", {}).get("batch_size", 1024),
        split_strategy=split.get("strategy", "temporal"),
        train_end_year=temporal.get("train_end_year", 2019),
        val_year=temporal.get("val_year", 2020),
        test_years=temporal.get("test_years", [2021, 2022]),
        mlflow_experiment=raw.get("mlflow", {}).get("experiment_name", "mk-trade-link-pred"),
    )
