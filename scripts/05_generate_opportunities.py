"""Phase 5 — Generate the opportunity report for North Macedonia.

Loads the best model checkpoint, scores all candidate links, ranks them,
runs gradient-based explanations, and writes reports to reports/.

Usage:
    python scripts/05_generate_opportunities.py
    python scripts/05_generate_opportunities.py --model gat --top-k 100
    python scripts/05_generate_opportunities.py --skip-task-b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from loguru import logger

from mktrade.config import load_data_config, load_train_config, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MKD opportunity report")
    parser.add_argument("--model", type=str, default="gat",
                        help="Model to use (default: gat, the best performer)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of top opportunities to report")
    parser.add_argument("--explain-top-n", type=int, default=15,
                        help="Number of top opportunities to explain")
    parser.add_argument("--skip-task-b", action="store_true",
                        help="Skip Task B (market expansion) — faster")
    parser.add_argument("--skip-ensemble", action="store_true",
                        help="Skip ensemble ranking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_data_config()
    train_cfg = load_train_config()
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"PHASE 5 — Opportunity Report Generation")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {args.model}, Top-K: {args.top_k}, Country: {data_cfg.focus_country}")

    # ── 1. Load data ──
    logger.info("\nLoading data...")
    atlas_df = pd.read_parquet(data_cfg.processed_dir / "atlas_exports.parquet")
    country_complexity = pd.read_parquet(data_cfg.processed_dir / "country_complexity.parquet")
    product_complexity = pd.read_parquet(data_cfg.processed_dir / "product_complexity.parquet")
    proximity_df = pd.read_parquet(data_cfg.processed_dir / "proximity_matrix.parquet")

    complexity_df = None
    cx_path = data_cfg.processed_dir / "complexity.parquet"
    if cx_path.exists():
        complexity_df = pd.read_parquet(cx_path)

    bilateral_df = None
    bil_path = data_cfg.processed_dir / "bilateral_flows.parquet"
    if bil_path.exists():
        bilateral_df = pd.read_parquet(bil_path)

    gravity_df = None
    grav_path = data_cfg.external_dir / "gravity_clean.parquet"
    if grav_path.exists():
        gravity_df = pd.read_parquet(grav_path)

    wdi_df = None
    wdi_path = data_cfg.external_dir / "wdi_indicators.parquet"
    if wdi_path.exists():
        wdi_df = pd.read_parquet(wdi_path)

    classical_df = None
    cl_path = data_cfg.processed_dir / "classical_baselines.parquet"
    if cl_path.exists():
        classical_df = pd.read_parquet(cl_path)

    # ── 2. Build graph for inference ──
    # Build the training-era graph (same topology as Phase 4) so the
    # checkpoint state_dict keys match. Candidates = products not exported
    # in the training window.
    logger.info("\nBuilding training-era graph...")
    from mktrade.train.splits import _build_snapshot_data

    train_years = list(range(atlas_df["year"].min(), train_cfg.train_end_year + 1))
    train_data = _build_snapshot_data(
        atlas_df, country_complexity, product_complexity,
        proximity_df, gravity_df, bilateral_df, wdi_df,
        years=train_years, cefta_members=set(data_cfg.cefta_members),
    )

    # ── 3. Load the trained model ──
    # Build model architecture from training data (must match checkpoint structure)
    logger.info(f"\nLoading model checkpoint: {args.model}")
    from mktrade.models.link_predictor import build_model

    model = build_model(args.model, train_data)
    ckpt_path = PROJECT_ROOT / "models" / f"{args.model}_best.pt"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Run Phase 4 first: python scripts/04_train_models.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
                f"val_AP={ckpt.get('val_ap', 0):.4f}")

    # ── 4. Task A: Product diversification ──
    logger.info(f"\n{'─'*40}")
    logger.info("Task A: Product Diversification")
    logger.info(f"{'─'*40}")

    from mktrade.opportunities.ranker import (
        rank_product_opportunities,
        rank_market_opportunities,
        ensemble_rankings,
    )

    # Use train_data for both encoding and candidate selection:
    # - Encoding: model was trained on this graph topology
    # - Candidates: products MKD doesn't export as of train_end_year
    task_a_df = rank_product_opportunities(
        model=model,
        data=train_data,
        country=data_cfg.focus_country,
        top_k=args.top_k,
        complexity_df=complexity_df,
    )

    task_a_path = reports_dir / "task_a_opportunities.csv"
    task_a_df.to_csv(task_a_path, index=False, float_format="%.6f")
    logger.info(f"  Saved {len(task_a_df)} opportunities to {task_a_path}")

    # Print top 20
    logger.info(f"\n  Top 20 product opportunities for {data_cfg.focus_country}:")
    for _, row in task_a_df.head(20).iterrows():
        density_str = f"density={row['density']:.3f}" if pd.notna(row.get('density')) else ""
        logger.info(f"    #{int(row['rank']):3d}  HS {row['hs4']}  "
                    f"score={row['score']:.4f}  {density_str}")

    # ── 5. Ensemble ranking ──
    if not args.skip_ensemble:
        logger.info(f"\n{'─'*40}")
        logger.info("Ensemble Rankings")
        logger.info(f"{'─'*40}")

        density_for_ensemble = None
        if complexity_df is not None:
            mkd_cx = complexity_df[complexity_df["iso3"] == data_cfg.focus_country]
            if not mkd_cx.empty:
                density_for_ensemble = mkd_cx[["hs4", "density"]].copy()

        ensemble_df = ensemble_rankings(
            gnn_df=task_a_df,
            density_df=density_for_ensemble,
            classical_df=classical_df,
        )

        ensemble_path = reports_dir / "ensemble_rankings.csv"
        ensemble_df.to_csv(ensemble_path, index=False, float_format="%.6f")
        logger.info(f"  Saved ensemble rankings to {ensemble_path}")

    # ── 6. Task B: Market expansion ──
    if not args.skip_task_b:
        logger.info(f"\n{'─'*40}")
        logger.info("Task B: Market Expansion")
        logger.info(f"{'─'*40}")

        task_b_df = rank_market_opportunities(
            model=model,
            data=train_data,
            country=data_cfg.focus_country,
            top_k=args.top_k,
            bilateral_df=bilateral_df,
        )

        task_b_path = reports_dir / "task_b_opportunities.csv"
        task_b_df.to_csv(task_b_path, index=False, float_format="%.6f")
        logger.info(f"  Saved {len(task_b_df)} market opportunities to {task_b_path}")

        logger.info(f"\n  Top 20 market expansion opportunities:")
        for _, row in task_b_df.head(20).iterrows():
            logger.info(f"    #{int(row['rank']):3d}  HS {row['hs4']} -> {row['partner_iso3']}  "
                        f"score={row['score']:.4f}")

    # ── 7. Explanations ──
    logger.info(f"\n{'─'*40}")
    logger.info("Generating explanations...")
    logger.info(f"{'─'*40}")

    from mktrade.opportunities.explainer import explain_top_opportunities

    explanations = explain_top_opportunities(
        model=model,
        data=train_data,
        opportunities_df=task_a_df,
        country=data_cfg.focus_country,
        top_n=args.explain_top_n,
    )

    explanations_path = reports_dir / "explanations.json"
    with open(explanations_path, "w") as f:
        json.dump(explanations, f, indent=2, default=str)
    logger.info(f"  Saved {len(explanations)} explanations to {explanations_path}")

    # Print first explanation
    if explanations:
        exp = explanations[0]
        logger.info(f"\n  Example explanation (#{exp.get('rank', 1)}: HS {exp['hs4']}):")
        logger.info(f"    Score: {exp['score']:.4f}")
        top_src = sorted(exp["src_importance"].items(), key=lambda x: -x[1])[:5]
        logger.info(f"    Top country features: {', '.join(f'{k}={v:.3f}' for k, v in top_src)}")
        top_dst = sorted(exp["dst_importance"].items(), key=lambda x: -x[1])[:5]
        logger.info(f"    Top product features: {', '.join(f'{k}={v:.3f}' for k, v in top_dst)}")

    # ── 8. Summary ──
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 5 COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Reports written to: {reports_dir}/")
    logger.info(f"  - task_a_opportunities.csv  ({len(task_a_df)} products)")
    if not args.skip_task_b:
        logger.info(f"  - task_b_opportunities.csv  ({len(task_b_df)} markets)")
    if not args.skip_ensemble:
        logger.info(f"  - ensemble_rankings.csv")
    logger.info(f"  - explanations.json         ({len(explanations)} explanations)")
    logger.info(f"\nLaunch dashboard: streamlit run src/mktrade/viz/app.py")


if __name__ == "__main__":
    main()
