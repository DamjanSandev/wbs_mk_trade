"""Phase 4 — Train GNN models and run baselines.

Trains all configured encoder-decoder combinations on the temporal split,
evaluates on the holdout test set, compares against baselines.

Usage:
    python scripts/04_train_models.py                   # train all + eval
    python scripts/04_train_models.py --model graphsage # single model
    python scripts/04_train_models.py --eval-only       # eval saved models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from loguru import logger

from mktrade.config import load_data_config, load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN link-prediction models")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate only")
    parser.add_argument("--model", type=str, default=None,
                        help="Train a single model (graphsage|gat|gcn|hgt). Default: all")
    parser.add_argument("--task", type=str, default=None, help="task_a | task_b")
    parser.add_argument("--year", type=int, default=2022, help="Latest year for data")
    return parser.parse_args()


ALL_MODELS = ["graphsage", "gat", "gcn", "hgt"]


def main() -> None:
    args = parse_args()
    train_cfg = load_train_config()
    data_cfg = load_data_config()

    if args.task:
        train_cfg.task = args.task

    models_to_train = [args.model] if args.model else ALL_MODELS

    logger.info(f"{'='*60}")
    logger.info(f"PHASE 4 — GNN Training & Evaluation")
    logger.info(f"{'='*60}")
    logger.info(f"Models: {models_to_train}")
    logger.info(f"Split: temporal (train<={train_cfg.train_end_year}, "
                f"val={train_cfg.val_year}, test={train_cfg.test_years})")

    # ── 1. Load data ──
    logger.info("\nLoading data...")
    atlas_df = pd.read_parquet(data_cfg.processed_dir / "atlas_exports.parquet")
    country_complexity = pd.read_parquet(data_cfg.processed_dir / "country_complexity.parquet")
    product_complexity = pd.read_parquet(data_cfg.processed_dir / "product_complexity.parquet")
    proximity_df = pd.read_parquet(data_cfg.processed_dir / "proximity_matrix.parquet")

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

    # ── 2. Temporal split ──
    logger.info("\nBuilding temporal split...")
    from mktrade.train.splits import temporal_split

    train_data, val_data, test_data = temporal_split(
        exports_df=atlas_df,
        country_features=country_complexity,
        product_features=product_complexity,
        cfg=train_cfg,
        proximity_df=proximity_df,
        gravity_df=gravity_df,
        bilateral_df=bilateral_df,
        wdi_df=wdi_df,
        cefta_members=set(data_cfg.cefta_members),
    )

    # ── 3. Train models ──
    from mktrade.models.link_predictor import build_model
    from mktrade.train.trainer import Trainer

    all_results: dict[str, dict[str, float]] = {}
    checkpoint_dir = Path("models")

    for model_name in models_to_train:
        logger.info(f"\n{'─'*40}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'─'*40}")

        try:
            model = build_model(model_name, train_data)

            if not args.eval_only:
                trainer = Trainer(
                    model=model,
                    train_data=train_data,
                    val_data=val_data,
                    cfg=train_cfg,
                    model_name=model_name,
                    checkpoint_dir=checkpoint_dir,
                )
                best_path = trainer.fit()
                logger.info(f"  Best checkpoint: {best_path}")

                # Evaluate on test set
                test_metrics = trainer.evaluate(test_data)
            else:
                # Load saved model
                ckpt_path = checkpoint_dir / f"{model_name}_best.pt"
                if not ckpt_path.exists():
                    logger.warning(f"  No checkpoint found at {ckpt_path}, skipping")
                    continue
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(device)
                model.eval()

                from mktrade.eval.metrics import compute_all_metrics
                import torch.nn.functional as F

                test_data_dev = test_data.to(device)
                edge_type = ("country", "exports", "product")
                with torch.no_grad():
                    pred = model(test_data_dev, test_data_dev[edge_type].edge_label_index,
                                "country", "product")
                scores = torch.sigmoid(pred).cpu().numpy()
                labels = test_data_dev[edge_type].edge_label.cpu().numpy()
                test_metrics = compute_all_metrics(scores, labels)

            all_results[model_name] = test_metrics
            logger.info(f"  Test AUC={test_metrics['roc_auc']:.4f} "
                       f"AP={test_metrics['avg_precision']:.4f} "
                       f"MRR={test_metrics['mrr']:.4f}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    # ── 4. Baselines ──
    logger.info(f"\n{'─'*40}")
    logger.info("Running baselines...")
    logger.info(f"{'─'*40}")

    # Classical baselines (Adamic-Adar, Jaccard, etc.)
    baselines_path = data_cfg.processed_dir / "classical_baselines.parquet"
    if baselines_path.exists():
        logger.info("  Classical baselines already computed (from Phase 3)")

    # Density baseline
    try:
        from mktrade.eval.density_baseline import density_baseline

        complexity_df = pd.read_parquet(data_cfg.processed_dir / "complexity.parquet")

        # Build train edge set
        train_exp = atlas_df[
            (atlas_df["year"] <= train_cfg.train_end_year) & (atlas_df["export_value"] > 0)
        ]
        train_edge_set = set(zip(train_exp["iso3"], train_exp["hs4"]))

        # Test edges from the test data
        test_ei = test_data["country", "exports", "product"].edge_label_index
        test_labels_t = test_data["country", "exports", "product"].edge_label
        countries = train_data["country"].iso3
        products = train_data["product"].hs4

        test_edges_list = [
            (countries[test_ei[0, i].item()], products[test_ei[1, i].item()])
            for i in range(test_ei.shape[1])
        ]

        density_results = density_baseline(
            density_df=complexity_df[["iso3", "hs4", "density"]].dropna(),
            train_edges=train_edge_set,
            test_edges=test_edges_list,
            test_labels=test_labels_t.numpy(),
            score_col="density",
        )
        all_results["density_baseline"] = density_results
    except Exception as e:
        logger.warning(f"  Density baseline failed: {e}")

    # ── 5. Comparison table ──
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS COMPARISON")
    logger.info(f"{'='*60}")

    from mktrade.eval.compare import compare_models
    comparison = compare_models(all_results)

    # Save results
    results_path = data_cfg.processed_dir / "model_comparison.parquet"
    comparison.to_parquet(results_path)
    logger.info(f"\nResults saved to {results_path}")

    # Also save as CSV for easy viewing
    csv_path = Path("reports") / "model_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, float_format="%.4f")
    logger.info(f"CSV saved to {csv_path}")

    logger.info(f"\nPhase 4 COMPLETE.")


if __name__ == "__main__":
    main()
