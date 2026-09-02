"""Phase 3 — Build the knowledge graph.

1. Loads processed parquets (atlas, complexity, bilateral, proximity, gravity, WDI).
2. Builds a NetworkX in-memory graph (always).
3. Loads into Neo4j + runs GDS embeddings (if Neo4j is available).
4. Converts to PyG HeteroData and saves to data/processed/.

Usage:
    python scripts/02_build_graph.py [--year 2022] [--skip-neo4j]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from mktrade.graph.build_nx import (
        build_bipartite_graph,
        build_full_hetero_graph,
        classical_link_baselines,
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from loguru import logger

from mktrade.config import load_data_config, load_neo4j_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Build knowledge graph")
    parser.add_argument("--year", type=int, default=2022, help="Snapshot year (default: 2022)")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j loading")
    args = parser.parse_args()

    data_cfg = load_data_config()
    neo4j_cfg = load_neo4j_config()
    year = args.year

    logger.info(f"{'='*60}")
    logger.info(f"PHASE 3 — Knowledge Graph Construction (year={year})")
    logger.info(f"{'='*60}")

    # ── 1. Load all processed data ──
    logger.info("Loading processed data...")

    atlas_df = pd.read_parquet(data_cfg.processed_dir / "atlas_exports.parquet")
    logger.info(f"  Atlas exports: {len(atlas_df):,} rows")

    country_complexity = pd.read_parquet(data_cfg.processed_dir / "country_complexity.parquet")
    product_complexity = pd.read_parquet(data_cfg.processed_dir / "product_complexity.parquet")
    logger.info(f"  Country complexity: {len(country_complexity)} rows")
    logger.info(f"  Product complexity: {len(product_complexity)} rows")

    proximity_df = pd.read_parquet(data_cfg.processed_dir / "proximity_matrix.parquet")
    logger.info(f"  Proximity matrix: {len(proximity_df):,} rows")

    bilateral_df = None
    bil_path = data_cfg.processed_dir / "bilateral_flows.parquet"
    if bil_path.exists():
        bilateral_df = pd.read_parquet(bil_path)
        logger.info(f"  Bilateral flows: {len(bilateral_df):,} rows")

    gravity_df = None
    grav_path = data_cfg.external_dir / "gravity_clean.parquet"
    if grav_path.exists():
        gravity_df = pd.read_parquet(grav_path)
        logger.info(f"  Gravity: {len(gravity_df):,} rows")

    wdi_df = None
    wdi_path = data_cfg.external_dir / "wdi_indicators.parquet"
    if wdi_path.exists():
        wdi_df = pd.read_parquet(wdi_path)
        logger.info(f"  WDI indicators: {len(wdi_df):,} rows")

    # ── 2. Build NetworkX graphs ──
    logger.info(f"\n{'─'*40}")
    logger.info("Building NetworkX graphs...")



    bipartite_G = build_bipartite_graph(
        exports_df=atlas_df,
        country_attrs=country_complexity,
        product_attrs=product_complexity,
        year=year,
    )

    hetero_G = build_full_hetero_graph(
        exports_df=atlas_df,
        bilateral_df=bilateral_df,
        gravity_df=gravity_df,
        proximity_df=proximity_df,
        country_attrs=country_complexity,
        product_attrs=product_complexity,
        year=year,
    )

    # Compute classical baselines for MKD (top candidate products)
    mkd_node = data_cfg.focus_country
    mkd_products = set(bipartite_G.neighbors(mkd_node)) if bipartite_G.has_node(mkd_node) else set()
    all_products = {n for n, d in bipartite_G.nodes(data=True) if d.get("bipartite") == 1}
    candidate_products = all_products - mkd_products

    if candidate_products:
        candidate_edges = [(mkd_node, p) for p in sorted(candidate_products)[:500]]
        baselines_df = classical_link_baselines(bipartite_G, candidate_edges)
        baselines_path = data_cfg.processed_dir / "classical_baselines.parquet"
        baselines_df.to_parquet(baselines_path, index=False)
        logger.info(f"  Classical baselines saved: {baselines_path} ({len(baselines_df)} edges)")

        # Show top 10 by Adamic-Adar
        if not baselines_df.empty and "adamic_adar" in baselines_df.columns:
            top10 = baselines_df.nlargest(10, "adamic_adar")
            logger.info(f"\n  Top 10 {mkd_node} candidate products (Adamic-Adar):")
            for _, row in top10.iterrows():
                logger.info(f"    {row['node_v']:>10s}  AA={row['adamic_adar']:.4f}  "
                           f"JC={row['jaccard']:.4f}  CN={row['cn']}")

    # ── 3. Neo4j loading (optional) ──
    logger.info(f"\n{'─'*40}")
    if args.skip_neo4j:
        logger.info("Neo4j loading skipped (--skip-neo4j)")
    else:
        logger.info("Attempting Neo4j loading...")
        from mktrade.graph.build_neo4j import Neo4jLoader

        neo4j = Neo4jLoader(neo4j_cfg)
        if neo4j.available:
            neo4j.clear_database()
            neo4j.create_constraints()

            # Country nodes: merge complexity + WDI
            atlas_year = atlas_df[atlas_df["year"] == year]
            country_nodes = country_complexity.copy()
            if wdi_df is not None:
                wdi_year = wdi_df[wdi_df["year"] == year] if "year" in wdi_df.columns else wdi_df
                country_nodes = country_nodes.merge(wdi_year, on="iso3", how="left", suffixes=("", "_wdi"))
                # Drop duplicate year column if exists
                country_nodes = country_nodes.drop(columns=[c for c in country_nodes.columns if c.endswith("_wdi")],
                                                   errors="ignore")
            neo4j.load_countries(country_nodes)

            # Product nodes
            neo4j.load_products(product_complexity)

            # EXPORTS edges
            exports_year = atlas_year[atlas_year["export_value"] > 0][["iso3", "hs4", "export_value"]].copy()
            if "export_rca" in atlas_year.columns:
                exports_year["export_rca"] = atlas_year.loc[exports_year.index, "export_rca"]
            neo4j.load_exports(exports_year)

            # Bilateral (TRADES_WITH)
            if bilateral_df is not None:
                bil_year = bilateral_df[bilateral_df["year"] == year]
                if not bil_year.empty:
                    agg = bil_year.groupby(["reporter_iso3", "partner_iso3", "flow"])["value"].sum().reset_index()
                    exp_agg = agg[agg["flow"] == "X"].rename(columns={"value": "export_value"})
                    imp_agg = agg[agg["flow"] == "M"].rename(columns={"value": "import_value"})
                    merged = exp_agg.merge(
                        imp_agg[["reporter_iso3", "partner_iso3", "import_value"]],
                        on=["reporter_iso3", "partner_iso3"], how="outer"
                    ).fillna(0)
                    merged = merged[merged["reporter_iso3"] != merged["partner_iso3"]]
                    neo4j.load_bilateral(merged)

            # Proximity
            prox_year = proximity_df[proximity_df["year"] == year] if "year" in proximity_df.columns else proximity_df
            prox_year = prox_year[prox_year["proximity"] > 0.1]  # Only significant proximities
            neo4j.load_proximity(prox_year)

            # Gravity (NEIGHBOR_OF)
            if gravity_df is not None:
                grav_year = gravity_df[gravity_df["year"] == year] if "year" in gravity_df.columns else gravity_df
                neo4j.load_gravity(grav_year)

            stats = neo4j.get_stats()
            logger.info(f"  Neo4j stats: {stats}")
            neo4j.close()
        else:
            logger.info("Neo4j not available — skipping")

    # ── 4. Build PyG HeteroData ──
    logger.info(f"\n{'─'*40}")
    logger.info("Building PyG HeteroData...")

    from mktrade.graph.pyg_data import (
        build_hetero_data,
        enrich_country_features,
        save_hetero_data,
    )

    hetero_data = build_hetero_data(
        exports_df=atlas_df,
        country_features=country_complexity,
        product_features=product_complexity,
        bilateral_df=bilateral_df,
        gravity_df=gravity_df,
        proximity_df=proximity_df,
        year=year,
        cefta_members=set(data_cfg.cefta_members),
    )

    # Enrich with WDI features
    if wdi_df is not None:
        hetero_data = enrich_country_features(
            hetero_data,
            wdi_df=wdi_df,
            countries=hetero_data["country"].iso3,
            year=year,
        )

    # Save
    pyg_path = data_cfg.processed_dir / f"hetero_data_y{year}.pt"
    save_hetero_data(hetero_data, pyg_path)

    # ── 5. Validation ──
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION")
    logger.info(f"{'='*60}")

    # Check MKD is present
    countries = hetero_data["country"].iso3
    if data_cfg.focus_country in countries:
        mkd_idx = countries.index(data_cfg.focus_country)
        mkd_feat = hetero_data["country"].x[mkd_idx]
        logger.info(f"  {data_cfg.focus_country} node found (index {mkd_idx})")
        logger.info(f"  {data_cfg.focus_country} features (first 8): {mkd_feat[:8].tolist()}")

        # Count MKD export edges
        exp_edge_index = hetero_data["country", "exports", "product"].edge_index
        mkd_exports = (exp_edge_index[0] == mkd_idx).sum().item()
        logger.info(f"  {data_cfg.focus_country} export edges: {mkd_exports}")
    else:
        logger.warning(f"  {data_cfg.focus_country} NOT FOUND in graph!")

    # Summary
    logger.info(f"\nNode types: {hetero_data.node_types}")
    logger.info(f"Edge types: {hetero_data.edge_types}")
    logger.info(f"\nPhase 3 COMPLETE. Saved: {pyg_path}")


if __name__ == "__main__":
    main()
