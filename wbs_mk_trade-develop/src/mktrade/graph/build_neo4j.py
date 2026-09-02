"""Ingest the knowledge graph into Neo4j and run GDS algorithms.

Handles batch-creation of nodes/edges via Cypher UNWIND and runs
GDS structural-embedding algorithms (FastRP, Node2Vec).

Graceful degradation: if Neo4j is not reachable, all methods log a warning
and return without error. Use `Neo4jLoader.available` to check connectivity.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from mktrade.config import Neo4jConfig
from mktrade.graph.build_nx import HS_SECTION_NAMES, _hs4_to_section


def _try_import_neo4j():
    """Import neo4j driver, returning None if not installed."""
    try:
        from neo4j import GraphDatabase
        return GraphDatabase
    except ImportError:
        return None


class Neo4jLoader:
    """Manages the Neo4j connection and bulk-loading.

    If Neo4j is unreachable or the driver is not installed, ``available``
    is ``False`` and all load/run methods become no-ops.
    """

    def __init__(self, cfg: Neo4jConfig) -> None:
        self._cfg = cfg
        self._driver = None
        self._db = cfg.database
        self._batch = cfg.import_batch_size
        self.available = False

        GraphDatabase = _try_import_neo4j()
        if GraphDatabase is None:
            logger.warning("neo4j Python driver not installed — skipping Neo4j")
            return

        try:
            self._driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
            self._driver.verify_connectivity()
            self.available = True
            logger.info(f"Connected to Neo4j at {cfg.uri}")
        except Exception as e:
            logger.warning(f"Neo4j not reachable at {cfg.uri}: {e}  — continuing without Neo4j")
            self._driver = None

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def _run(self, query: str, **params: Any) -> list[dict]:
        """Execute a Cypher query, returning list of record dicts."""
        if not self.available:
            return []
        with self._driver.session(database=self._db) as session:
            result = session.run(query, **params)
            return [r.data() for r in result]

    def _run_write(self, query: str, **params: Any) -> None:
        """Execute a write Cypher query."""
        if not self.available:
            return
        with self._driver.session(database=self._db) as session:
            session.execute_write(lambda tx: tx.run(query, **params))

    # ── Schema ──

    def create_constraints(self) -> None:
        """Create uniqueness constraints for node keys."""
        if not self.available:
            return
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Country) REQUIRE c.iso3 IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.hs_code IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:ProductSection) REQUIRE s.code IS UNIQUE",
        ]
        for q in constraints:
            self._run_write(q)
        logger.info("Neo4j constraints created")

    def clear_database(self) -> None:
        """Delete all nodes and relationships (use with caution)."""
        if not self.available:
            return
        self._run_write("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j database cleared")

    # ── Node loading ──

    def load_countries(self, df: pd.DataFrame) -> None:
        """Batch-create Country nodes from country_complexity + WDI data.

        Expected columns: iso3, plus optional eci, diversity, gdp, gdp_pc, population.
        """
        if not self.available:
            return

        records = df.to_dict("records")
        for i in range(0, len(records), self._batch):
            batch = records[i:i + self._batch]
            self._run_write(
                """
                UNWIND $rows AS row
                MERGE (c:Country {iso3: row.iso3})
                SET c += row
                """,
                rows=batch,
            )
        logger.info(f"Neo4j: loaded {len(records)} Country nodes")

    def load_products(self, df: pd.DataFrame) -> None:
        """Batch-create Product nodes + ProductSection nodes + IN_SECTION edges.

        Expected columns: hs4, plus optional pci, ubiquity.
        """
        if not self.available:
            return

        # Build product records with section info
        records = []
        sections_seen: set[str] = set()
        for _, row in df.iterrows():
            hs4 = str(row["hs4"])
            sec = _hs4_to_section(hs4)
            rec = row.to_dict()
            rec["hs_code"] = hs4
            rec["section"] = sec
            records.append(rec)
            sections_seen.add(sec)

        # Create section nodes
        sec_records = [
            {"code": s, "name": HS_SECTION_NAMES.get(s, f"Section {s}")}
            for s in sorted(sections_seen)
        ]
        self._run_write(
            """
            UNWIND $rows AS row
            MERGE (s:ProductSection {code: row.code})
            SET s.name = row.name
            """,
            rows=sec_records,
        )

        # Create product nodes + IN_SECTION edges
        for i in range(0, len(records), self._batch):
            batch = records[i:i + self._batch]
            self._run_write(
                """
                UNWIND $rows AS row
                MERGE (p:Product {hs_code: row.hs_code})
                SET p += row
                WITH p, row
                MATCH (s:ProductSection {code: row.section})
                MERGE (p)-[:IN_SECTION]->(s)
                """,
                rows=batch,
            )
        logger.info(f"Neo4j: loaded {len(records)} Product nodes, {len(sec_records)} sections")

    def load_exports(self, df: pd.DataFrame) -> None:
        """Batch-create EXPORTS edges (Country→Product).

        Expected columns: iso3, hs4, export_value, optional export_rca.
        """
        if not self.available:
            return

        records = []
        for _, row in df.iterrows():
            rec = {
                "iso3": row["iso3"],
                "hs_code": str(row["hs4"]),
                "value": float(row["export_value"]),
            }
            if "export_rca" in row and pd.notna(row.get("export_rca")):
                rec["rca"] = float(row["export_rca"])
            records.append(rec)

        for i in range(0, len(records), self._batch):
            batch = records[i:i + self._batch]
            self._run_write(
                """
                UNWIND $rows AS row
                MATCH (c:Country {iso3: row.iso3})
                MATCH (p:Product {hs_code: row.hs_code})
                MERGE (c)-[e:EXPORTS]->(p)
                SET e.value = row.value
                SET e.rca = row.rca
                """,
                rows=batch,
            )
        logger.info(f"Neo4j: loaded {len(records)} EXPORTS edges")

    def load_bilateral(self, df: pd.DataFrame) -> None:
        """Batch-create TRADES_WITH edges (Country→Country, aggregated).

        Expected columns: reporter_iso3, partner_iso3, export_value, import_value.
        """
        if not self.available:
            return

        records = df.to_dict("records")
        for i in range(0, len(records), self._batch):
            batch = records[i:i + self._batch]
            self._run_write(
                """
                UNWIND $rows AS row
                MATCH (r:Country {iso3: row.reporter_iso3})
                MATCH (p:Country {iso3: row.partner_iso3})
                MERGE (r)-[e:TRADES_WITH]->(p)
                SET e.export_value = row.export_value,
                    e.import_value = row.import_value
                """,
                rows=batch,
            )
        logger.info(f"Neo4j: loaded {len(records)} TRADES_WITH edges")

    def load_proximity(self, df: pd.DataFrame) -> None:
        """Batch-create PROXIMITY edges between products.

        Expected columns: hs4_1, hs4_2, proximity.
        """
        if not self.available:
            return

        records = [
            {
                "hs_code_1": str(row["hs4_1"]),
                "hs_code_2": str(row["hs4_2"]),
                "weight": float(row["proximity"]),
            }
            for _, row in df.iterrows()
        ]

        for i in range(0, len(records), self._batch):
            batch = records[i:i + self._batch]
            self._run_write(
                """
                UNWIND $rows AS row
                MATCH (p1:Product {hs_code: row.hs_code_1})
                MATCH (p2:Product {hs_code: row.hs_code_2})
                MERGE (p1)-[e:PROXIMITY]->(p2)
                SET e.weight = row.weight
                """,
                rows=batch,
            )
        logger.info(f"Neo4j: loaded {len(records)} PROXIMITY edges")

    def load_gravity(self, df: pd.DataFrame) -> None:
        """Batch-create NEIGHBOR_OF edges with gravity covariates.

        Expected columns: iso3_o, iso3_d, dist, contig, comlang_off, fta_wto.
        """
        if not self.available:
            return

        records = [
            {
                "iso3_o": row["iso3_o"],
                "iso3_d": row["iso3_d"],
                "distance": float(row.get("dist", 0)),
                "contiguous": int(row.get("contig", 0)),
                "common_lang": int(row.get("comlang_off", 0)),
                "fta": int(row.get("fta_wto", 0)),
            }
            for _, row in df.iterrows()
        ]

        for i in range(0, len(records), self._batch):
            batch = records[i:i + self._batch]
            self._run_write(
                """
                UNWIND $rows AS row
                MATCH (o:Country {iso3: row.iso3_o})
                MATCH (d:Country {iso3: row.iso3_d})
                MERGE (o)-[e:NEIGHBOR_OF]->(d)
                SET e.distance = row.distance,
                    e.contiguous = row.contiguous,
                    e.common_lang = row.common_lang,
                    e.fta = row.fta
                """,
                rows=batch,
            )
        logger.info(f"Neo4j: loaded {len(records)} NEIGHBOR_OF edges")

    # ── GDS algorithms ──

    def run_fastrp(self, dimensions: int = 64) -> pd.DataFrame | None:
        """Run GDS FastRP and return embeddings as a DataFrame.

        Returns DataFrame with columns: node_id, node_type, embedding (list).
        Returns None if GDS is not available.
        """
        if not self.available:
            return None

        try:
            # Check if GDS is available
            result = self._run("RETURN gds.version() AS version")
            logger.info(f"GDS version: {result[0]['version']}")
        except Exception as e:
            logger.warning(f"GDS not available: {e}")
            return None

        try:
            # Project graph
            self._run_write("""
                CALL gds.graph.project(
                    'trade-graph',
                    ['Country', 'Product', 'ProductSection'],
                    {
                        EXPORTS: {orientation: 'UNDIRECTED'},
                        PROXIMITY: {orientation: 'UNDIRECTED'},
                        IN_SECTION: {orientation: 'UNDIRECTED'}
                    }
                )
            """)

            # Run FastRP
            result = self._run(f"""
                CALL gds.fastRP.stream('trade-graph', {{
                    embeddingDimension: {dimensions},
                    iterationWeights: [0.0, 1.0, 1.0, 0.8]
                }})
                YIELD nodeId, embedding
                RETURN gds.util.asNode(nodeId).iso3 AS iso3,
                       gds.util.asNode(nodeId).hs_code AS hs_code,
                       labels(gds.util.asNode(nodeId))[0] AS node_type,
                       embedding
            """)

            # Cleanup
            self._run_write("CALL gds.graph.drop('trade-graph')")

            if result:
                df = pd.DataFrame(result)
                logger.info(f"FastRP embeddings: {len(df)} nodes, {dimensions} dims")
                return df
        except Exception as e:
            logger.warning(f"FastRP failed: {e}")
            # Try to clean up
            try:
                self._run_write("CALL gds.graph.drop('trade-graph', false)")
            except Exception:
                pass

        return None

    def run_node2vec(self, dimensions: int = 64, walk_length: int = 80,
                     walks_per_node: int = 10) -> pd.DataFrame | None:
        """Run GDS Node2Vec and return embeddings."""
        if not self.available:
            return None

        try:
            result = self._run("RETURN gds.version() AS version")
        except Exception:
            logger.warning("GDS not available for Node2Vec")
            return None

        try:
            self._run_write("""
                CALL gds.graph.project(
                    'trade-graph-n2v',
                    ['Country', 'Product'],
                    {
                        EXPORTS: {orientation: 'UNDIRECTED'},
                        PROXIMITY: {orientation: 'UNDIRECTED'}
                    }
                )
            """)

            result = self._run(f"""
                CALL gds.node2vec.stream('trade-graph-n2v', {{
                    embeddingDimension: {dimensions},
                    walkLength: {walk_length},
                    walksPerNode: {walks_per_node}
                }})
                YIELD nodeId, embedding
                RETURN gds.util.asNode(nodeId).iso3 AS iso3,
                       gds.util.asNode(nodeId).hs_code AS hs_code,
                       labels(gds.util.asNode(nodeId))[0] AS node_type,
                       embedding
            """)

            self._run_write("CALL gds.graph.drop('trade-graph-n2v')")

            if result:
                df = pd.DataFrame(result)
                logger.info(f"Node2Vec embeddings: {len(df)} nodes, {dimensions} dims")
                return df
        except Exception as e:
            logger.warning(f"Node2Vec failed: {e}")
            try:
                self._run_write("CALL gds.graph.drop('trade-graph-n2v', false)")
            except Exception:
                pass

        return None

    def run_gds_link_prediction(self) -> pd.DataFrame:
        """Run GDS built-in link-prediction pipeline as a baseline.

        This is a placeholder — will be implemented when we set up the
        evaluation pipeline in Phase 4.
        """
        logger.warning("GDS link prediction not yet implemented (Phase 4)")
        return pd.DataFrame()

    def get_stats(self) -> dict[str, int]:
        """Return node and edge counts from Neo4j."""
        if not self.available:
            return {}
        nodes = self._run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt")
        edges = self._run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt")
        return {
            "nodes": {r["label"]: r["cnt"] for r in nodes},
            "edges": {r["rel_type"]: r["cnt"] for r in edges},
        }
