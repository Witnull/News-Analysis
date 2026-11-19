# import os
from datetime import datetime
# from typing import List, Dict
# from neo4j import GraphDatabase

# from modules.analysis.kg_models import Entity, Fact


# class Neo4jGraphStore:
#     """Persistent, schema-less Knowledge Graph store with incremental updates."""
    
#     def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = ""):
#         self.driver = GraphDatabase.driver(uri, auth=(user, password))
#         print("[Neo4jGraphStore] Connected to Neo4j")

#     def close(self):
#         self.driver.close()

#     def upsert_triples(self, triples: List[Triple], source: str):
#         """
#         Batch upsert triples into Neo4j.
#         Uses MERGE and map projection (+=) to dynamically add/update ANY properties 
#         without requiring schema migrations.
#         """
#         if not triples:
#             return

#         now = datetime.now().isoformat()
#         records = []
        
#         for t in triples:
#             # 1. Handle both Triple objects and Dicts seamlessly

#             subj = t.subject
#             pred = t.predicate.replace(" ", "_").upper()
#             obj = t.object
#             props = t.to_neo4j_properties()
            
            
#             if not pred:
#                 pred = "RELATED_TO"

#             # 2. Ensure core tracking properties exist in the dynamic dict
#             rel_props = props["rel_props"]
#             rel_props["hit_count"] = rel_props.get("hit_count", 1)
#             rel_props["first_seen"] = rel_props.get("first_seen", now)
#             rel_props["last_seen"] = now
            
#             # 3. Handle 'sources' array safely (avoids complex Cypher list deduplication)
#             existing_sources = rel_props.get("sources", [])
#             if isinstance(existing_sources, str):
#                 existing_sources = [existing_sources]
            
#             if source and source not in existing_sources:
#                 existing_sources.append(source)
#             rel_props["sources"] = existing_sources

#             records.append({
#                 "subject": subj,
#                 "predicate": pred,
#                 "object": obj,
#                 "subject_props": props["subject_props"],
#                 "object_props": props["object_props"],
#                 "rel_props": rel_props,
#                 "now": now
#             })

#         # 4. DYNAMIC CYPHER QUERY
#         # The `+=` operator merges the incoming dictionary with existing node properties.
#         # If a new key (e.g., "L1_topic") appears, it is added automatically.
#         cypher_query = """
#         UNWIND $records AS row
        
#         // Merge Subject Node and dynamically update/add properties
#         MERGE (s:Entity {name: row.subject})
#         ON CREATE SET s.created = row.now
#         SET s += row.subject_props, s.last_seen = row.now
        
#         // Merge Object Node and dynamically update/add properties
#         MERGE (o:Entity {name: row.object})
#         ON CREATE SET o.created = row.now
#         SET o += row.object_props, o.last_seen = row.now
        
#         // Merge Relationship. We use a generic :RELATIONSHIP label and store 
#         // the actual predicate as a property. This avoids dynamic relationship 
#         // type creation (which requires APOC) and keeps queries uniform.
#         MERGE (s)-[r:RELATIONSHIP {predicate: row.predicate}]->(o)
#         ON CREATE SET r += row.rel_props
#         ON MATCH SET r.hit_count = coalesce(r.hit_count, 0) + 1, 
#                     r.last_seen = row.now,
#                     r.sources = row.rel_props.sources
#         """
        
#         with self.driver.session() as session:
#             session.run(cypher_query, records=records)
        
#         print(f"[Neo4jGraphStore] Upserted {len(records)} triples from '{source}'")

#     def get_entity_summary(self, entity_name: str, limit: int = 10, hops: int = 1, predicate: str = "") -> List[Dict]:
#         """
#         LightRAG Retrieval: Supports 1-hop or n-hop, with optional predicate filtering.
#         """
#         # Base WHERE clause for predicate filtering
#         # Checks both the 'predicate' property (for generic :RELATIONSHIP) and the relationship type (for dynamic types)
#         pred_filter = "WHERE $predicate IS NULL OR toLower(coalesce(r.predicate, type(r))) CONTAINS toLower($predicate)"
        
#         if hops == 1:
#             query = f"""
#             MATCH (n:Entity {{name: $entity}})-[r]-(other:Entity)
#             {pred_filter}
#             WITH n, r, other,
#                  CASE WHEN n.name = $entity THEN n ELSE other END AS subject_node,
#                  CASE WHEN n.name = $entity THEN other ELSE n END AS object_node
#             RETURN subject_node.name AS subject,
#                    coalesce(r.predicate, type(r)) AS predicate,
#                    r.hit_count AS hit_count,
#                    object_node.name AS object
#             ORDER BY r.hit_count DESC
#             LIMIT $limit
#             """
#             with self.driver.session() as session:
#                 result = session.run(query, entity=entity_name, limit=limit, predicate=predicate)
#                 records = []
#                 for record in result:
#                     records.append({
#                         "type": "triple",
#                         "subject": record["subject"],
#                         "predicate": record["predicate"],
#                         "object": record["object"],
#                         "hit_count": record["hit_count"]
#                     })
#                 return records

#         else:
#             # For n-hop, we filter paths where ALL relationships match the predicate
#             path_pred_filter = "WHERE $predicate IS NULL OR ALL(rel IN relationships(path) WHERE toLower(coalesce(rel.predicate, type(rel))) CONTAINS toLower($predicate))"
#             query = f"""
#             MATCH path = (start:Entity {{name: $entity}})-[*1..$hops]-(end:Entity)
#             {path_pred_filter}
#             RETURN path
#             ORDER BY length(path) ASC
#             LIMIT $limit
#             """
#             with self.driver.session() as session:
#                 result = session.run(query, entity=entity_name, limit=limit, hops=hops, predicate=predicate)
#                 records = []
#                 for record in result:
#                     path = record["path"]
#                     nodes = path.nodes
#                     rels = path.relationships
                    
#                     chain_parts = [nodes[0].get("name", "Unknown")]
#                     total_hit_count = 0
                    
#                     for i, rel in enumerate(rels):
#                         pred = rel.get("predicate") or getattr(rel, 'type', 'RELATED_TO')
#                         total_hit_count += rel.get("hit_count", 0)
                        
#                         if rel.start_node == nodes[i]:
#                             chain_parts.append(f" -[{pred}]-> {nodes[i+1].get('name', 'Unknown')}")
#                         else:
#                             chain_parts.append(f" <-[{pred}]- {nodes[i+1].get('name', 'Unknown')}")
                    
#                     records.append({
#                         "type": "path",
#                         "path_string": "".join(chain_parts),
#                         "hit_count": total_hit_count / len(rels) if rels else 0,
#                         "hops": len(rels)
#                     })
#                 return records

#     def get_global_stats(self) -> Dict:
#         query = """
#         MATCH (n:Entity)
#         OPTIONAL MATCH ()-[r:RELATIONSHIP]->()
#         RETURN count(DISTINCT n) AS entities, count(r) AS relationships
#         """
#         with self.driver.session() as session:
#             return dict(session.run(query).single()) # type: ignore
from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from src.modules.analysis.kg_models import Entity, Fact


class Neo4jGraphStore:
    """Persistent, schema-less Knowledge Graph store with incremental updates."""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password: Optional[str] = None):
        password = password or os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("Set NEO4J_PASSWORD or pass a Neo4j password.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_schema()
        print("[Neo4jGraphStore] Connected to Neo4j")

    def _ensure_schema(self):
        """Unique constraint on Entity.name → index-backed MERGE + fast lookups."""
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (n:Entity) REQUIRE n.name IS UNIQUE"
            )

    @staticmethod
    def _pretty_predicate(p: Optional[str]) -> str:
        """'INVESTS_IN' -> 'invests in' (readable for LLM context)."""
        return (p or "").replace("_", " ").lower()

    # ── entity resolution ───────────────────────────────────────────────
    def resolve_entity(self, name: str) -> Optional[str]:
        """Canonical stored name: exact (indexed) → alias → substring."""
        if not name:
            return None
        with self.driver.session() as session:
            rec = session.run(
                "MATCH (n:Entity {name: $name}) RETURN n.name AS name LIMIT 1",
                name=name,
            ).single()
            if rec:
                return rec["name"]
            rec = session.run(
                """MATCH (n:Entity)
                WHERE ANY(a IN coalesce(n.aliases, []) WHERE toLower(a) = toLower($name))
                RETURN n.name AS name LIMIT 1""",
                name=name,
            ).single()
            if rec:
                return rec["name"]
            rec = session.run(
                """MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($name)
                RETURN n.name AS name ORDER BY size(n.name) ASC LIMIT 1""",
                name=name,
            ).single()
            return rec["name"] if rec else None

    def search_entities(self, query: str, limit: int = 10) -> List[Dict]:
        """Name/alias substring search, ranked by connectivity."""
        if not query:
            return []
        cypher = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($q)
           OR ANY(a IN coalesce(n.aliases, []) WHERE toLower(a) CONTAINS toLower($q))
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        ORDER BY degree DESC
        LIMIT $limit
        RETURN n.name AS name, n.type AS type, n.aliases AS aliases, degree
        """
        with self.driver.session() as session:
            return [dict(r) for r in session.run(cypher, q=query, limit=limit)]

    # ── main retrieval ──────────────────────────────────────────────────
    def get_entity_summary(
        self,
        entity_name: str,
        limit: int = 10,
        hops: int = 1,
        predicate: str = "",
        edge_type: str = "",
    ) -> List[Dict]:
        """
        LightRAG retrieval around one entity.
        - Auto-resolves names/aliases ("samsung" -> "Samsung")
        - 1-hop: true-direction triples with full metadata
        - n-hop: shortest, strongest paths
        - Optional predicate (substring) and edge_type (semantic/causal/...) filters
        """
        canonical = self.resolve_entity(entity_name)
        if canonical is None:
            return []

        pred = predicate.strip() or None
        etype = edge_type.strip().lower() or None

        if hops <= 1:
            filters = []
            if pred:
                filters.append("toLower(r.predicate) CONTAINS toLower($predicate)")
            if etype:
                filters.append("r.edge_type = $edge_type")
            where = f"WHERE {' AND '.join(filters)}" if filters else ""

            cypher = f"""
            MATCH (a:Entity {{name: $entity}})-[r:RELATIONSHIP]->(b:Entity)
            {where}
            RETURN a.name AS subject, a.type AS subject_type,
                   r.predicate AS predicate, b.name AS object, b.type AS object_type,
                   coalesce(r.hit_count, 1) AS hit_count, r.description AS description,
                   r.edge_type AS edge_type, r.sources AS sources, r.last_seen AS last_seen,
                   'out' AS direction
            UNION ALL
            MATCH (b:Entity)-[r:RELATIONSHIP]->(a:Entity {{name: $entity}})
            {where}
            RETURN b.name AS subject, b.type AS subject_type,
                   r.predicate AS predicate, a.name AS object, a.type AS object_type,
                   coalesce(r.hit_count, 1) AS hit_count, r.description AS description,
                   r.edge_type AS edge_type, r.sources AS sources, r.last_seen AS last_seen,
                   'in' AS direction
            ORDER BY hit_count DESC, last_seen DESC
            LIMIT $limit
            """
            with self.driver.session() as session:
                return [{
                    "type": "triple",
                    "subject": rec["subject"],
                    "predicate": self._pretty_predicate(rec["predicate"]),
                    "object": rec["object"],
                    "direction": rec["direction"],
                    "edge_type": rec["edge_type"] or "semantic",
                    "hit_count": rec["hit_count"],
                    "description": rec["description"] or "",
                    "sources": rec["sources"] or [],
                    "subject_type": rec["subject_type"] or "",
                    "object_type": rec["object_type"] or "",
                } for rec in session.run(
                    cypher, entity=canonical, predicate=pred, edge_type=etype, limit=limit
                )]

        # ── n-hop paths ──
        hops = min(int(hops), 5)
        filters = []
        if pred:
            filters.append(
                "ANY(rel IN relationships(path) "
                "WHERE toLower(rel.predicate) CONTAINS toLower($predicate))"
            )
        if etype:
            filters.append(
                "ALL(rel IN relationships(path) WHERE rel.edge_type = $edge_type)"
            )
        where = f"WHERE {' AND '.join(filters)}" if filters else ""

        cypher = f"""
        MATCH path = (start:Entity {{name: $entity}})-[:RELATIONSHIP*1..$hops]-(end:Entity)
        {where}
        WITH path, reduce(s = 0, rel IN relationships(path) | s + coalesce(rel.hit_count, 1)) AS strength
        RETURN path, strength
        ORDER BY length(path) ASC, strength DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            records = []
            for rec in session.run(
                cypher, entity=canonical, predicate=pred, edge_type=etype,
                limit=limit, hops=hops,
            ):
                nodes, rels = rec["path"].nodes, rec["path"].relationships
                parts = [nodes[0].get("name", "Unknown")]
                for i, rel in enumerate(rels):
                    fwd = self._node_key(rel.start_node) == self._node_key(nodes[i])
                    arrow = "->" if fwd else "<-"
                    p = self._pretty_predicate(rel.get("predicate") or "RELATED_TO")
                    parts.append(f" -[{p}]{arrow} {nodes[i + 1].get('name', 'Unknown')}")
                records.append({
                    "type": "path",
                    "path_string": "".join(parts),
                    "hops": len(rels),
                    "hit_count": rec["strength"] / len(rels) if rels else 0,
                    "end_node": nodes[-1].get("name", "Unknown"),
                })
            return records

    @staticmethod
    def _node_key(n) -> str:
        return getattr(n, "element_id", None) or str(n.id)

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _clean_props(props: Dict[str, Any]) -> Dict[str, Any]:
        """Drop empty values so `SET n +=` never clobbers stored data with blanks."""
        return {k: v for k, v in props.items() if v not in ("", None, [], {})}

    @staticmethod
    def _as_fact(t: Any) -> Fact:
        if isinstance(t, Fact):
            return t
        if isinstance(t, dict):
            s, p, o = t.get("subject"), t.get("predicate"), t.get("object")
            if s and p and o:
                subj = s if isinstance(s, Entity) else Entity(name=str(s))
                obj = o if isinstance(o, Entity) else Entity(name=str(o))
                return Fact(subject=subj, predicate=str(p), object=obj)
        raise ValueError(f"Cannot coerce to Fact: {t!r}")

    # ── writes ─────────────────────────────────────────────────────────
    def upsert_triples(self, triples: List[Fact], source: str = "unknown") -> None:
        """
        Batch upsert facts (S, P, O) into Neo4j.
        Nodes get FULL Entity properties (type, aliases, topics, description).
        Generic :RELATIONSHIP label with predicate property — no APOC needed.
        """
        if not triples:
            return

        now = datetime.now().isoformat()
        node_sources = [source] if source else []
        records = []

        for t in triples:
            f = self._as_fact(t)
            pred = f.predicate.replace(" ", "_").upper() or "RELATED_TO"

            subject_props = self._clean_props(f.subject.to_neo4j_properties())
            object_props = self._clean_props(f.object.to_neo4j_properties())

            rel_props = self._clean_props(f.to_neo4j_properties()["rel_props"])
            rel_props.setdefault("hit_count", 1)
            rel_props.setdefault("first_seen", now)
            rel_props["last_seen"] = now

            rel_sources = rel_props.get("sources", [])
            if isinstance(rel_sources, str):
                rel_sources = [rel_sources]
            if source and source not in rel_sources:
                rel_sources.append(source)
            rel_props["sources"] = rel_sources

            records.append({
                "subject": f.subject.name,   # MERGE key must be the NAME string
                "predicate": pred,
                "object": f.object.name,
                "subject_props": subject_props,
                "object_props": object_props,
                "rel_props": rel_props,
                "node_sources": node_sources,
                "now": now,
            })

        cypher = """
        UNWIND $records AS row

        MERGE (s:Entity {name: row.subject})
        ON CREATE SET s.created = row.now
        SET s += row.subject_props,
            s.last_seen = row.now,
            s.sources = reduce(acc = coalesce(s.sources, []),
                               x IN row.node_sources |
                               CASE WHEN x IN acc THEN acc ELSE acc + x END)

        MERGE (o:Entity {name: row.object})
        ON CREATE SET o.created = row.now
        SET o += row.object_props,
            o.last_seen = row.now,
            o.sources = reduce(acc = coalesce(o.sources, []),
                               x IN row.node_sources |
                               CASE WHEN x IN acc THEN acc ELSE acc + x END)

        MERGE (s)-[r:RELATIONSHIP {predicate: row.predicate}]->(o)
        ON CREATE SET r += row.rel_props
        ON MATCH SET r.hit_count = coalesce(r.hit_count, 0) + 1,
                    r.last_seen = row.now,
                    r.description = coalesce(row.rel_props.description, r.description),
                    r.sources = reduce(acc = coalesce(r.sources, []),
                                       x IN coalesce(row.rel_props.sources, []) |
                                       CASE WHEN x IN acc THEN acc ELSE acc + x END)
        """

        with self.driver.session() as session:
            session.run(cypher, records=records)

        print(f"[Neo4jGraphStore] Upserted {len(records)} triples from '{source}'")

    def upsert_entities(self, entities: List[Entity], source: str = "unknown") -> None:
        """Upsert standalone entities (listed by the LLM but never used in a
        triple). Same merge semantics as nodes in upsert_triples."""
        if not entities:
            return

        now = datetime.now().isoformat()
        records = [{
            "name": e.name,
            "props": self._clean_props(e.to_neo4j_properties()),
            "sources": [source] if source else [],
            "now": now,
        } for e in entities]

        cypher = """
        UNWIND $records AS row
        MERGE (n:Entity {name: row.name})
        ON CREATE SET n.created = row.now
        SET n += row.props,
            n.last_seen = row.now,
            n.sources = reduce(acc = coalesce(n.sources, []),
                               x IN row.sources |
                               CASE WHEN x IN acc THEN acc ELSE acc + x END)
        """

        with self.driver.session() as session:
            session.run(cypher, records=records)

        print(f"[Neo4jGraphStore] Upserted {len(records)} standalone entities from '{source}'")
