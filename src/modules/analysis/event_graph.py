"""Event-Centric Knowledge Graph & Value-Chain Topology.

Ingests structured opportunity records from src/modules/classifier, maps them
onto the multi-industry checkpoint value chains, and traces upstream/downstream
causal ripple effects and cross-chain opportunities.
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from src.core.types import Article

DATA_DIR = Path(__file__).resolve().parent.parent / "classifier"
TEMPLATE_PATH = DATA_DIR / "checkpoint_graph_template.json"


def _extract_entities_from_text(text: str) -> List[str]:
    """Fast extraction of named actors, companies, institutions, and geographic entities
    from factual event summaries using capitalization and known legal/domain patterns."""
    entities: List[str] = []
    # Match sequences of Capitalized words, acronyms, or corporate/government markers
    matches = re.findall(
        r"\b(?:[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zđáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+(?:\s+[A-ZĐÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zđáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+)*|[A-Z]{2,6})\b",
        text,
    )
    stop_words = {
        "The", "A", "An", "In", "On", "At", "By", "For", "With", "About", "Against",
        "Decision", "No", "QĐ-TTg", "GDP", "QoQ", "YoY", "DEX", "AI", "FDI", "USD", "VND",
        "Labor", "Day", "Central", "Committee", "Resolution", "National", "Ministry",
    }
    for m in matches:
        m_str = m.strip()
        if len(m_str) >= 3 and m_str not in stop_words and not m_str.isdigit():
            entities.append(m_str)
    return list(dict.fromkeys(entities))[:6]


def compute_event_impact(record: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates quantitative event impact magnitude:
    Impact Magnitude = Duration Factor x Scope Factor
    Signed Impact = Duration Factor x Scope Factor x Direction Sign
    """
    duration_str = str(record.get("duration") or "").lower()
    scope_str = str(record.get("scope") or "").lower()
    dir_str = str(record.get("direction") or "").lower()

    # Duration multiplier (1.0 to 3.0)
    if any(k in duration_str for k in ("perm", "struct")):
        dur_weight = 3.0
        dur_label = "Permanent/Structural"
    elif any(k in duration_str for k in ("long", "multi")):
        dur_weight = 2.0
        dur_label = "Long-Term"
    elif any(k in duration_str for k in ("med", "mid")):
        dur_weight = 1.5
        dur_label = "Medium-Term"
    else:
        dur_weight = 1.0
        dur_label = "Short-Term/Transient"

    # Scope multiplier (1.0 to 3.0)
    if any(k in scope_str for k in ("whole", "macro", "global", "national", "system")):
        scope_weight = 3.0
        scope_label = "Systemic/Market-Wide"
    elif any(k in scope_str for k in ("multi", "ind", "sector", "indirect")):
        scope_weight = 2.0
        scope_label = "Multi-Player/Sectoral"
    else:
        scope_weight = 1.0
        scope_label = "Firm-Level/Localized"

    # Direction factor
    if dir_str in ("increase", "positive", "expansion", "growth", "up"):
        dir_sign = 1.0
        dir_label = "Expansionary (+)"
    elif dir_str in ("decrease", "negative", "contraction", "loss", "down"):
        dir_sign = -1.0
        dir_label = "Contractionary (-)"
    else:
        dir_sign = 0.0
        dir_label = "Neutral/Rebalancing"

    magnitude = round(dur_weight * scope_weight, 2)
    signed_impact = round(magnitude * dir_sign, 2)

    if magnitude >= 5.0:
        significance = "CRITICAL / MACRO SHOCK"
    elif magnitude >= 3.0:
        significance = "HIGH / SECTOR DISRUPTION"
    elif magnitude >= 2.0:
        significance = "MODERATE / VALUE-CHAIN SHIFT"
    else:
        significance = "TACTICAL / LOCALIZED"

    return {
        "duration_factor": dur_weight,
        "duration_label": dur_label,
        "scope_factor": scope_weight,
        "scope_label": scope_label,
        "direction_sign": dir_sign,
        "direction_label": dir_label,
        "impact_magnitude": magnitude,
        "signed_impact": signed_impact,
        "significance": significance,
    }


def get_sub_sector_key(checkpoint: Optional[str], gics: Optional[str], sector: Optional[str]) -> str:
    """Extracts granular sub-sector / sub-industry key to avoid clumping broad sectors."""
    if checkpoint:
        if "." in checkpoint:
            return checkpoint.split(".")[0]
        return checkpoint
    if gics and gics not in ("None", "Other", None):
        return gics.lower().replace(" ", "_")
    if sector and sector not in ("None", "Other", None):
        return sector.lower().replace(" ", "_")
    return "general"


class EventGraph:
    """Multi-directed knowledge graph connecting Events, Value Chain Checkpoints,
    Industrial Sectors, GICS Sub-Industries, Stakeholder Roles, and Entities."""

    def __init__(self, template_path: Path = TEMPLATE_PATH):
        self.graph = nx.MultiDiGraph()
        self.template_path = template_path
        self._dirty: bool = True
        self._cached_projection: Optional[nx.Graph] = None
        self._load_checkpoint_topology()

    def _load_checkpoint_topology(self):
        """Loads value chain checkpoints, sequential stages, and cross-chain edges
        from checkpoint_graph_template.json into the base graph."""
        if not self.template_path.exists():
            return

        with open(self.template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

        # 1. Macro layer nodes
        for node in template.get("macro_layer", {}).get("nodes", []):
            nid = node["id"]
            self.graph.add_node(
                nid,
                node_type="Checkpoint",
                is_macro=True,
                description=node.get("description", ""),
                sub_industry="macro",
            )

        # 2. Sub-industry chains
        for sub_name, sub_data in template.get("sub_industries", {}).items():
            cps = sub_data.get("checkpoints", [])
            for idx, cp in enumerate(cps):
                nid = f"{sub_name}.{cp['stage']}"
                self.graph.add_node(
                    nid,
                    node_type="Checkpoint",
                    is_macro=False,
                    sub_industry=sub_name,
                    stage=cp["stage"],
                    description=cp.get("description", ""),
                )
                # Sequential upstream -> downstream edge within the value chain
                if idx > 0:
                    prev_nid = f"{sub_name}.{cps[idx-1]['stage']}"
                    self.graph.add_edge(
                        prev_nid,
                        nid,
                        edge_type="VALUE_CHAIN_FLOW",
                        relation="downstream",
                        weight=1.0,
                    )

            # Cross-industry edges
            for ce in sub_data.get("cross_edges", []):
                target = ce.get("to")
                rel = ce.get("relation", "cross_link")
                if cps and target:
                    source = f"{sub_name}.{cps[-1]['stage']}"  # default from terminal or relevant stage
                    if "." in target:
                        self.graph.add_edge(
                            source,
                            target,
                            edge_type="CROSS_CHAIN_EDGE",
                            relation=rel,
                            weight=1.5,
                        )

    def add_event_record(self, record: Dict[str, Any], event_id: Optional[str] = None) -> str:
        """Ingests a single classifier opportunity record into the graph."""
        summary = (record.get("main_event_summary") or "").strip()
        if not summary:
            return ""

        if not event_id:
            event_id = f"evt_{hashlib.md5(summary.encode('utf-8')).hexdigest()[:12]}"

        cp = record.get("matched_checkpoint")
        gics = record.get("matched_gics_sub_industry")
        sec = record.get("target_sector")
        sub_sector = get_sub_sector_key(cp, gics, sec)
        impact_metrics = compute_event_impact(record)

        # Add Event Node
        self.graph.add_node(
            event_id,
            node_type="Event",
            summary=summary,
            status=record.get("event_status", "Ongoing"),
            date=record.get("event_date", "_"),
            cluster=record.get("cluster"),
            tag=record.get("tag"),
            direction=record.get("direction"),
            duration=record.get("duration"),
            scope=record.get("scope"),
            target_sector=sec,
            sub_sector=sub_sector,
            matched_gics=gics,
            matched_checkpoint=cp,
            **impact_metrics,
        )

        # 1. Link to Matched Checkpoint
        if cp:
            if cp not in self.graph:
                self.graph.add_node(cp, node_type="Checkpoint", is_macro="macro_" in cp)
            self.graph.add_edge(
                event_id,
                cp,
                edge_type="TOUCHES_CHECKPOINT",
                weight=3.0,
            )

        # 2. Link to Target Sector
        sec = record.get("target_sector")
        if sec and sec not in ("Other", "_", "None", None):
            sec_node = f"sec_{sec.replace(' ', '_').lower()}"
            self.graph.add_node(sec_node, node_type="Sector", name=sec)
            self.graph.add_edge(
                event_id,
                sec_node,
                edge_type="BELONGS_TO_SECTOR",
                weight=1.0,
            )

        # 3. Link to GICS Sub-Industry
        gics = record.get("matched_gics_sub_industry")
        if gics:
            gics_node = f"gics_{re.sub(r'\W+', '_', gics.lower())}"
            self.graph.add_node(gics_node, node_type="SubIndustry", name=gics)
            self.graph.add_edge(
                event_id,
                gics_node,
                edge_type="CLASSIFIED_AS_GICS",
                weight=1.5,
            )

        # 4. Link to Stakeholder Roles & Sentiments
        roles = record.get("roles") or []
        for r in roles:
            role_name = r.get("role")
            if role_name:
                role_node = f"role_{role_name.replace(' ', '_').lower()}"
                self.graph.add_node(role_node, node_type="StakeholderRole", name=role_name)
                self.graph.add_edge(
                    event_id,
                    role_node,
                    edge_type="IMPACTS_ROLE",
                    sentiment=r.get("sentiment", "neutral"),
                    scope=r.get("scope", "direct"),
                    weight=1.2,
                )

        # 5. Link to Named Entities
        entities = _extract_entities_from_text(summary)
        for ent in entities:
            ent_node = f"ent_{re.sub(r'\W+', '_', ent.lower())}"
            self.graph.add_node(ent_node, node_type="Entity", name=ent)
            self.graph.add_edge(
                event_id,
                ent_node,
                edge_type="INVOLVES_ENTITY",
                weight=2.0,
            )

        self._dirty = True
        return event_id

    def ingest_records(self, records: List[Dict[str, Any]]) -> List[str]:
        """Batch-ingests records into the graph."""
        event_ids = []
        for r in records:
            if r and r.get("tag"):
                eid = self.add_event_record(r)
                if eid:
                    event_ids.append(eid)
        return event_ids

    def ingest_articles(self, articles: List[Article], verbose: bool = False) -> List[str]:
        """Chains Articles directly from src/core/types into the graph.
        If opportunity records are not yet computed, runs src.modules.classifier.run_pipeline.
        """
        if not articles:
            return []

        needs_pipeline = []
        records_with_articles = []

        for a in articles:
            pool = getattr(a, "no_save_pool", None) or {}
            rec = pool.get("opportunity_record")
            if not rec and pool.get("data_card"):
                try:
                    from src.modules.classifier.sector_mapper import build_opportunity_record
                    rec = build_opportunity_record(a, discard_empty=False)
                except Exception:
                    rec = None
            if rec:
                records_with_articles.append((a, rec))
            else:
                needs_pipeline.append(a)

        if needs_pipeline:
            from src.modules.classifier.pipeline_driver import run_pipeline
            pipe_records = run_pipeline(needs_pipeline, verbose=verbose, discard_empty=False) or []
            for a, r in zip(needs_pipeline, pipe_records):
                if r:
                    if a.no_save_pool is None:
                        a.no_save_pool = {}
                    a.no_save_pool["opportunity_record"] = r
                    records_with_articles.append((a, r))

        event_ids = []
        for a, r in records_with_articles:
            if r and r.get("tag"):
                eid = self.add_event_record(r)
                if eid:
                    event_ids.append(eid)
                    if getattr(a, "url", None):
                        self.graph.nodes[eid]["source_url"] = a.url
                    if getattr(a, "title", None):
                        self.graph.nodes[eid]["source_title"] = a.title
                    a.processed_stage = "analyzed"

        return event_ids

    def trace_ripple_effects(self, event_id: str, max_hops: int = 3) -> Dict[str, Any]:
        """Multi-hop causal propagation & ripple reasoning:
        Traces downstream transmission (e.g. storm/crop loss -> milling shortage -> price rise)
        and cross-chain substitutes (e.g. fertilizer shortage -> biochar/BSFL opportunity).
        """
        if event_id not in self.graph:
            return {"error": f"Event {event_id} not found in graph"}

        event_data = self.graph.nodes[event_id]
        cp = event_data.get("matched_checkpoint")
        if not cp or cp not in self.graph:
            return {
                "event": event_data,
                "ripple_paths": [],
                "downstream_checkpoints": [],
                "related_events": [],
            }

        # 1. Trace downstream along value chain
        downstream: List[Dict[str, Any]] = []
        visited = {cp}
        queue: List[Tuple[str, int, List[str]]] = [(cp, 0, [cp])]

        while queue:
            curr, depth, path = queue.pop(0)
            if depth >= max_hops:
                continue

            for _, neighbor, data in self.graph.out_edges(curr, data=True):
                if data.get("edge_type") in ("VALUE_CHAIN_FLOW", "CROSS_CHAIN_EDGE") and neighbor not in visited:
                    visited.add(neighbor)
                    downstream.append({
                        "checkpoint": neighbor,
                        "hop": depth + 1,
                        "relation": data.get("relation", "downstream"),
                        "path": path + [neighbor],
                        "description": self.graph.nodes[neighbor].get("description", ""),
                    })
                    queue.append((neighbor, depth + 1, path + [neighbor]))

        # 2. Find events co-located on downstream checkpoints (O(deg) in_edges instead of O(V) full node scan)
        downstream_cp_set = {d["checkpoint"] for d in downstream}
        downstream_rel_map = {d["checkpoint"]: d["relation"] for d in downstream}
        related_events: List[Dict[str, Any]] = []
        seen_events: Set[str] = set()

        for cp_node in downstream_cp_set:
            if cp_node not in self.graph:
                continue
            for src_node, _, ed in self.graph.in_edges(cp_node, data=True):
                if ed.get("edge_type") == "TOUCHES_CHECKPOINT" and src_node != event_id and src_node not in seen_events:
                    seen_events.add(src_node)
                    ndata = self.graph.nodes[src_node]
                    related_events.append({
                        "event_id": src_node,
                        "summary": ndata.get("summary"),
                        "direction": ndata.get("direction"),
                        "matched_checkpoint": cp_node,
                        "relation": downstream_rel_map.get(cp_node, "downstream"),
                    })

        return {
            "source_event": event_data,
            "source_checkpoint": cp,
            "downstream_stages": downstream,
            "related_events": related_events,
        }

    def build_event_projection_graph(self) -> nx.Graph:
        """Projects the multi-modal graph into an undirected Event-Event correlation graph.
        Groups communities strictly by SUB-SECTOR / SUB-INDUSTRY value chains (not broad sector).
        Edge weights quantify shared checkpoints (3.0), shared entities (2.5),
        same sub-sector (2.5), value chain adjacency (2.0), and shared GICS (2.0).
        Results are cached until _dirty flag is set.
        """
        if not self._dirty and self._cached_projection is not None:
            return self._cached_projection

        event_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "Event"]
        proj = nx.Graph()

        for eid in event_nodes:
            proj.add_node(eid, **self.graph.nodes[eid])

        for i in range(len(event_nodes)):
            e1 = event_nodes[i]
            d1 = self.graph.nodes[e1]
            cp1 = d1.get("matched_checkpoint")
            sub1 = d1.get("sub_sector")
            gics1 = d1.get("matched_gics")
            ent1 = {v for _, v, ed in self.graph.out_edges(e1, data=True) if ed.get("edge_type") == "INVOLVES_ENTITY"}

            for j in range(i + 1, len(event_nodes)):
                e2 = event_nodes[j]
                d2 = self.graph.nodes[e2]
                cp2 = d2.get("matched_checkpoint")
                sub2 = d2.get("sub_sector")
                gics2 = d2.get("matched_gics")
                ent2 = {v for _, v, ed in self.graph.out_edges(e2, data=True) if ed.get("edge_type") == "INVOLVES_ENTITY"}

                weight = 0.0

                # Same checkpoint stage
                if cp1 and cp2 and cp1 == cp2:
                    weight += 3.0
                # Connected checkpoints along the value chain (1-hop neighbor)
                elif cp1 and cp2 and (self.graph.has_edge(cp1, cp2) or self.graph.has_edge(cp2, cp1)):
                    weight += 2.0

                # Same sub-sector / sub-industry value chain (e.g. rice_agriculture, agri_biotech, semiconductor)
                if sub1 and sub2 and sub1 == sub2 and sub1 != "general":
                    weight += 2.5

                # Shared named entity (companies, specific geographies, key commodities)
                shared_ents = ent1.intersection(ent2)
                if shared_ents:
                    weight += 2.5 * len(shared_ents)

                # Same GICS sub-industry
                if gics1 and gics2 and gics1 == gics2:
                    weight += 2.0

                # Note: Broad sector equality (sec1 == sec2) is NOT linked to keep communities fine-grained by sub-sector!

                if weight > 0.0:
                    proj.add_edge(e1, e2, weight=weight)

        self._cached_projection = proj
        self._dirty = False
        return proj

    QUERY_STOP_WORDS = {
        "what", "are", "the", "for", "and", "in", "how", "does", "did", "do", "with",
        "about", "from", "by", "an", "is", "of", "to", "this", "that", "there",
        "their", "then", "into", "over", "can", "could", "would", "should", "will",
        "any", "all", "has", "have", "had", "been", "being", "which", "where",
        "when", "who", "whom", "why", "not", "out", "new", "more", "most", "some",
        "such", "than", "too", "very", "opportunities", "opportunity", "impact",
        "impacts", "happening", "trend", "trends", "happened", "across",
        # Generic supply-chain words that appear in almost every event record
        # and should NOT drive sector relevance on their own:
        "supply", "disruption", "disruptions", "sector", "market", "industry",
        "value", "chain", "upstream", "downstream", "demand", "production",
    }

    def find_events_by_query(self, query: str, top_k: int = 10, min_score: float = 2.5) -> List[Dict[str, Any]]:
        """Fast keyword & entity matching across events in graph, ranked by Impact Magnitude.

        Hallucination guard:
        1. Generic supply-chain words (supply, disruption, market, …) are in QUERY_STOP_WORDS
           and never drive a match on their own.
        2. At least ONE domain-specific query token must hit a *structural* field
           (sub_sector, target_sector, matched_checkpoint). If all matches are confined
           to the free-text summary, the event is discarded — summaries are written in
           natural language and can accidentally contain query words across sectors.
        """
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", query.lower())
        q_tokens = [t for t in raw_tokens if t not in self.QUERY_STOP_WORDS and len(t) >= 2]
        if not q_tokens:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for n, data in self.graph.nodes(data=True):
            if data.get("node_type") != "Event":
                continue

            summary = (data.get("summary") or "").lower()
            cp = (data.get("matched_checkpoint") or "").lower()
            sub = (data.get("sub_sector") or "").lower()
            sec = (data.get("target_sector") or "").lower()
            tag = (data.get("tag") or "").lower()

            # Structural fields: must have at least one token match here
            structural = f"{cp} {sub} {sec}"

            match_score = 0.0
            structural_hits = 0
            for tok in q_tokens:
                pattern = rf"\b{re.escape(tok)}\b"
                struct_hit = bool(re.search(pattern, structural))
                if struct_hit:
                    structural_hits += 1
                if re.search(pattern, summary):
                    match_score += 3.0
                if re.search(pattern, cp):
                    match_score += 4.0
                if re.search(pattern, sub):
                    match_score += 3.5
                if re.search(pattern, sec):
                    match_score += 3.0
                if re.search(pattern, tag):
                    match_score += 2.0

            # Require at least one query token to hit a structural field.
            # This prevents events from unrelated sectors matching purely on
            # summary words (e.g. "supply", "shortage") that appear in every event.
            if structural_hits == 0:
                continue

            if match_score >= min_score:
                mag = float(data.get("impact_magnitude") or 1.0)
                final_score = round(match_score * (1.0 + 0.3 * mag), 2)
                scored.append((final_score, {"event_id": n, **data, "search_score": final_score}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def sync_to_neo4j(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: Optional[str] = None,
    ) -> Dict[str, int]:
        """Synchronizes the Event Graph (Events, Value-Chain Checkpoints, Roles, Entities)
        into Neo4j with rich impact properties and relationships."""
        import os
        pwd = password or os.getenv("NEO4J_PASSWORD", "neo4j")
        try:
            from neo4j import GraphDatabase
        except ImportError:
            return {"error": "neo4j package not installed"}

        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        counts = {"events": 0, "checkpoints": 0, "roles": 0, "entities": 0, "edges": 0}

        with driver.session() as session:
            # 1. Sync Checkpoints & Value Chain Flow
            for n, data in self.graph.nodes(data=True):
                if data.get("node_type") == "Checkpoint":
                    session.run(
                        """
                        MERGE (c:Checkpoint {id: $id})
                        SET c.sub_industry = $sub_industry,
                            c.stage = $stage,
                            c.description = $description,
                            c.is_macro = $is_macro
                        """,
                        id=n,
                        sub_industry=data.get("sub_industry", ""),
                        stage=data.get("stage", ""),
                        description=data.get("description", ""),
                        is_macro=bool(data.get("is_macro", False)),
                    )
                    counts["checkpoints"] += 1

            for u, v, data in self.graph.edges(data=True):
                if data.get("edge_type") == "VALUE_CHAIN_FLOW":
                    session.run(
                        """
                        MATCH (c1:Checkpoint {id: $u}), (c2:Checkpoint {id: $v})
                        MERGE (c1)-[r:VALUE_CHAIN_FLOW]->(c2)
                        SET r.relation = $relation, r.weight = $weight
                        """,
                        u=u,
                        v=v,
                        relation=data.get("relation", "downstream"),
                        weight=float(data.get("weight", 1.0)),
                    )
                    counts["edges"] += 1

            # 2. Sync Events & Impact Properties
            for n, data in self.graph.nodes(data=True):
                if data.get("node_type") == "Event":
                    session.run(
                        """
                        MERGE (e:Event {id: $id})
                        SET e.summary = $summary,
                            e.cluster = $cluster,
                            e.tag = $tag,
                            e.direction = $direction,
                            e.duration = $duration,
                            e.scope = $scope,
                            e.impact_magnitude = $impact_magnitude,
                            e.signed_impact = $signed_impact,
                            e.significance = $significance,
                            e.sub_sector = $sub_sector,
                            e.target_sector = $target_sector,
                            e.date = $date
                        """,
                        id=n,
                        summary=data.get("summary", ""),
                        cluster=data.get("cluster", ""),
                        tag=data.get("tag", ""),
                        direction=data.get("direction", ""),
                        duration=data.get("duration", ""),
                        scope=data.get("scope", ""),
                        impact_magnitude=float(data.get("impact_magnitude", 1.0)),
                        signed_impact=float(data.get("signed_impact", 0.0)),
                        significance=data.get("significance", ""),
                        sub_sector=data.get("sub_sector", ""),
                        target_sector=data.get("target_sector", ""),
                        date=data.get("date", ""),
                    )
                    counts["events"] += 1

                    # Connect Event -> Checkpoint
                    cp = data.get("matched_checkpoint")
                    if cp:
                        session.run(
                            """
                            MATCH (e:Event {id: $eid}), (c:Checkpoint {id: $cid})
                            MERGE (e)-[r:TOUCHES_CHECKPOINT]->(c)
                            """,
                            eid=n,
                            cid=cp,
                        )
                        counts["edges"] += 1

            # 3. Sync Stakeholder Roles & Impacts
            for u, v, data in self.graph.edges(data=True):
                if data.get("edge_type") == "IMPACTS_ROLE":
                    role_name = self.graph.nodes[v].get("name", v)
                    session.run(
                        """
                        MERGE (r:StakeholderRole {name: $name})
                        WITH r
                        MATCH (e:Event {id: $eid})
                        MERGE (e)-[rel:IMPACTS_ROLE]->(r)
                        SET rel.sentiment = $sentiment,
                            rel.scope = $scope,
                            rel.weight = $weight
                        """,
                        name=role_name,
                        eid=u,
                        sentiment=data.get("sentiment", "neutral"),
                        scope=data.get("scope", "direct"),
                        weight=float(data.get("weight", 1.0)),
                    )
                    counts["roles"] += 1
                    counts["edges"] += 1

                elif data.get("edge_type") == "INVOLVES_ENTITY":
                    ent_name = self.graph.nodes[v].get("name", v)
                    session.run(
                        """
                        MERGE (ent:Entity {name: $name})
                        WITH ent
                        MATCH (e:Event {id: $eid})
                        MERGE (e)-[rel:INVOLVES_ENTITY]->(ent)
                        """,
                        name=ent_name,
                        eid=u,
                    )
                    counts["entities"] += 1
                    counts["edges"] += 1

        driver.close()
        return counts


# ── Self Check (Ponytail verification) ────────────────────────────────────────

if __name__ == "__main__":
    eg = EventGraph()
    assert len(eg.graph.nodes) > 80, "Value chain checkpoints must be loaded"

    sample_r1 = {
        "main_event_summary": "Typhoon Yagi severely damages rice crops in Red River Delta causing severe harvest shortage",
        "event_status": "Started",
        "tag": "Natural disaster",
        "cluster": "objective_event_tag",
        "direction": "decrease",
        "roles": [{"role": "Producer", "sentiment": "negative", "scope": "direct"}],
        "matched_checkpoint": "rice_agriculture.cultivation",
        "target_sector": "Agriculture",
        "matched_gics_sub_industry": "Agricultural Products",
    }
    sample_r2 = {
        "main_event_summary": "Rice milling companies face surge in paddy input prices and warehouse supply shortage",
        "event_status": "Ongoing",
        "tag": "Input cost shock",
        "cluster": "supply_side_tag",
        "direction": "increase",
        "roles": [{"role": "Producer", "sentiment": "negative", "scope": "direct"}],
        "matched_checkpoint": "rice_agriculture.processing",
        "target_sector": "Agriculture",
        "matched_gics_sub_industry": "Agricultural Products",
    }

    eid1 = eg.add_event_record(sample_r1)
    eid2 = eg.add_event_record(sample_r2)

    ripple = eg.trace_ripple_effects(eid1, max_hops=2)
    assert len(ripple["downstream_stages"]) > 0, "Must trace downstream rice stages"
    assert ripple["related_events"][0]["event_id"] == eid2, "Must link to milling shortage event"

    proj = eg.build_event_projection_graph()
    assert proj.has_edge(eid1, eid2), "Events connected via value chain must project an edge"

    # Test Article chaining
    sample_article = Article(
        url="https://example.com/test-article",
        title="Sample Agricultural Article",
        original_text="Some text",
        no_save_pool={"opportunity_record": sample_r1},
    )
    art_eids = eg.ingest_articles([sample_article])
    assert len(art_eids) == 1
    assert sample_article.processed_stage == "analyzed"
    assert eg.graph.nodes[art_eids[0]]["source_url"] == "https://example.com/test-article"

    print("EventGraph & Value-Chain ripple tracing self-check PASSED!")
