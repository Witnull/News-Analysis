"""Community Detection & Value-Chain Cluster Engine.

Discovers tightly coupled event clusters using NetworkX community detection
(Louvain / Greedy Modularity) across the projected Event Correlation Graph,
profiling macro themes, stakeholder balances, and causal linkages.
"""

from typing import Optional
from collections import Counter
from typing import Any, Dict, List, Set
import networkx as nx
from networkx.algorithms.community import louvain_communities, greedy_modularity_communities

from src.modules.analysis.event_graph import EventGraph


class CommunityDetector:
    """Detects and profiles macro-thematic clusters of interconnected events."""

    def __init__(self, event_graph: EventGraph):
        self.event_graph = event_graph
        self._cached_profiles: Optional[List[Dict[str, Any]]] = None
        self._cached_resolution: float = 1.0

    def detect_communities(self, resolution: float = 1.0, force_recompute: bool = False) -> List[Dict[str, Any]]:
        """Partitions events into communities using Louvain modularity optimization.
        Returns a list of structured community cluster profiles. Cached until graph changes.
        """
        if (
            not force_recompute
            and not getattr(self.event_graph, "_dirty", True)
            and self._cached_profiles is not None
            and self._cached_resolution == resolution
        ):
            return self._cached_profiles

        proj = self.event_graph.build_event_projection_graph()
        if len(proj.nodes) == 0:
            return []

        # If graph has no edges, each node is its own cluster
        if len(proj.edges) == 0:
            raw_communities = [{n} for n in proj.nodes]
        else:
            try:
                raw_communities = list(louvain_communities(proj, weight="weight", resolution=resolution, seed=42))
            except Exception:
                raw_communities = list(greedy_modularity_communities(proj, weight="weight"))

        # Sort communities by size descending
        raw_communities.sort(key=len, reverse=True)

        profiles: List[Dict[str, Any]] = []
        for idx, member_set in enumerate(raw_communities):
            profile = self._profile_community(idx, member_set, proj)
            profiles.append(profile)

        self._cached_profiles = profiles
        self._cached_resolution = resolution
        return profiles

    def _profile_community(self, community_id: int, member_set: Set[str], proj: nx.Graph) -> Dict[str, Any]:
        member_events = []
        sectors: Counter = Counter()
        sub_sectors: Counter = Counter()
        checkpoints: Counter = Counter()
        gics_counter: Counter = Counter()
        entities: Counter = Counter()
        stakeholder_sentiment: Dict[str, Dict[str, int]] = {}

        for eid in member_set:
            edata = self.event_graph.graph.nodes[eid]
            member_events.append({"event_id": eid, **edata})

            sec = edata.get("target_sector")
            if sec and sec not in ("Other", "_", "None", None):
                sectors[sec] += 1

            sub = edata.get("sub_sector")
            if sub and sub not in ("general", "Other", "_", "None", None):
                sub_sectors[sub] += 1

            cp = edata.get("matched_checkpoint")
            if cp:
                checkpoints[cp] += 1

            gics = edata.get("matched_gics")
            if gics:
                gics_counter[gics] += 1

            # Roles and sentiments from graph edges
            for _, role_node, edge_data in self.event_graph.graph.out_edges(eid, data=True):
                if edge_data.get("edge_type") == "IMPACTS_ROLE":
                    role_name = self.event_graph.graph.nodes[role_node].get("name", role_node)
                    sent = edge_data.get("sentiment", "neutral")
                    if role_name not in stakeholder_sentiment:
                        stakeholder_sentiment[role_name] = {"positive": 0, "negative": 0, "neutral": 0}
                    stakeholder_sentiment[role_name][sent] = stakeholder_sentiment[role_name].get(sent, 0) + 1

                elif edge_data.get("edge_type") == "INVOLVES_ENTITY":
                    ent_name = self.event_graph.graph.nodes[role_node].get("name", role_node)
                    entities[ent_name] += 1

        # Sort member events by Impact Magnitude (D x S) descending
        member_events.sort(key=lambda x: float(x.get("impact_magnitude") or 1.0), reverse=True)

        # Impact Magnitude aggregation (Duration x Scope x Direction)
        total_mag = sum(float(e.get("impact_magnitude") or 1.0) for e in member_events)
        avg_mag = round(total_mag / max(len(member_events), 1), 2)
        critical_count = sum(1 for e in member_events if float(e.get("impact_magnitude") or 1.0) >= 4.5)

        # Trace internal value-chain ripple links within this community
        internal_ripples = []
        member_list = list(member_set)
        for i in range(len(member_list)):
            e1 = member_list[i]
            ripple = self.event_graph.trace_ripple_effects(e1, max_hops=2)
            for re_item in ripple.get("related_events", []):
                if re_item["event_id"] in member_set:
                    internal_ripples.append({
                        "from_event": e1,
                        "to_event": re_item["event_id"],
                        "relation": re_item["relation"],
                        "from_checkpoint": ripple.get("source_checkpoint"),
                        "to_checkpoint": re_item["matched_checkpoint"],
                    })

        # Derive an intuitive community title by SUB-SECTOR (e.g. Rice Agriculture, Semiconductor, Software Services)
        if sub_sectors:
            raw_sub = sub_sectors.most_common(1)[0][0]
            dom_sub = raw_sub.replace("_", " ").title()
        elif gics_counter:
            dom_sub = gics_counter.most_common(1)[0][0]
        elif sectors:
            dom_sub = sectors.most_common(1)[0][0]
        else:
            dom_sub = "Cross-Industry"

        dom_sec = sectors.most_common(1)[0][0] if sectors else "Multi-Sector"
        top_cps = [c.split(".")[-1].replace("_", " ") for c, _ in checkpoints.most_common(2)]
        cp_label = f" ({', '.join(top_cps)})" if top_cps else ""
        community_name = f"{dom_sub} Value-Chain Cluster{cp_label}"

        return {
            "community_id": f"cluster_{community_id + 1:02d}",
            "name": community_name,
            "size": len(member_set),
            "dominant_sub_sector": dom_sub,
            "dominant_sector": dom_sec,
            "impact_metrics": {
                "total_impact_magnitude": round(total_mag, 2),
                "avg_impact_magnitude": avg_mag,
                "critical_events_count": critical_count,
            },
            "top_checkpoints": [c for c, _ in checkpoints.most_common(4)],
            "top_gics": [g for g, _ in gics_counter.most_common(3)],
            "key_entities": [e for e, _ in entities.most_common(5)],
            "stakeholder_matrix": stakeholder_sentiment,
            "internal_value_chain_ripples": internal_ripples,
            "events": member_events,
        }


# ── Self Check (Ponytail verification) ────────────────────────────────────────

if __name__ == "__main__":
    from src.modules.analysis.event_graph import EventGraph

    eg = EventGraph()
    e1 = eg.add_event_record({
        "main_event_summary": "Tesla eliminates rare-earth magnets from Cybercab motors",
        "tag": "Material substitution",
        "direction": "decrease",
        "matched_checkpoint": "iron_steel_mineral.extraction",
        "target_sector": "Heavy Industry",
    })
    e2 = eg.add_event_record({
        "main_event_summary": "Brazil Senate approves National Strategic Critical Minerals policy",
        "tag": "National strategic project",
        "direction": "increase",
        "matched_checkpoint": "iron_steel_mineral.extraction",
        "target_sector": "Mineral",
    })
    e3 = eg.add_event_record({
        "main_event_summary": "Faraday Copper reports high-grade copper discovery in Arizona",
        "tag": "Resource discovery",
        "direction": "increase",
        "matched_checkpoint": "iron_steel_mineral.exploration",
        "target_sector": "Mineral",
    })

    detector = CommunityDetector(eg)
    communities = detector.detect_communities()
    assert len(communities) > 0, "Must detect at least one community"
    assert "Mineral" in communities[0]["name"] or "Heavy Industry" in communities[0]["name"]

    print(f"CommunityDetector self-check PASSED! Found {len(communities)} community clusters.")
