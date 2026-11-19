"""Knowledge Graph Retriever & Interactive KG Chat Engine.

Provides multi-hop value-chain retrieval, sub-sector community filtering,
quantitative impact ranking (Duration x Scope x Direction), and God's Eye
opportunity generation over both in-memory EventGraph and active Neo4j.
"""

from typing import Any, Dict, List, Optional
import re

from src.modules.analysis.event_graph import EventGraph
from src.modules.analysis.community_engine import CommunityDetector
from src.modules.analysis.god_eye_reasoning import GodEyeSynthesizer


class KGRetriever:
    """Retrieval engine for events, value-chain ripple paths, and communities."""

    def __init__(self, event_graph: EventGraph):
        self.event_graph = event_graph
        self.community_detector = CommunityDetector(event_graph)
        self.synthesizer = GodEyeSynthesizer(event_graph)
        self._cached_report: Optional[Dict[str, Any]] = None

    def search_events(
        self,
        query: str,
        min_impact: float = 0.0,
        sub_sector: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Searches events in the graph by query text, filtered by minimum impact magnitude
        and optional sub-sector."""
        results = self.event_graph.find_events_by_query(query, top_k=top_k * 2)

        filtered = []
        for r in results:
            mag = float(r.get("impact_magnitude") or 1.0)
            if mag < min_impact:
                continue
            if sub_sector and sub_sector.lower() not in (r.get("sub_sector") or "").lower():
                continue
            filtered.append(r)

        return filtered[:top_k]

    def get_value_chain_trace(self, event_id: str, max_hops: int = 3) -> Dict[str, Any]:
        """Returns the full downstream value-chain ripple trajectory for an event."""
        return self.event_graph.trace_ripple_effects(event_id, max_hops=max_hops)

    def retrieve_opportunities(
        self,
        query: Optional[str] = None,
        min_impact: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Runs community detection and multi-aspect synthesis, returning opportunities
        filtered by query keywords and impact significance.

        Hallucination guard:
        - When query has 3+ meaningful terms, at least 2 must match the combined
          opportunity text (not just a single generic word like 'supply').
        - Additionally, at least 1 query token must match the cluster's structural
          fields (cluster_name or dominant_sub_sector) to ensure sector alignment.
        """
        if getattr(self.event_graph, "_dirty", True) or self._cached_report is None:
            self._cached_report = self.synthesizer.generate_god_eye_report()
        report = self._cached_report
        all_opps = []

        stop_words = getattr(self.event_graph, "QUERY_STOP_WORDS", set())
        raw_terms = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", (query or "").lower())
        q_terms = [t for t in raw_terms if t not in stop_words and len(t) >= 2]

        # How many terms must match the combined opportunity text?
        # With 3+ query terms, require 2 matches (avoids single-word pollution).
        # With 1-2 terms, require 1 (preserve recall for narrow queries).
        min_overlap = 2 if len(q_terms) >= 3 else 1

        for comm in report.get("community_analyses", []):
            cname = comm.get("cluster_name", "")
            dom_sub = comm.get("dominant_sub_sector", "")
            events = comm.get("events", [])
            max_ev_mag = max((float(e.get("impact_magnitude") or 1.0) for e in events), default=1.0)

            if max_ev_mag < min_impact:
                continue

            # Sector/cluster structural alignment: at least one query token must
            # match the cluster's structural identity fields (cluster name or dominant
            # sub-sector). This prevents generic terms in opportunity thesis text
            # from pulling in completely unrelated sector clusters.
            structural_cluster = f"{cname} {dom_sub}".lower()
            if q_terms:
                struct_overlap = sum(1 for t in q_terms if re.search(rf"\b{re.escape(t)}\b", structural_cluster))
                if struct_overlap == 0:
                    continue

            for opp in comm.get("hidden_opportunities", []):
                combined_txt = f"{cname} {dom_sub} {opp.get('type', '')} {opp.get('thesis', '')} {opp.get('actionable_signal', '')}".lower()

                if q_terms:
                    overlap = sum(1 for t in q_terms if re.search(rf"\b{re.escape(t)}\b", combined_txt))
                    if overlap < min_overlap:
                        continue

                all_opps.append({
                    "cluster_name": cname,
                    "dominant_sub_sector": dom_sub,
                    "max_impact_magnitude": max_ev_mag,
                    "opportunity_type": opp.get("type"),
                    "thesis": opp.get("thesis"),
                    "actionable_signal": opp.get("actionable_signal"),
                    "value_chain_ripples": comm.get("value_chain_ripples", [])[:2],
                    "winners": comm.get("winners", []),
                    "losers": comm.get("losers", []),
                })

        all_opps.sort(key=lambda x: x["max_impact_magnitude"], reverse=True)
        return all_opps


class KGChat:
    """Conversational interface to the Event Knowledge Graph."""

    def __init__(self, event_graph: EventGraph):
        self.retriever = KGRetriever(event_graph)
        self.event_graph = event_graph

    def ask(self, question: str) -> Dict[str, Any]:
        """Interprets question, retrieves matched events, traces value-chain ripples,
        and provides structured factual answers with actionable opportunity thesis."""
        q_clean = question.strip()

        # 1. Search relevant events (filtered for actual keyword relevance)
        events = self.retriever.search_events(q_clean, top_k=5)

        # 2. Retrieve matched opportunities (also domain-gated in retrieve_opportunities)
        opportunities = self.retriever.retrieve_opportunities(query=q_clean, min_impact=1.0)

        # Honest no-match response: never dump irrelevant sectors if nothing matches
        if not events and not opportunities:
            return {
                "query": q_clean,
                "answer": (
                    f"### Analysis for Query: '{q_clean}'\n\n"
                    f"**No directly matched events or value-chain opportunities found in the current Knowledge Graph.**\n"
                    f"The graph currently does not contain ingested records covering these specific entities or sectors. "
                    f"Please ingest relevant news or policy articles to analyze this value chain."
                ),
                "matched_events": [],
                "ripple_path": None,
                "causal_steps": [],
                "opportunities": [],
            }

        # 3. Trace ripple chains for top event
        ripple_path = None
        causal_steps = []
        if events:
            top_eid = events[0].get("event_id")
            if top_eid:
                ripple = self.retriever.get_value_chain_trace(top_eid, max_hops=3)
                stages = [ripple.get("source_checkpoint")] + [d["checkpoint"] for d in ripple.get("downstream_stages", [])[:3]]
                ripple_path = " -> ".join([s for s in stages if s])

                # Build context-aware causal steps
                direction = events[0].get("direction", "neutral")
                mag = events[0].get("impact_magnitude", 1.0)
                summary = events[0].get("summary", "")
                tag_txt = (events[0].get("tag") or "").lower()
                comb_txt = (summary + " " + tag_txt).lower()

                if direction == "decrease":
                    causal_steps = [
                        f"Step 1 (Shock): {summary[:80]}... [Impact Magnitude: {mag}]",
                        f"Step 2 (Transmission): Raw material contraction along {ripple_path}",
                        "Step 3 (Market Effect): Input scarcity and margin compression for downstream processors",
                        "Step 4 (Opportunity): Unaffected substitute suppliers & inventory holders gain pricing power",
                    ]
                elif any(w in comb_txt for w in ["debt", "loan", "leverage", "borrow", "phình to", "credit"]):
                    causal_steps = [
                        f"Step 1 (Credit Shock): {summary[:80]}... [Impact Magnitude: {mag}]",
                        f"Step 2 (Transmission): Rising financial leverage along {ripple_path}",
                        "Step 3 (Market Effect): Refinancing pressures and capital constraints for high-leverage borrowers",
                        "Step 4 (Opportunity): Distressed asset acquirers and workout advisors capitalize on debt restructuring",
                    ]
                elif any(w in comb_txt for w in ["cdn", "cloud", "software", "platform", "adoption", "employ"]):
                    causal_steps = [
                        f"Step 1 (Adoption Event): {summary[:80]}... [Impact Magnitude: {mag}]",
                        f"Step 2 (Transmission): Enterprise infrastructure standardization along {ripple_path}",
                        "Step 3 (Market Effect): Network effects and ecosystem lock-in across downstream enterprise clients",
                    ]
                else:
                    causal_steps = [
                        f"Step 1 (Event): {summary[:80]}... [Impact Magnitude: {mag}]",
                        f"Step 2 (Transmission): Operational throughput along {ripple_path}",
                        "Step 3 (Market Effect): Value chain realignment across downstream participants",
                    ]

        # 4. Formulate structured response
        response_lines = [f"### Analysis for Query: '{q_clean}'\n"]

        if events:
            response_lines.append(f"**Found {len(events)} Key Interconnected Events:**")
            for e in events[:3]:
                sig = e.get("significance", "STANDARD")
                mag = e.get("impact_magnitude", 1.0)
                dur = e.get("duration", "N/A")
                scp = e.get("scope", "N/A")
                response_lines.append(
                    f"- **[{sig} | Mag: {mag} (Dur: {dur}, Scope: {scp})]** {e.get('summary')} (Sub-Sector: {e.get('sub_sector')})"
                )
            response_lines.append("")

        if ripple_path:
            response_lines.append(f"**Value-Chain Ripple Trajectory:**\n`{ripple_path}`")
            if causal_steps:
                for step in causal_steps:
                    response_lines.append(f"  • {step}")
            response_lines.append("")

        if opportunities:
            response_lines.append(f"**Synthesized Potential Opportunities ({len(opportunities)} discovered):**")
            for opp in opportunities[:2]:
                response_lines.append(f"#### [{opp['opportunity_type']}] in {opp['cluster_name']}")
                response_lines.append(f"**Thesis:** {opp['thesis']}")
                response_lines.append(f"**Actionable Signal:** {opp['actionable_signal']}")
                if opp.get("winners"):
                    response_lines.append(f"**Key Winners:** {', '.join(opp['winners'][:3])}")
                if opp.get("losers"):
                    response_lines.append(f"**Key Losers:** {', '.join(opp['losers'][:3])}")
                response_lines.append("")
        else:
            response_lines.append("No direct multi-aspect opportunities converged for this specific query.")

        formatted_answer = "\n".join(response_lines)

        return {
            "query": q_clean,
            "answer": formatted_answer,
            "matched_events": events,
            "ripple_path": ripple_path,
            "causal_steps": causal_steps,
            "opportunities": opportunities,
        }


# ── Self Check (Ponytail verification) ────────────────────────────────────────

if __name__ == "__main__":
    from src.modules.analysis.event_graph import EventGraph

    eg = EventGraph()
    eg.add_event_record({
        "main_event_summary": "Typhoon damages Red River Delta rice crops causing paddy shortage",
        "tag": "Natural disaster",
        "cluster": "objective_event_tag",
        "direction": "decrease",
        "matched_checkpoint": "rice_agriculture.cultivation",
        "target_sector": "Agriculture",
    })
    eg.add_event_record({
        "main_event_summary": "Rice milling companies report input shortage and higher costs",
        "tag": "Input cost shock",
        "cluster": "supply_side_tag",
        "direction": "increase",
        "matched_checkpoint": "rice_agriculture.processing",
        "target_sector": "Agriculture",
    })

    retriever = KGRetriever(eg)
    events = retriever.search_events("rice shortage", top_k=5)
    assert len(events) >= 1, "Must match rice events"

    opps = retriever.retrieve_opportunities("rice", min_impact=1.0)
    assert len(opps) >= 1, "Must retrieve at least one opportunity"

    chat = KGChat(eg)
    ans = chat.ask("What are the impacts of the rice shortage?")
    assert "Analysis for Query" in ans["answer"]
    assert len(ans["matched_events"]) >= 1

    print("KGRetriever & KGChat self-check PASSED!")
