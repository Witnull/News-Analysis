"""God's Eye View Multi-Aspect Opportunity Synthesizer.

Combines Macro Shocks, Supply Dynamics, and Demand Shifts across Value-Chain
Community Clusters to uncover multi-hop causal ripple effects and hidden
investment/operational opportunities.
"""

from typing import Any, Dict, List, Optional
from src.modules.analysis.event_graph import EventGraph
from src.modules.analysis.community_engine import CommunityDetector


class GodEyeSynthesizer:
    """Synthesizes high-conviction, multi-aspect economic opportunities
    from connected events and supply chain value-chain ripples."""

    def __init__(self, event_graph: EventGraph, llm: Optional[Any] = None):
        self.event_graph = event_graph
        self.community_detector = CommunityDetector(event_graph)
        self.llm = llm

    def analyze_community(self, community_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Performs multi-aspect factual analysis and ripple effect synthesis
        on a single community cluster (L1 Community Summary)."""
        events = community_profile.get("events", [])
        name = community_profile.get("name", "Cluster")

        # 1. Categorize events by aspect
        macro_policy: List[Dict] = []
        supply_side: List[Dict] = []
        demand_side: List[Dict] = []
        corporate_moves: List[Dict] = []

        for e in events:
            clust = e.get("cluster")
            if clust in ("objective_event_tag", "policy_change_tag") or e.get("target_sector") == "Macro":
                macro_policy.append(e)
            elif clust == "supply_side_tag":
                supply_side.append(e)
            elif clust == "demand_side_tag":
                demand_side.append(e)
            else:
                corporate_moves.append(e)

        # 2. Trace Multi-Hop Value Chain Ripple Effects
        ripple_chains: List[Dict[str, Any]] = []
        for e in events:
            eid = e.get("event_id")
            if not eid:
                continue
            ripple = self.event_graph.trace_ripple_effects(eid, max_hops=3)
            downstream = ripple.get("downstream_stages", [])
            related = ripple.get("related_events", [])

            if downstream or related:
                chain_summary = self._synthesize_ripple_chain(e, downstream, related)
                ripple_chains.append(chain_summary)

        # 3. Stakeholder Winners vs Losers Analysis
        matrix = community_profile.get("stakeholder_matrix", {})
        winners: List[str] = []
        losers: List[str] = []
        for role, sents in matrix.items():
            pos = sents.get("positive", 0)
            neg = sents.get("negative", 0)
            if pos > neg:
                winners.append(f"{role} (+{pos})")
            elif neg > pos:
                losers.append(f"{role} (-{neg})")

        # 4. Extract Hidden Opportunities (L1 level)
        opportunities = self._extract_hidden_opportunities(
            macro_policy, supply_side, demand_side, corporate_moves, ripple_chains
        )

        return {
            "cluster_id": community_profile.get("community_id"),
            "cluster_name": name,
            "event_count": len(events),
            "dominant_sector": community_profile.get("dominant_sector"),
            "dominant_sub_sector": community_profile.get("dominant_sub_sector"),
            "aspect_breakdown": {
                "macro_policy_count": len(macro_policy),
                "supply_side_count": len(supply_side),
                "demand_side_count": len(demand_side),
                "corporate_moves_count": len(corporate_moves),
            },
            "winners": winners,
            "losers": losers,
            "value_chain_ripples": ripple_chains,
            "hidden_opportunities": opportunities,
        }

    def _synthesize_ripple_chain(
        self,
        source_event: Dict[str, Any],
        downstream_stages: List[Dict[str, Any]],
        related_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Constructs a deterministic multi-hop ripple explanation with explicit causal steps."""
        src_cp = source_event.get("matched_checkpoint") or "upstream_node"
        src_summary = source_event.get("summary") or ""
        src_direction = source_event.get("direction") or "neutral"
        tag = (source_event.get("tag") or "").lower()

        stages_text = [src_cp] + [d["checkpoint"] for d in downstream_stages[:3]]
        path_str = " -> ".join(stages_text)

        causal_steps: List[str] = []
        is_disruption = src_direction == "decrease" or any(w in (src_summary + " " + tag).lower() for w in ["storm", "typhoon", "disaster", "flood", "drought", "shortage", "halt", "loss", "ban"])

        if is_disruption:
            causal_steps.append(f"1. Upstream Shock [{src_cp}]: Disruption/contraction reduces available raw output ({src_summary[:60]}...).")
            if len(downstream_stages) >= 1:
                stage1 = downstream_stages[0]["checkpoint"]
                causal_steps.append(f"2. Processing Bottleneck [{stage1}]: Intermediate processors experience severe raw material shortages and surging procurement costs.")
            if len(downstream_stages) >= 2:
                stage2 = downstream_stages[1]["checkpoint"]
                causal_steps.append(f"3. Distribution/Inventory Squeeze [{stage2}]: Wholesale channels face depleted inventories and shipment rationing.")
            if len(downstream_stages) >= 3:
                stage3 = downstream_stages[2]["checkpoint"]
                causal_steps.append(f"4. Market Price Surge [{stage3}]: End-product spot and retail prices spike due to systemic supply deficit.")
            else:
                causal_steps.append(f"4. Market Price Surge: End-product spot and retail prices increase due to supply deficit.")
            causal_steps.append(f"5. Hidden Opportunity: Unaffected alternative producers gain instant pricing power; pre-shock inventory holders capture windfall revaluation; emergency recovery suppliers benefit.")
            transmission = f"Supply contraction at [{src_cp}] propagates downstream through value chain stages, causing input shortages and driving finished goods price escalation."
        elif src_direction == "increase":
            lower_comb = (src_summary + " " + tag).lower()
            if any(w in lower_comb for w in ["debt", "loan", "borrow", "leverage", "phình to", "credit"]):
                causal_steps.append(f"1. Leverage Escalation [{src_cp}]: Borrowing surge expands financial obligations ({src_summary[:60]}...).")
                if downstream_stages:
                    causal_steps.append(f"2. Debt Service Absorption [{downstream_stages[0]['checkpoint']}]: Escalating interest commitments absorb corporate operating liquidity.")
                causal_steps.append("3. Refinancing Exposure: Downstream solvency depends on cash-flow recovery or asset workouts.")
                transmission = f"Leverage expansion at [{src_cp}] intensifies refinancing exposure along the capital chain."
            elif any(w in lower_comb for w in ["cdn", "cloud", "software", "platform", "adoption", "employ"]):
                causal_steps.append(f"1. Platform Consolidation [{src_cp}]: Increased adoption concentrates front-end deployment ({src_summary[:60]}...).")
                if downstream_stages:
                    causal_steps.append(f"2. Infrastructure Standardization [{downstream_stages[0]['checkpoint']}]: Downstream clients standardize on primary platform architecture.")
                causal_steps.append("3. Network Security & Routing: Concentrated provider status drives operational stability and vendor lock-in.")
                transmission = f"Platform adoption at [{src_cp}] standardizes enterprise infrastructure."
            else:
                causal_steps.append(f"1. Upstream Expansion [{src_cp}]: Capacity addition or policy stimulus scales throughput ({src_summary[:60]}...).")
                if downstream_stages:
                    causal_steps.append(f"2. Downstream Feed [{downstream_stages[0]['checkpoint']}]: Expanded throughput lowers unit processing costs and stimulates processing utilization.")
                causal_steps.append("3. Value Chain Capture: Downstream integrators capture volume growth and operational leverage.")
                transmission = f"Capacity/stimulus expansion at [{src_cp}] drives throughput growth and economies of scale downstream."
        else:
            causal_steps.append(f"1. Neutral Event [{src_cp}]: Structural realignment or monitoring along value chain.")
            transmission = f"Informational signal tracked across value chain checkpoints."

        return {
            "trigger_event": src_summary,
            "value_chain_path": path_str,
            "transmission_logic": transmission,
            "causal_steps": causal_steps,
            "co_located_events": [r.get("summary") for r in related_events],
        }

    def _extract_hidden_opportunities(
        self,
        macro_policy: List[Dict],
        supply_side: List[Dict],
        demand_side: List[Dict],
        corporate_moves: List[Dict],
        ripple_chains: List[Dict],
    ) -> List[Dict[str, str]]:
        """Synthesizes high-conviction opportunities across multi-event convergence,
        extracting non-obvious asymmetric plays (pricing power, arbitrage, substitution)."""
        opps: List[Dict[str, str]] = []

        # Opportunity Pattern A: Upstream Shock / Storm / Disruption -> Alternative Supply & Inventory Windfall
        shortage_events = [
            e for e in supply_side + macro_policy
            if e.get("direction") == "decrease"
            or any(w in (e.get("summary") or "").lower() for w in ["storm", "typhoon", "disaster", "flood", "drought", "shortage", "halt", "loss", "ban", "tariff"])
        ]
        for trig in shortage_events[:2]:
            summary_txt = trig.get("summary", "")
            cp = trig.get("matched_checkpoint", "upstream")
            sector = trig.get("target_sector", "Commodities")

            opps.append({
                "type": "Upstream Disruption & Alternative Pricing Power",
                "thesis": (
                    f"Shock in '{cp}' ({summary_txt[:70]}...) restricts regional supply, triggering downstream input shortages "
                    f"and commodity price surges. Operators in unaffected regions and holders of existing inventories gain immediate pricing power."
                ),
                "actionable_signal": (
                    f"1. Long alternative/unaffected producers in {sector} capturing displaced demand.\n"
                    f"2. Overweight warehouse/distribution operators holding pre-crisis inventory at low basis.\n"
                    f"3. Anticipate post-shock recovery wave (replanting inputs, restoration equipment)."
                ),
            })

        # Opportunity Pattern B: Policy Stimulus / Mandate -> First-Mover Infrastructure & Subsidies
        # Must be an ACTUAL policy event or stimulus, NOT a credit crisis or debt surge
        policy_events = [
            e for e in macro_policy
            if (e.get("cluster") == "policy_change_tag" or any(w in (e.get("summary") or "").lower() for w in ["stimulus", "subsid", "tax incentive", "infrastructure plan", "development plan", "public investment", "master plan"]))
            and not any(w in (e.get("summary") or "").lower() for w in ["debt", "default", "crisis", "overhang", "phình to", "ratio rises", "disaster"])
            and e.get("tag") not in ("Debt restructuring", "Loan default", "Bond default", "Natural disaster", "Credit crunch/liquidity crisis")
        ]
        for pol in policy_events[:2]:
            opps.append({
                "type": "Policy Mandate & Capital Allocation",
                "thesis": (
                    f"Government approval/stimulus ({pol.get('summary', '')[:70]}...) unlocks subsidization, regulatory tailwinds, "
                    f"and capital expenditure across the '{pol.get('target_sector', 'relevant')}' value chain."
                ),
                "actionable_signal": "Position into first-mover project developers, certified equipment contractors, and strategic concession holders.",
            })

        # Opportunity Pattern C: Material Substitution & Tech Innovation
        sub_events = [
            e for e in demand_side + supply_side
            if e.get("tag") in ("Material substitution", "Process innovation", "Automation/efficiency upgrade", "Technology breakthrough/patent")
            or any(w in (e.get("summary") or "").lower() for w in ["substitut", "patent", "breakthrough", "efficiency", "automation", "replace"])
        ]
        for tech in sub_events[:2]:
            opps.append({
                "type": "Cost Deflation & Disruption Arbitrage",
                "thesis": (
                    f"Innovation ({tech.get('summary', '')[:70]}...) bypasses high-cost bottleneck materials, "
                    f"expanding gross margins for early enterprise adopters while threatening legacy incumbent market share."
                ),
                "actionable_signal": "Long adoption enablers and patent holders; short or underweight legacy component providers.",
            })

        # Opportunity Pattern D: Corporate Debt Restructuring & Turnarounds
        debt_events = [
            e for e in corporate_moves + macro_policy
            if e.get("tag") in ("Refinancing/debt restructuring", "Debt restructuring", "Loan default", "Bond default", "Credit rating change", "Credit crunch/liquidity crisis")
            or any(w in (e.get("summary") or "").lower() for w in ["debt", "recapitaliz", "bond redemption", "turnaround", "nợ vay", "leverage", "phình to", "borrowing"])
        ]
        for corp in debt_events[:2]:
            summary_txt = corp.get("summary", "")
            direction = corp.get("direction", "neutral")
            is_surging_debt = direction == "increase" or any(w in summary_txt.lower() for w in ["phình to", "rise", "surge", "swell", "climb", "high", "overhang", "default"])

            if is_surging_debt:
                opps.append({
                    "type": "Distressed Credit Workout & Refinancing Squeeze",
                    "thesis": (
                        f"Rising leverage and refinancing obligations ({summary_txt[:70]}...) intensify debt service pressures. "
                        f"Creates distressed asset acquisition opportunities for well-capitalized players and workout demand for restructuring advisors."
                    ),
                    "actionable_signal": "Position into distressed debt/workout specialists and cash-rich acquirers; underweight high-leverage issuers facing refinancing walls.",
                })
            else:
                opps.append({
                    "type": "Credit Turnaround & Multiple Expansion",
                    "thesis": (
                        f"Significant balance sheet de-risking ({summary_txt[:70]}...) eliminates structural debt overhang, "
                        f"freeing operating cash flows for capital returns and equity multiple re-rating."
                    ),
                    "actionable_signal": "High-conviction value rerating in turnarounds concluding multi-year debt workouts.",
                })

        # Default multi-aspect synthesis if no specific pattern matched
        if not opps and (macro_policy or supply_side or demand_side):
            opps.append({
                "type": "Cross-Aspect Value Chain Alignment",
                "thesis": "Convergence of multi-faceted industry developments creates asymmetric risk/reward along value chain transit checkpoints.",
                "actionable_signal": "Monitor downstream inventory absorption and procurement pricing.",
            })

        return opps

    def generate_god_eye_report(self, use_llm: bool = False) -> Dict[str, Any]:
        """Runs end-to-end community detection and hierarchical synthesis:
        - L1: Community cluster profiles & value-chain ripple paths
        - L2: God's Eye meta-synthesis ('summary of summaries' across clusters)
        """
        communities = self.community_detector.detect_communities()
        l1_analyses = [self.analyze_community(c) for c in communities]

        total_opps = sum(len(a.get("hidden_opportunities", [])) for a in l1_analyses)

        # ── L2 Meta-Synthesis (Summary of Summaries) ─────────────────────────
        macro_themes = [f"{a['cluster_name']} ({a['dominant_sector']}): {a['event_count']} events" for a in l1_analyses]
        all_winners = [w for a in l1_analyses for w in a.get("winners", [])]
        all_losers = [l for a in l1_analyses for l in a.get("losers", [])]

        cross_cluster_opps = []
        for a in l1_analyses:
            for opp in a.get("hidden_opportunities", []):
                cross_cluster_opps.append({
                    "origin_cluster": a["cluster_name"],
                    "dominant_sector": a["dominant_sector"],
                    **opp,
                })

        l2_executive_summary = (
            f"God's Eye View across {len(communities)} active clusters ({len(cross_cluster_opps)} asymmetric opportunities). "
            f"Dominant themes: {'; '.join(macro_themes[:3]) if macro_themes else 'None'}."
        )

        # Optional LLM-driven L2 meta-synthesis
        if use_llm and self.llm and hasattr(self.llm, "chat"):
            try:
                prompt = (
                    f"Synthesize an executive God's Eye macro report (summary of summaries) from these cluster findings:\n"
                    f"Clusters: {macro_themes}\n"
                    f"Top Winners: {all_winners[:6]}\n"
                    f"Top Losers: {all_losers[:6]}\n"
                    f"Opportunities: {[o.get('type') for o in cross_cluster_opps[:6]]}"
                )
                l2_executive_summary = self.llm.chat(prompt, "You are a senior macro intelligence strategist. Summarize cross-cluster implications in 3 concise bullet points.")
            except Exception:
                pass

        return {
            "total_communities": len(communities),
            "total_opportunities_discovered": total_opps,
            "l2_god_eye_summary": l2_executive_summary,
            "cross_cluster_opportunities": cross_cluster_opps,
            "community_analyses": l1_analyses,
        }


# ── Self Check (Ponytail verification) ────────────────────────────────────────

if __name__ == "__main__":
    from src.modules.analysis.event_graph import EventGraph

    eg = EventGraph()
    eg.add_event_record({
        "main_event_summary": "Typhoon Yagi severely damages Red River Delta rice crops causing harvest drop",
        "tag": "Natural disaster",
        "cluster": "objective_event_tag",
        "direction": "decrease",
        "matched_checkpoint": "rice_agriculture.cultivation",
        "target_sector": "Agriculture",
    })
    eg.add_event_record({
        "main_event_summary": "Rice milling companies face surge in paddy input prices and export supply shortage",
        "tag": "Input cost shock",
        "cluster": "supply_side_tag",
        "direction": "increase",
        "matched_checkpoint": "rice_agriculture.processing",
        "target_sector": "Agriculture",
    })

    synthesizer = GodEyeSynthesizer(eg)
    report = synthesizer.generate_god_eye_report()

    assert report["total_communities"] >= 1
    assert report["total_opportunities_discovered"] >= 1
    first_comm = report["community_analyses"][0]
    assert any("rice_agriculture.cultivation" in r["value_chain_path"] for r in first_comm["value_chain_ripples"])
    print("GodEyeSynthesizer & Multi-Aspect Opportunity Extraction self-check PASSED!")
