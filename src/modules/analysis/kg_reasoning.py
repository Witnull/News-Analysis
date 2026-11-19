from typing import Dict, List, Optional
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# 1. Topic / theme report generation
# ─────────────────────────────────────────────────────────────────────────────

class TopicReportGenerator:
    def __init__(self, graph_store, llm):
        self.graph_store = graph_store
        self.llm = llm

    def _match_communities(self, topic_keywords: List[str]) -> List[Dict]:
        """Rank existing communities by keyword overlap with the requested topic.
        Falls back to entity/keyword search across ALL facts if no community
        matches well (e.g. graph hasn't been re-clustered since last ingest)."""
        communities = self.graph_store.get_community_summaries()
        scored = []
        for c in communities:
            text = f"{c.get('summary', '')} {' '.join(c.get('members', []))}".lower()
            overlap = sum(1 for kw in topic_keywords if kw.lower() in text)
            if overlap > 0:
                scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def _recent_facts_for_members(self, members: List[str], since: Optional[str],
                                   limit_per_member: int = 15) -> List[Dict]:
        facts = []
        for m in members:
            hits = self.graph_store.get_entity_summary(m, limit=limit_per_member, hops=1,
                                                          include_expired=False)
            for h in hits:
                if since and h.get("valid_at") and h["valid_at"] < since:
                    continue
                facts.append(h)
        return facts

    def generate_report(self, topic: str, keywords: List[str], days_back: int = 90) -> str:
        since = (datetime.now() - timedelta(days=days_back)).date().isoformat()
        matched = self._match_communities(keywords)

        if not matched:
            # Fallback: no community structure covers this topic at all.
            return (f"[No knowledge-graph coverage found for '{topic}'. "
                     f"The graph has not ingested any sources discussing this topic, "
                     f"or community detection has not been (re)run since ingestion. "
                     f"Cannot compile a report — this would require sourcing new articles first.]")

        sections = []
        for community in matched[:3]:  # top 3 most relevant communities
            members = community.get("members", [])
            facts = self._recent_facts_for_members(members, since=since)
            if not facts:
                continue
            facts_text = "\n".join(
                f"- {f.get('subject')} {f.get('predicate','').replace('_',' ')} {f.get('object')} "
                f"(valid_at: {f.get('valid_at','unknown')})"
                for f in facts
            )
            section = self._synthesize_section(topic, community.get("summary", ""), facts_text)
            sections.append(section)

        if not sections:
            return (f"[Found thematically related communities for '{topic}', but no facts "
                     f"dated within the last {days_back} days. The graph's knowledge on this "
                     f"topic may be stale — recent sources have not been ingested.]")

        return "\n\n".join(f"### {s['heading']}\n{s['body']}" for s in sections)

    def _synthesize_section(self, topic: str, community_context: str, facts_text: str) -> Dict:
        schema = {
            "type": "object",
            "properties": {"heading": {"type": "string"}, "body": {"type": "string"}},
            "required": ["heading", "body"],
        }
        system_msg = (
            "You write one section of a knowledge-graph-grounded report. "
            "Use ONLY the provided facts (with their valid_at dates). "
            "Give the section a short heading and a factual, dated body. "
            "Do not speculate beyond what the facts state. If facts are sparse, say so."
        )
        user_msg = f"Report topic: {topic}\nCommunity context: {community_context}\n\nFacts:\n{facts_text}"
        raw = self.llm.chat(user_msg, system_msg, schema_override=schema)
        parsed, _ = self.llm._parse_llm_json(raw)
        return {"heading": parsed.get("heading", topic), "body": parsed.get("body", facts_text)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Claim verification / argumentative fact-checking
# ─────────────────────────────────────────────────────────────────────────────

class ClaimVerifier:
    def __init__(self, graph_store, retriever, llm):
        self.graph_store = graph_store
        self.retriever = retriever
        self.llm = llm

    def _parse_claim(self, claim: str) -> Dict:
        schema = {
            "type": "object",
            "properties": {
                "entities": {"type": "array", "items": {"type": "string"}},
                "predicate": {"type": "string"},
                "claimed_value": {"type": "string", "description": "The specific outcome/direction the user asserts (e.g. 'decrease', 'increase', 'stable')"},
                "implied_timeframe": {"type": "string", "description": "e.g. 'this year', '2026', 'Q3'"},
            },
            "required": ["entities", "predicate", "claimed_value", "implied_timeframe"],
        }
        system_msg = (
            "Decompose the user's ASSERTION (not a question) into its structural claim: "
            "which entities it's about, the relationship/attribute being claimed, the specific "
            "direction/value claimed, and the implied timeframe."
        )
        raw = self.llm.chat(f"Assertion: {claim}", system_msg, schema_override=schema)
        parsed, _ = self.llm._parse_llm_json(raw)
        return parsed

    def verify(self, claim: str, recipe: str = "temporal") -> Dict:
        parsed_claim = self._parse_claim(claim)
        entities = parsed_claim.get("entities", [])

        if not entities:
            return {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "argument": "Could not identify a specific, checkable subject in this statement.",
                "evidence": [],
            }

        evidence = self.retriever.search(claim, entities=entities, recipe=recipe, top_k=10)

        if not evidence:
            return {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "argument": (
                    f"The knowledge graph has no ingested facts about "
                    f"{', '.join(entities)}. This claim cannot be verified or "
                    f"disputed — it requires sourcing relevant data first, not "
                    f"an assumption either way."
                ),
                "evidence": [],
            }

        return self._argue_verdict(claim, parsed_claim, evidence)

    def _argue_verdict(self, claim: str, parsed_claim: Dict, evidence: List[Dict]) -> Dict:
        evidence_text = "\n".join(
            f"- {e.get('subject','')} {str(e.get('predicate','')).replace('_',' ')} {e.get('object','')} "
            f"(valid_at: {e.get('valid_at','?')}, invalid_at: {e.get('invalid_at') or 'still active'}, "
            f"source-confidence: {e.get('hit_count', e.get('rerank_score', '?'))})"
            for e in evidence
        )
        schema = {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED", "PARTIALLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"]},
                "argument": {"type": "string"},
            },
            "required": ["verdict", "argument"],
        }
        system_msg = (
            "You are a strict fact-checker arguing against or for a user's claim using ONLY "
            "the cited knowledge-graph evidence below. Rules:\n"
            "1. If the evidence explicitly states the opposite direction/value of the claim, verdict=CONTRADICTED "
            "and explain the discrepancy with specific cited numbers/dates.\n"
            "2. If the evidence aligns, verdict=SUPPORTED.\n"
            "3. If evidence is mixed or covers only part of the claim, verdict=PARTIALLY_SUPPORTED.\n"
            "4. If evidence is too indirect/old/off-topic to judge, verdict=INSUFFICIENT_EVIDENCE — "
            "do NOT force a verdict beyond what the facts support.\n"
            "5. Never invent facts not present in the evidence list."
        )
        user_msg = f"User's claim: {claim}\nParsed claim structure: {parsed_claim}\n\nEvidence:\n{evidence_text}"
        raw = self.llm.chat(user_msg, system_msg, schema_override=schema)
        parsed, _ = self.llm._parse_llm_json(raw)
        return {
            "verdict": parsed.get("verdict", "INSUFFICIENT_EVIDENCE"),
            "argument": parsed.get("argument", "Could not form a grounded argument."),
            "evidence": evidence,
        }