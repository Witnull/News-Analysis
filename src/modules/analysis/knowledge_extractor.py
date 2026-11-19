from typing import Any, Dict, List, Optional, Tuple
import re

from src.modules.analysis.kg_models import EdgeType, PredicateNormalizer, Fact, Entity
from src.modules.store.kg_store import Neo4jGraphStore
from src.core.types import Article, GeneralLLMConfig, LLMConfig, GENConfig
from src.modules.llm.llmBase import APIConfig, LLM_Base
# ── 3. Prompts & Config ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Knowledge Graph builder. Extract structured facts from numbered sentences.

INPUT FORMAT: Each sentence is prefixed with [n], e.g. "[1] Tesla announced a new factory."

TASK: Extract entities and (Subject, Predicate, Object) triples that will construct the structure of a comprehensive knowledge graph about the text.

## OUTPUT (JSON, exactly 3 keys; "entities" and "triples" are SINGLE-LINE strings, no \\n):
"main_topic": one sentence, what the article is about.
"entities": entities separated by ";", each: name|TYPE|L1>L2>L3|aliases|refs
"triples": triples separated by ";", each: Subject|Predicate|Object[|EDGE[|DATE[|note]]]|refs

## ENTITY RULES:
- TYPE: ORG, PERSON, LOC, PRODUCT, INDUSTRY, AMOUNT, TOPIC, EVENT; you can add new types if needed
- L1>L2>L3: topic path, broad>mid>specific (Business>Automotive>EV); empty ok
- aliases: comma-joined; empty ok
- refs: comma-separated sentence numbers where the entity appears/mentioned in the text, e.g. [1,3,5]
- MAX 3-7 words, shortest common name ("Samsung Electronics Co." -> "Samsung")
- SAME entity = SAME exact name everywhere (so triples link)
- Include orgs, people, govs, products, places, industries, amounts ($2B, 61% growth), topics (FDI, carbon credits)

## TRIPLE RULES:
- Subject/Object must exactly match entity names above
- Predicate: present tense, short ("acquires", "invests", "expands to", "announces to build")
- EDGE (optional, omit if unsure): causal = X caused/led to Y; temporal = X happened before/during Y; hierarchical = X is part/subsidiary of Y; proximity = X near/co-located with Y. Default = semantic (omit)
- DATE (optional): YYYY-MM-DD or YYYY-MM, event date only if stated in the article
- note (optional): <10 words of context
- refs: comma-separated sentence numbers supporting this triple, e.g. [1,2]
- Keep empty slots but keep separators: S|P|O||2025-03|[1] = no EDGE, with DATE, ref sentence 1
- NEVER use "|" or ";" inside any field
- ENSURE the TRIPLES are linked together via the ENTITIES, so they form a connected graph.
- CREATE triples for SURE relevant facts, even if they are not the main topic of the article. ex: "Vietnamese government|IN|Vietnam"

## EXAMPLE:
Input: "[1] In March 2025, Tesla announced it will build a new factory in Vietnam, investing $2B — a move driven by US tariffs."
Output:
{
  "main_topic": "Tesla builds Vietnam EV factory, driven by US tariffs",
  "entities": "Tesla|ORG|Business>Automotive>EV||[1];Vietnam|LOC|Geography>Asia>Southeast Asia||[1];Factory|PRODUCT|Business>Automotive>EV|new EV plant|[1];$2B|AMOUNT|Business>Investment>Capital||[1];US tariffs|TOPIC|Economics>Trade>Tariffs||[1]",
  "triples": "Tesla|announces to build|Factory||2025-03|[1];Factory|located in|Vietnam|||[1];Tesla|invests|$2B|||[1];Factory|owned by|Tesla|hierarchical||[1];US tariffs|drives|$2B|causal||[1]"
}

Output ONLY valid JSON.
"""

USER_PROMPT_TEMPLATE = """
Extract concept graph from these numbered sentences.

TITLE: {title}

SENTENCES:
{text}

"""

SCHEMA = {
    "type": "object",
    "properties": {
        "main_topic": {"type": "string"},
        "entities": {
            "type": "string",
            "description": "name|TYPE|L1>L2>L3|aliases|refs;... ONE LINE. refs = [n1,n2,...] sentence numbers",
        },
        "triples": {
            "type": "string",
            "description": "S|P|O[|edge[|date[|note]]]|refs;... ONE LINE. edge: causal|temporal|proximity|hierarchical or empty. refs = [n1,n2,...] sentence numbers",
        },
    },
    "required": ["main_topic", "entities", "triples"],
    "additionalProperties": False,
}

GENERAL_CONFIG = GeneralLLMConfig().from_dict({"verbose": True})
GENERAL_CONFIG.prompt_token_update(
    system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT_TEMPLATE
)

LLM_CONFIG = LLMConfig().from_dict(
    {"n_ctx": GENERAL_CONFIG.calc_max_n_ctx(), "verbose": False}
)
GEN_CONFIG = GENConfig().from_dict(
    {
        "max_tokens": LLM_CONFIG.n_ctx,
        "temperature": 0.1,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "repeat_last_n": 256,
        "stop": ["</s>", "[/INST]", "###"],
    }
)
API_CONFIG = APIConfig().from_dict({"use_api": False})

_EDGE_VALUES = {e.value for e in EdgeType}
_DATE_RE = re.compile(r"^\d{4}(-\d{2}){0,2}$")
# ── 4. ConceptGraphGenerator ────────────────────────────────────────────────
def split_text_to_sentences(text: str) -> list[str]:
    """
    1. Replace all newlines/carriage returns with a single space.
    2. Split by sentence-ending punctuation: ..., ., ?, !
    3. Keep punctuation attached to the sentence.
    """
    # 1. Normalize all newlines to spaces
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # 2. Split after '...' OR after '.', '?', '!' followed by whitespace
    # Order matters: '...' must be checked before '.' so the dots inside '...' don't trigger early splits
    raw_sentences = re.split(r'(?<=\.\.\.)\s+|(?<=[.?!])\s+', text)
    
    # 3. Clean and filter out empty strings
    return [s.strip() for s in raw_sentences if s.strip()]

def number_lines(lines: list[str]) -> str:
    """Convert text to numbered format: [0] line1\n[1] line2\n..."""
    return '\n'.join(f"[{i}] {line}" for i, line in enumerate(lines))


class ConceptGraphGenerator(LLM_Base):
    def __init__(
        self,
        llm_config: LLMConfig = LLM_CONFIG,
        gen_config: GENConfig = GEN_CONFIG,
        general_config: GeneralLLMConfig = GENERAL_CONFIG,
        api_config: APIConfig = API_CONFIG,
        verbose: bool = False,
    ):
        general_config.verbose = verbose
        super().__init__(
            llm_config=llm_config,
            gen_config=gen_config,
            general_config=general_config,
            schema=SCHEMA,
            api_config=api_config,
        )

        self.predicate_normalizer = PredicateNormalizer()
        self.graph_store = Neo4jGraphStore()

    def _build_prompt(self, article: Article) -> tuple[str, str]:
        txt = article.original_text
        title = article.title
        if article.lang != "en" or not article.lang:
            translated = article.no_save_pool.get("translated") or {}
            txt = translated.get("body") or article.original_text
            title = translated.get("title") or article.title
        txt = number_lines(split_text_to_sentences(txt))
        text = self._truncate_to_budget(txt)
        return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(
            title=title or "(no title)",
            text=text,
        )

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    @staticmethod
    def _parse_refs(s: str) -> List[int]:
        """Parses '[1, 2, 3]' into [1, 2, 3]."""
        if not s:
            return []
        s = s.strip().strip("[]")
        if not s:
            return []
        try:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except ValueError:
            return []

    @classmethod
    def _parse_entities_string(cls, s: str) -> Dict[str, Entity]:
        out: Dict[str, Entity] = {}
        for chunk in (c.strip() for c in (s or "").split(";")):
            if not chunk:
                continue
            # Format: name|TYPE|L1>L2>L3|aliases|refs
            parts = (chunk.split("|") + [""] * 5)[:5]
            name, etype, topics, aliases, refs_str = (p.strip() for p in parts)
            if not name:
                continue
            
            l1, l2, l3 = (topics.split(">") + ["", "", ""])[:3]
            refs = cls._parse_refs(refs_str)
            
            ent = Entity(
                name=name,
                description="",  # Temporarily empty, resolved in _apply_to_article
                entity_type=etype,
                aliases=[a for a in (x.strip() for x in aliases.split(",")) if a],
                L1_topic=l1.strip(),
                L2_topic=l2.strip(),
                L3_topic=l3.strip(),
                no_save_pool={"refs": refs},
            )
            out.setdefault(ent.key, ent)
        return out

    @classmethod
    def _parse_triple_string(
        cls, s: str, entities: Optional[Dict[str, Entity]] = None
    ) -> List[Fact]:
        entities = entities or {}
        facts: List[Fact] = []
        for t in (x.strip() for x in (s or "").split(";")):
            if not t:
                continue
            parts = [x.strip() for x in t.split("|")]
            if len(parts) < 3 or not all(parts[:3]):
                continue
            
            # Format: S|P|O[|edge[|date[|note]]]|refs -> max 7 parts
            parts = (parts + [""] * 7)[:7]
            subj_s, pred, obj_s, edge_s, date_s, note_s, refs_s = parts

            edge = EdgeType.SEMANTIC
            if edge_s and edge_s.lower() in _EDGE_VALUES:
                edge = EdgeType(edge_s.lower())
                
            valid_from = date_s if date_s and _DATE_RE.match(date_s) else None
            refs = cls._parse_refs(refs_s)

            subj = entities.get(cls._norm(subj_s), Entity(name=subj_s))
            obj = entities.get(cls._norm(obj_s), Entity(name=obj_s))
            
            facts.append(
                Fact(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    description="",  # Temporarily empty, resolved in _apply_to_article
                    edge_type=edge,
                    valid_from=valid_from,
                    no_save_pool={"refs": refs, "note": note_s},
                )
            )
        return facts

    @classmethod
    def parse_llm_output(
        cls, data: Dict[str, Any]
    ) -> Tuple[str, List[Entity], List[Fact]]:
        ents = cls._parse_entities_string(data.get("entities", ""))
        facts = cls._parse_triple_string(data.get("triples", ""), ents)
        return data.get("main_topic", ""), list(ents.values()), facts

    @staticmethod
    def _apply_to_article(article: Article, parsed: dict) -> Article:
        """Store raw parsed data. Graph merging happens in process_single."""
        main_topic, entities, triples = ConceptGraphGenerator.parse_llm_output(parsed)

        # Resolve descriptions by concatenating referenced sentences
        sentences = split_text_to_sentences(article.original_text)
        
        def _get_desc(refs: List[int]) -> str:
            parts = []
            for r in refs:
                # Support list (1-based index) or dict mapping from number_lines
                if isinstance(sentences, dict):
                    s = sentences.get(r, "")
                elif isinstance(sentences, list) and 0 < r <= len(sentences):
                    s = sentences[r - 1]
                else:
                    s = ""
                
                # Strip the "[n] " prefix if number_lines added it
                s = re.sub(r"^\[\d+\]\s*", "", str(s)).strip()
                if s:
                    parts.append(s)
            return " ".join(parts)

        for ent in entities:
            ent.description = _get_desc(ent.no_save_pool.get("refs", []))
        for fact in triples:
            fact.description = _get_desc(fact.no_save_pool.get("refs", []))

        article.no_save_pool["concept_graph"] = {
            "main_topic": main_topic,
            "entities": entities,
            "triples": triples,
        }
        return article
    

    def process_single(self, article: Article, use_kg_rag: bool = False) -> Article:
        # Step 1: LLM Extraction
        # Prompt now returns: main_topic, entities (compact string), triples (compact string)
        article = super().process_single(article, use_kg_rag=use_kg_rag)

        # Step 2: Parse — accepts new compact strings OR legacy dict lists
        cg = article.no_save_pool.get("concept_graph", {})
        entities, facts = self._extract_graph(cg)
        if not facts and not entities:
            return article
        # Step 2.5: entity backstop — GLiNER/spaCy entities the LLM dropped
        for cand in (article.no_save_pool.get("pre_graph") or {}).get("entities", []):
            entities.setdefault(cand.key, cand)
        # Step 3: Canonicalize predicates AFTER parsing, so Fact.key (S|P|O)
        # is deduped on the canonical form ("invests in" and "invests" merge)
        for f in facts:
            f.predicate = self.predicate_normalizer.normalize(f.predicate)

        # Step 4: Provenance — valid_from/created_at already set by Fact.__post_init__
        source = article.url or article.title or "unknown"
        for f in facts:
            if source not in f.sources:
                f.sources.append(source)

        # Step 5: Push to Neo4j (batch upsert; store merges on Fact.key)
        if facts:
            self.graph_store.upsert_triples(facts, source=source)

        # Entities with no triple still deserve a node (LightRAG-style)
        linked = {f.subject.key for f in facts} | {f.object.key for f in facts}
        orphans = [e for k, e in entities.items() if k not in linked]
        if orphans and hasattr(self.graph_store, "upsert_entities"):
            self.graph_store.upsert_entities(orphans, source=source)

        # Step 6: Stash parsed results for downstream stages (dedup pass, RAG, eval)
        cg["entities_parsed"] = list(entities.values())
        cg["facts_parsed"] = facts
        article.no_save_pool["concept_graph"] = cg
        return article

    @classmethod
    def _extract_graph(cls, cg: Dict[str, Any]) -> Tuple[Dict[str, Entity], List[Fact]]:
        """Normalize LLM output to (entity_key -> Entity, [Fact]).
        Handles the new compact-string format AND legacy dict lists."""
        # ---- entities ----
        ent_raw = cg.get("entities") or []
        entities: Dict[str, Entity] = {}
        if isinstance(ent_raw, str):
            entities = cls._parse_entities_string(ent_raw)
        else:
            for e in ent_raw:
                if isinstance(e, Entity):
                    ent = e
                elif isinstance(e, dict):
                    valid_fields = Entity.__dataclass_fields__
                    ent = Entity(
                        **{k: v for k, v in e.items() if k in valid_fields and v is not None}
                    )
                else:
                    continue
                entities.setdefault(ent.key, ent)

        # ---- triples ----
        trip_raw = cg.get("triples") or cg.get("raw_triples") or []
        facts: List[Fact] = []
        if isinstance(trip_raw, str):
            facts = cls._parse_triple_string(trip_raw, entities)
        elif trip_raw:
            for t in trip_raw:
                if isinstance(t, Fact):
                    facts.append(t)
                    continue
                s_raw, pred, o_raw = (
                    t.get("subject"),
                    t.get("predicate"),
                    t.get("object"),
                )
                if not (s_raw and pred and o_raw):
                    continue
                subj = (
                    s_raw
                    if isinstance(s_raw, Entity)
                    else entities.get(cls._norm(str(s_raw)), Entity(name=str(s_raw)))
                )
                obj = (
                    o_raw
                    if isinstance(o_raw, Entity)
                    else entities.get(cls._norm(str(o_raw)), Entity(name=str(o_raw)))
                )
                edge = (t.get("edge_type") or "semantic").lower()
                facts.append(
                    Fact(
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        description=t.get("description", ""),
                        edge_type=EdgeType(edge)
                        if edge in _EDGE_VALUES
                        else EdgeType.SEMANTIC,
                        valid_from=t.get("valid_from"),
                    )
                )
        return entities, facts
