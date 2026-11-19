import json
import math
from pathlib import Path
from typing import Optional

from src.core.types import Article, GeneralLLMConfig, LLMConfig, GENConfig
from src.modules.llm.llmBase import APIConfig, LLM_Base

# ── Stage-3 module ───────────────────────────────────────────────────────────
# Chains off stage 1 (ArticleSummarizer -> data_card) and stage 2
# (EventSentimentExtractor -> sentiment_card). Implements
# impact_evaluation_framework.md: two axes, duration and scope, resolved by
# lexicon match -> tense hint -> tag default (later) time-series
# correction for duration; vector match against the checkpoint graph -> LLM
# tie-break only among top candidates for scope.
#
# Unlike stage 1/2, most of this is plain deterministic code, not an
# LLM_Base subclass — per the framework doc, the model is only ever used for
# a closed pick among 2-3 checkpoint candidates, never an open question.

# DATA_DIR points at this file's own folder (Classifier_module/) — all three
# JSONs (duration_scope_lexicon.json, tag_duration_scope_defaults.json,
# checkpoint_graph_template.json) must live alongside this file directly.
DATA_DIR = Path(__file__).parent

TAG_DURATION_SCOPE_DEFAULTS = json.loads(
    (DATA_DIR / "tag_duration_scope_defaults.json").read_text(encoding="utf-8-sig")
)
DURATION_SCOPE_LEXICON = json.loads(
    (DATA_DIR / "duration_scope_lexicon.json").read_text(encoding="utf-8-sig")
)
CHECKPOINT_GRAPH = json.loads(
    (DATA_DIR / "checkpoint_graph_template.json").read_text(encoding="utf-8-sig")
)

DURATION_ENUM = ["short", "mid", "long"]
SCOPE_ENUM = ["single", "multi", "whole"]

CLUSTER_FIELD_TO_TOP = {
    "demand_side_tag": "Demand side",
    "supply_side_tag": "Supply side",
    "objective_event_tag": "Objective event",
    "policy_change_tag": "Policy change",
    "self_entity_tag": "Self-entity",
}

# Flatten tag_duration_scope_defaults.json into tag -> {top, sub, duration, scope}
# for O(1) lookup by tag string alone (matches stage-1's data_card tag values).
_TAG_INDEX: dict[str, dict] = {}
for _top, _subclusters in TAG_DURATION_SCOPE_DEFAULTS.items():
    for _sub, _tags in _subclusters.items():
        for _tag, _vals in _tags.items():
            _TAG_INDEX[_tag] = {
                "top": _top,
                "sub": _sub,
                "duration": _vals["duration"],
                "scope": _vals["scope"],
            }


# ── Duration resolution ──────────────────────────────────────────────────────


def _lexicon_duration_match(text: str) -> Optional[str]:
    text_l = text.lower()
    for label, keywords in DURATION_SCOPE_LEXICON["duration_override"].items():
        for kw in keywords:
            if kw in text_l:
                return label
    return None


def _tense_hint_duration(text: str) -> Optional[str]:
    text_l = text.lower()
    hints = DURATION_SCOPE_LEXICON["tense_hint"]
    for kw in hints.get("future_ongoing", []):
        if kw in text_l:
            return "mid"  # weak signal: not yet concluded, treat as ongoing/mid
    for kw in hints.get("past_concluded", []):
        if kw in text_l:
            return "short"  # weak signal: already resolved, treat as short
    return None


def resolve_duration(article: Article, tag: str) -> dict:
    """Resolution order per impact_evaluation_framework.md:
    1. lexicon match  2. tense hint  3. tag default.
    Step 4 (time-series correction) runs later, across the dedup/cluster
    pipeline when the same event recurs in later ingested articles — not
    here, since this function only sees one article at a time.
    """
    text = article.original_text or ""

    lex = _lexicon_duration_match(text)
    if lex:
        return {"duration": lex, "confidence": "text_confirmed"}

    tense = _tense_hint_duration(text)
    if tense:
        return {"duration": tense, "confidence": "text_confirmed"}

    default = _TAG_INDEX.get(tag)
    if default:
        return {"duration": default["duration"], "confidence": "default_only"}

    return {"duration": "short", "confidence": "default_only"}


# ── Scope resolution (checkpoint graph) ──────────────────────────────────────


def _flatten_checkpoint_nodes() -> list[dict]:
    nodes = []
    for m in CHECKPOINT_GRAPH["macro_layer"]["nodes"]:
        nodes.append(
            {
                "id": m["id"],
                "sub_industry": None,
                "stage": None,
                "description": m["description"],
                "is_macro": True,
            }
        )
    for sub_name, sub_data in CHECKPOINT_GRAPH["sub_industries"].items():
        for cp in sub_data["checkpoints"]:
            nodes.append(
                {
                    "id": f"{sub_name}.{cp['stage']}",
                    "sub_industry": sub_name,
                    "stage": cp["stage"],
                    "description": cp["description"],
                    "is_macro": False,
                }
            )
    return nodes


_CHECKPOINT_NODES = _flatten_checkpoint_nodes()
_NODE_EMBEDDING_CACHE: dict[str, list[float]] = {}

SECTOR_TO_CHECKPOINT_CHAINS = {
    "Computer Science": ["software_services", "semiconductor", "electronics_manufacturing"],
    "Finance": ["banking_finance"],
    "Mineral": ["iron_steel_mineral"],
    "Agriculture": ["rice_agriculture", "agri_biotech_feed", "aquaculture", "poultry_swine"],
    "Heavy Industry": [
        "oil_gas",
        "power_energy",
        "iron_steel_mineral",
        "pharma_biotech",
        "coal_energy",
        "fertilizer",
        "construction_materials",
        "textile_apparel",
    ],
    "Logistics": ["logistics"],
    "Macro": [],  # only macro nodes
    "Geopolitical": [],
}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_node_embeddings(embed_fn) -> dict[str, list[float]]:
    """embed_fn: str -> list[float]. Embeds context-rich description including
    sub-industry and stage name to prevent cross-industry embedding sink collisions.
    Cached at process level since checkpoint descriptions never change."""
    if not _NODE_EMBEDDING_CACHE:
        for node in _CHECKPOINT_NODES:
            if node.get("is_macro"):
                text = f"Macro economic condition {node['id']}: {node['description']}"
            else:
                sub = (node.get("sub_industry") or "").replace("_", " ")
                stage = (node.get("stage") or "").replace("_", " ")
                text = f"{sub} industry value chain {stage} stage: {node['description']}"
            _NODE_EMBEDDING_CACHE[node["id"]] = embed_fn(text)
    return _NODE_EMBEDDING_CACHE


def _top_k_checkpoints(
    event_text: str,
    embed_fn,
    k: int = 3,
    sector_constraint: Optional[str] = None,
    cluster: Optional[str] = None,
) -> list[tuple[dict, float]]:
    node_embeds = _get_node_embeddings(embed_fn)
    event_vec = embed_fn(event_text)

    # Filter candidates based on target sector and cluster
    is_macro_context = (
        sector_constraint in ("Macro", "Geopolitical")
        or (cluster == "objective_event_tag" and sector_constraint not in SECTOR_TO_CHECKPOINT_CHAINS)
    )

    if is_macro_context:
        candidates = [n for n in _CHECKPOINT_NODES if n.get("is_macro")]
    elif sector_constraint in SECTOR_TO_CHECKPOINT_CHAINS and SECTOR_TO_CHECKPOINT_CHAINS[sector_constraint]:
        allowed = set(SECTOR_TO_CHECKPOINT_CHAINS[sector_constraint])
        candidates = [
            n for n in _CHECKPOINT_NODES
            if not n.get("is_macro") and n.get("sub_industry") in allowed
        ]
    elif cluster in ("supply_side_tag", "demand_side_tag", "self_entity_tag"):
        # Firm / industry event without specific recognized sector: candidate nodes from any supply chain
        candidates = [n for n in _CHECKPOINT_NODES if not n.get("is_macro")]
    else:
        candidates = _CHECKPOINT_NODES

    if not candidates:
        candidates = _CHECKPOINT_NODES

    scored = [
        (node, _cosine(event_vec, node_embeds[node["id"]]))
        for node in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def resolve_scope(
    article: Article,
    tag: str,
    embed_fn,
    tiebreaker: Optional["ScopeTiebreaker"] = None,
    top_k: int = 3,
    ambiguous_margin: float = 0.04,
    cluster: Optional[str] = None,
) -> dict:
    """Resolution order: 1. vector match against checkpoint_graph_template.json
    2. LLM tie-break only if top candidates are within ambiguous_margin
    3. tag default fallback if no checkpoint nodes exist at all.
    """
    data_card = (article.no_save_pool or {}).get("data_card") or {}
    event_text = data_card.get("main_event_summary") or article.title or ""
    llm_sector = data_card.get("target_sector")

    default = _TAG_INDEX.get(tag)
    fallback_scope = default["scope"] if default else "single"

    # Society events (demographics, animal welfare, civic programs) or non-industry
    # do not sit on an industrial value chain
    if llm_sector in ("Society", "Other", "Other/Not in scope"):
        return {
            "scope": fallback_scope,
            "confidence": "default_only",
            "matched_checkpoint": None,
        }

    if cluster is None:
        primary = _get_primary_tag(data_card)
        if primary:
            cluster = primary[0]

    top_matches = _top_k_checkpoints(
        event_text,
        embed_fn,
        k=top_k,
        sector_constraint=llm_sector,
        cluster=cluster,
    )

    if not top_matches:
        return {
            "scope": fallback_scope,
            "confidence": "default_only",
            "matched_checkpoint": None,
        }

    best_node, best_score = top_matches[0]
    second_score = top_matches[1][1] if len(top_matches) > 1 else 0.0
    ambiguous = (best_score - second_score) < ambiguous_margin

    chosen_node = best_node
    confidence = "graph_matched"

    if ambiguous and tiebreaker is not None:
        candidate_ids = [n["id"] for n, _ in top_matches]
        picked_id = tiebreaker.pick(article, event_text, candidate_ids)
        if picked_id:
            chosen_node = next(
                (n for n, _ in top_matches if n["id"] == picked_id), best_node
            )
            confidence = "llm_tiebreak"

    if chosen_node["is_macro"]:
        scope = "whole"
    else:
        close_subs = {
            n["sub_industry"]
            for n, s in top_matches
            if not n["is_macro"] and (best_score - s) < 0.1
        }
        if len(close_subs) <= 1:
            scope = "single"
        elif len(close_subs) <= 3:
            scope = "multi"
        else:
            scope = "whole"

    return {
        "scope": scope,
        "confidence": confidence,
        "matched_checkpoint": chosen_node["id"],
    }


# ── Config (for ScopeTiebreaker only — the deterministic functions above
# need no LLM config at all) ─────────────────────────────────────────────────

GENERAL_CONFIG = GeneralLLMConfig().from_dict({"verbose": False})

LLM_CONFIG = LLMConfig().from_dict(
    {
        "n_ctx": GENERAL_CONFIG.calc_max_n_ctx(),
        "verbose": False,
    }
)

GEN_CONFIG = GENConfig().from_dict(
    {
        "temperature": 0.0,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
)

API_CONFIG = APIConfig().from_dict({"use_api": False})


class ScopeTiebreaker(LLM_Base):
    """Closed-choice tie-break: pick exactly one checkpoint id among 2-3
    vector-search candidates. Never an open question about how far an event
    spreads — per impact_evaluation_framework.md's small-model usage rule.

    Unlike ArticleSummarizer/EventSentimentExtractor, this is instantiated
    fresh per ambiguous case, because its enum (candidate checkpoint ids)
    changes every call instead of being fixed at import time.
    """

    def __init__(
        self,
        candidate_nodes: list[dict],
        llm_config: LLMConfig = LLM_CONFIG,
        gen_config: GENConfig = GEN_CONFIG,
        general_config: GeneralLLMConfig = GENERAL_CONFIG,
        api_config: APIConfig = API_CONFIG,
        verbose: bool = False,
    ):
        self._candidate_desc = {n["id"]: n["description"] for n in candidate_nodes}
        candidate_ids = list(self._candidate_desc.keys())

        schema = {
            "type": "object",
            "properties": {
                "picked_checkpoint": {"type": "string", "enum": candidate_ids},
            },
            "required": ["picked_checkpoint"],
            "additionalProperties": False,
        }

        general_config.verbose = verbose

        super().__init__(
            llm_config=llm_config,
            gen_config=gen_config,
            general_config=general_config,
            schema=schema,
            api_config=api_config,
        )

    def _build_prompt(self, event_text: str) -> tuple[str, str]:
        options = "\n".join(
            f"- {cid}: {desc}" for cid, desc in self._candidate_desc.items()
        )
        system = (
            "You are a closed-choice checkpoint picker for a value-chain graph.\n"
            "Pick exactly ONE checkpoint id that best matches where this event sits "
            "in the value chain. Return JSON only, no explanation.\n\n"
            f"CANDIDATES:\n{options}"
        )
        user = f"EVENT: {event_text}"
        return system, user

    def pick(
        self, article: Article, event_text: str, candidate_ids: list[str]
    ) -> Optional[str]:
        """Runs one closed-choice generation via LLM_Base.chat (confirmed
        signature: chat(prompt, sys_prompt) -> str, followed by
        _parse_llm_json). candidate_ids kept for interface compatibility
        with resolve_scope's call site, though the candidates were already
        baked into the schema at construction time."""
        system, user = self._build_prompt(event_text)
        raw = self.chat(user, system)
        try:
            parsed, _think = self._parse_llm_json(raw)
        except Exception:
            return None
        if not parsed:
            return None
        return parsed.get("picked_checkpoint")


# ── Orchestrator ─────────────────────────────────────────────────────────────

DATA_CARD_TAG_FIELDS = [
    "demand_side_tag",
    "supply_side_tag",
    "objective_event_tag",
    "policy_change_tag",
    "self_entity_tag",
]


def _get_primary_tag(data_card: dict) -> Optional[tuple[str, str]]:
    for field in DATA_CARD_TAG_FIELDS:
        val = data_card.get(field, "Other")
        if val and val not in ("Other", "_"):
            return field, val
    return None


class ImpactDurationScorer:
    """Plain orchestrator, not an LLM_Base subclass — most of stage 3 is
    deterministic lookups. Mirrors the _apply_to_article contract used by
    stage 1/2 so it slots into the same per-article processing chain,
    writing into article.no_save_pool["impact_card"].
    """

    def __init__(self, embed_fn, use_tiebreak: bool = True, top_k: int = 3):
        self.embed_fn = embed_fn
        self.use_tiebreak = use_tiebreak
        self.top_k = top_k

    def process(self, article: Article) -> Article:
        data_card = (article.no_save_pool or {}).get("data_card")
        if not data_card:
            article.no_save_pool = article.no_save_pool or {}
            article.no_save_pool["impact_card"] = None
            return article

        primary = _get_primary_tag(data_card)
        if primary is None:
            article.no_save_pool["impact_card"] = None
            return article

        cluster_field, tag = primary

        duration_result = resolve_duration(article, tag)

        llm_sector = data_card.get("target_sector")
        top_matches = _top_k_checkpoints(
            (data_card.get("main_event_summary") or article.title or ""),
            self.embed_fn,
            k=self.top_k,
            sector_constraint=llm_sector,
            cluster=cluster_field,
        )
        tiebreaker = None
        if self.use_tiebreak and top_matches and llm_sector not in ("Society", "Other", "Other/Not in scope"):
            best_score = top_matches[0][1]
            second_score = top_matches[1][1] if len(top_matches) > 1 else 0.0
            if (best_score - second_score) < 0.04:
                tiebreaker = ScopeTiebreaker([n for n, _ in top_matches])

        scope_result = resolve_scope(
            article,
            tag,
            self.embed_fn,
            tiebreaker=tiebreaker,
            top_k=self.top_k,
            cluster=cluster_field,
        )

        article.no_save_pool["impact_card"] = {
            "tag": tag,
            "cluster": cluster_field,
            "duration": duration_result["duration"],
            "duration_confidence": duration_result["confidence"],
            "scope": scope_result["scope"],
            "scope_confidence": scope_result["confidence"],
            "matched_checkpoint": scope_result.get("matched_checkpoint"),
        }

        return article

    def process_batch(self, articles: list[Article]) -> list[Article]:
        for article in articles:
            self.process(article)
        return articles

    # TODO: time-series correction (duration resolution step 4). Needs the
    # dedup/cluster pipeline to detect recurring articles about the same
    # real-world event across a time window and upgrade duration_confidence
    # to "time_series_confirmed" when the observed span exceeds the current
    # label. Does not belong in this single-article class.
