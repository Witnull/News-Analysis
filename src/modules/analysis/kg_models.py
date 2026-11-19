import re
import hashlib
import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import chromadb
from src.modules.llm.llmEmbed import LLMEmbed

class EdgeType(Enum):
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    PROXIMITY = "proximity"
    HIERARCHICAL = "hierarchical"
    CAUSAL = "causal"


@dataclass
class Entity:
    name: str
    description: str = ""
    entity_type: str = ""
    aliases: List[str] = field(default_factory=list)
    L1_topic: str = ""
    L2_topic: str = ""
    L3_topic: str = ""
    additional_info: Dict[str, Any] = field(default_factory=dict)
    no_save_pool: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.name

    @property
    def key(self) -> str:
        """Normalized dedup key for the entity itself."""
        return re.sub(r"\s+", " ", self.name.strip().lower())

    def to_neo4j_properties(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.entity_type,
            "aliases": self.aliases,
            "L1_topic": self.L1_topic,
            "L2_topic": self.L2_topic,
            "L3_topic": self.L3_topic,
            **self.additional_info,
        }


@dataclass
class Fact:
    subject: Entity
    object: Entity

    predicate: str # edge name
    description: str = ""
    edge_type: EdgeType = EdgeType.SEMANTIC

    sources: List[str] = field(default_factory=list)

    hit_count: int = 1
    weight: float = 1.0

    # ── Bi-temporal fields ───────────────────────────────────────
    valid_from: Optional[str] = None
    invalid_from: Optional[str] = None
    created_at: Optional[str] = None

    # ── Hierarchical metadata ────────────────────────────────────
    additional_info: Dict[str, Any] = field(default_factory=dict)
    no_save_pool: Dict[str, Any] = field(default_factory=dict)


    def __post_init__(self):
        now = datetime.now().isoformat()
        if self.created_at is None:
            self.created_at = now
        if self.valid_from is None:
            self.valid_from = self.created_at

    @property
    def key(self) -> str:
        """Deterministic dedup key for exact-match SPO (case/space normalized)."""
        s = re.sub(r"\s+", " ", self.subject.name.strip().lower())
        p = re.sub(r"\s+", " ", self.predicate.strip().lower())
        o = re.sub(r"\s+", " ", self.object.name.strip().lower())
        return f"{s}|{p}|{o}"

    @property
    def text(self) -> str:
        """Natural-language rendering used for embeddings / BM25 / LLM context."""
        base = f"{self.subject.name} {self.predicate} {self.object.name}"
        return f"{base}. {self.description}".strip() if self.description else base

    def to_neo4j_properties(self) -> Dict[str, Any]:
        return {
            "subject_props": {
                "L1_topic": self.subject.L1_topic,
                "L2_topic": self.subject.L2_topic,
                "L3_topic": self.subject.L3_topic,
                **self.subject.additional_info,
            },
            "object_props": {
                "L1_topic": self.object.L1_topic,
                "L2_topic": self.object.L2_topic,
                "L3_topic": self.object.L3_topic,
                **self.object.additional_info,
            },
            "rel_props": {

                "sources": self.sources,
                "hit_count": self.hit_count,
                "weight": self.weight,
                "description": self.description,
                "edge_type": self.edge_type.value,
                "valid_from": self.valid_from,
                "invalid_from": self.invalid_from,
                "created_at": self.created_at,
                **self.additional_info,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fact (temporal SPO edge) — replaces `Triple`
# ─────────────────────────────────────────────────────────────────────────────

class PredicateNormalizer:
    """Normalizes free-form predicates to a canonical set, and merges
    relation *descriptions* attached to that canonical predicate so the
    vocabulary carries semantic context, not just a bare verb string."""

    SEED_PREDICATES = [
        "invests", "acquires", "announces", "increases", "declines", "partners with",
        "expands to", "reports", "launches", "regulates", "approves", "rejects",
        "develops", "builds", "operates in", "employs", "sues", "fines",
    ]

    def __init__(
        self,
        persist_dir: str = "./cache/concept_graph",
        similarity_threshold: float = 0.88,
        desc_merge_threshold: float = 0.85,
        max_descriptions_per_predicate: int = 12,
    ):
        self.threshold = similarity_threshold
        self.desc_merge_threshold = desc_merge_threshold
        self.max_descriptions = max_descriptions_per_predicate
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.embedder = LLMEmbed()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="predicates", metadata={"hnsw:space": "cosine"}
        )
        # Separate collection just for description dedup within a predicate bucket
        self.desc_collection = self.client.get_or_create_collection(
            name="predicate_descriptions", metadata={"hnsw:space": "cosine"}
        )

        if self.collection.count() == 0:
            self._seed()

    def _seed(self):
        texts = [p.lower().strip() for p in self.SEED_PREDICATES]
        embeddings = self.embedder.encode_batch(texts)
        ids = [hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts]
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=texts,
            metadatas=[{"canonical": t, "hit_count": 0, "description_count": 0} for t in texts],
        )

    def normalize(self, predicate: str) -> str:
        if not predicate or predicate.lower() == "null":
            return "acts"
        pred = re.sub(r"\s+", " ", predicate.strip().lower())
        embedding = self.embedder.encode(pred)
        results = self.collection.query(
            query_embeddings=[embedding], n_results=1,
            include=["distances", "documents", "metadatas"],
        )
        if results["distances"][0]:
            similarity = 1.0 - results["distances"][0][0]
            canonical = results["documents"][0][0]
            meta = results["metadatas"][0][0]
            if similarity >= self.threshold:
                self.collection.update(
                    ids=results["ids"][0],
                    metadatas=[{**meta, "hit_count": meta.get("hit_count", 0) + 1}],
                )
                return canonical

        doc_id = hashlib.sha256(pred.encode()).hexdigest()
        self.collection.add(
            ids=[doc_id], embeddings=[embedding], documents=[pred],
            metadatas=[{"canonical": pred, "hit_count": 1, "description_count": 0}],
        )
        return pred

    def merge_description(self, canonical_predicate: str, description: str) -> None:
        """Attach a description instance to a canonical predicate, deduping
        near-identical descriptions via embedding similarity instead of exact
        string match. Enforces max_descriptions budget."""
        if not description:
            return
        desc = re.sub(r"\s+", " ", description.strip())
        if not desc:
            return

        existing = self.desc_collection.get(where={"predicate": canonical_predicate})
        if existing and len(existing.get("ids", [])) >= self.max_descriptions:
            return

        embedding = self.embedder.encode(desc)
        query_res = self.desc_collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"predicate": canonical_predicate},
            include=["distances"],
        )
        if query_res.get("distances") and query_res["distances"][0]:
            sim = 1.0 - query_res["distances"][0][0]
            if sim >= self.desc_merge_threshold:
                return  # near-duplicate description, skip storing

        bucket_id = hashlib.sha256(canonical_predicate.encode()).hexdigest()[:16]
        doc_id = f"{bucket_id}_{hashlib.sha256(desc.encode()).hexdigest()[:12]}"
        self.desc_collection.upsert(
            ids=[doc_id], embeddings=[embedding], documents=[desc],
            metadatas=[{"predicate": canonical_predicate}],
        )

    def get_descriptions(self, canonical_predicate: str, limit: int = 5) -> List[str]:
        res = self.desc_collection.get(where={"predicate": canonical_predicate}, include=["documents"])
        docs = res.get("documents", []) or []
        return docs[:limit]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ─────────────────────────────────────────────────────────────────────────────
# Entity deduplication: deterministic first, LLM fallback for ambiguous cases
# ─────────────────────────────────────────────────────────────────────────────

class EntityDeduper:
    """Resolves a raw extracted entity name to an existing canonical node
    name if one already represents the same real-world entity.

    Strategy:
      1. Deterministic: normalized exact match / known alias match.
      2. Embedding similarity band: HIGH -> auto-merge, LOW -> definitely new,
         AMBIGUOUS band -> ask the LLM to make the final call.
    """

    def __init__(
        self,
        llm,
        persist_dir: str = "./cache/concept_graph",
        high_sim: float = 0.93,
        low_sim: float = 0.80,
    ):
        self.llm = llm
        self.high_sim = high_sim
        self.low_sim = low_sim
        self.embedder = LLMEmbed()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="entities", metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _norm(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())

    def resolve(self, raw_name: str) -> Tuple[str, bool]:
        """Returns (canonical_name, is_new_entity)."""
        norm = self._norm(raw_name)
        if not norm:
            return raw_name, True

        embedding = self.embedder.encode(norm)
        results = self.collection.query(
            query_embeddings=[embedding], n_results=3,
            include=["distances", "documents", "metadatas"],
        )
        docs = results["documents"][0] if results["documents"] else []
        dists = results["distances"][0] if results["distances"] else []

        # Fast path: check exact normalized match first across candidates
        for doc in docs:
            if self._norm(doc) == norm:
                return doc, False

        for doc, dist in zip(docs, dists):
            similarity = 1.0 - dist
            if similarity >= self.high_sim:
                return doc, False  # confident embedding match
            if similarity >= self.low_sim:
                # ambiguous zone: ask LLM to disambiguate
                if self._llm_confirms_same_entity(raw_name, doc):
                    return doc, False

        # No match found -> register as new canonical entity using upsert to avoid collision
        doc_id = hashlib.sha256(norm.encode()).hexdigest()
        self.collection.upsert(
            ids=[doc_id], embeddings=[embedding], documents=[raw_name],
            metadatas=[{"canonical": raw_name}],
        )
        return raw_name, True

    def _llm_confirms_same_entity(self, name_a: str, name_b: str) -> bool:
        schema = {
            "type": "object",
            "properties": {"same_entity": {"type": "boolean"}},
            "required": ["same_entity"],
        }
        system_msg = (
            "You resolve entity deduplication for a knowledge graph. "
            "Answer whether the two given names refer to the SAME real-world "
            "entity (allowing for abbreviations, aliases, translations, or "
            "formatting differences). Be conservative: only say true if certain."
        )
        user_msg = f"Name A: {name_a}\nName B: {name_b}"
        try:
            raw = self.llm.chat(user_msg, system_msg, schema_override=schema)
            parsed, _ = self.llm._parse_llm_json(raw)
            return bool(parsed.get("same_entity", False))
        except Exception:
            return False