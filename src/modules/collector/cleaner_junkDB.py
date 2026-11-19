import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
from src.modules.llm.llmEmbed import LLMEmbed

# ── Junk Cache ─────────────────────────────────────────────────────────────





class JunkCache():
    """
    ChromaDB-backed line-level junk cache.
    Operates on pre-split lines — same granularity as the LLM prompt.
    """

    def __init__(
        self,
        persist_dir: str = "./cache/chroma_junk",
        collection_name: str = "junk_lines",
        similarity_threshold: float = 0.93,
        min_line_length: int = 15,
    ):
        self.threshold = similarity_threshold
        self.min_line_length = min_line_length

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.model = LLMEmbed()

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[JunkCache] {self.collection.count()} cached patterns")

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _doc_id(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    # ── write ──────────────────────────────────────────────────────────

    def add_lines(self, lines: list[str]):
        """Batch-add junk lines to cache. Deduplicates automatically."""
        to_add_ids, to_add_docs, to_add_metas = [], [], []

        for line in lines:
            norm = self._normalize(line)
            if len(norm) < self.min_line_length:
                continue

            doc_id = self._doc_id(norm)

            # Skip if already cached (just bump hit count)
            existing = self.collection.get(ids=[doc_id], include=["metadatas"])
            if existing["ids"]:
                meta = existing["metadatas"][0]
                self.collection.update(
                    ids=[doc_id],
                    metadatas=[{
                        **meta,
                        "hit_count": meta.get("hit_count", 0) + 1,
                        "last_seen": datetime.now().isoformat(),
                    }],
                )
                continue

            to_add_ids.append(doc_id)
            to_add_docs.append(norm)
            to_add_metas.append({
                "hit_count": 1,
                "length": len(norm),
                "last_seen": datetime.now().isoformat(),
            })

        if not to_add_ids:
            return

        # Batch embed + insert
        embeddings = self.model.encode_batch(to_add_docs)
        self.collection.add(
            ids=to_add_ids,
            embeddings=embeddings,
            documents=to_add_docs,
            metadatas=to_add_metas,
        )
        print(f"[JunkCache] Added {len(to_add_ids)} new patterns (total: {self.collection.count()})")

    # ── read (the hot path) ────────────────────────────────────────────

    def classify_lines(self, lines: list[str]) -> set[int]:
        """
        Classify each line as junk (1) or clean (0).
        Returns set of line INDICES that are junk.

        Single batch embed + single batch query — no loops over text.
        """
        if self.collection.count() == 0:
            return set()

        # Filter to lines worth checking
        candidates = {}  # original_index -> normalized_text
        for i, line in enumerate(lines):
            norm = self._normalize(line)
            if len(norm) >= self.min_line_length:
                candidates[i] = norm

        if not candidates:
            return set()

        indices = list(candidates.keys())
        texts = [candidates[i] for i in indices]

        # 1) Batch embed all candidate lines (~50ms for 200 lines)
        embeddings = self.model.encode_batch(texts)

        # 2) Batch query ChromaDB (~10ms for 200 queries)
        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=1,
            include=["distances", "documents"],
        )

        # 3) Threshold check
        junk_indices = set()
        for idx, dists, docs in zip(indices, results["distances"], results["documents"]):
            if not dists:
                continue
            similarity = 1.0 - dists[0]
            if similarity >= self.threshold:
                junk_indices.add(idx)

        return junk_indices

    # ── maintenance ────────────────────────────────────────────────────

    def prune_stale(self, days: int = 30, min_hits: int = 2):
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        all_data = self.collection.get(include=["metadatas"])
        to_delete = [
            doc_id
            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"])
            if meta.get("last_seen", "") < cutoff
            or meta.get("hit_count", 0) < min_hits
        ]
        if to_delete:
            self.collection.delete(ids=to_delete)
            print(f"[JunkCache] Pruned {len(to_delete)} stale patterns")

