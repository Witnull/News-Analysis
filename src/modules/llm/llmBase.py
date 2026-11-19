"""

Knowledge-graph-aware extension of `LLM_Core` (see llmCore.py). Adds
entity/predicate extraction, KG retrieval, and retrieval-augmented chat on
top of the plain chat/process_single/process_batch machinery, without
touching the local-vs-API backend dispatch owned by `LLMRouter`.

Fail-safe KG-disable mode
-------------------------
The KG backend (Neo4j driver, predicate normalizer) is optional at runtime:

- If `kg_models`/`kg_store` fail to import (missing package, syntax error,
  etc.), the failure is caught at module load time.
- If `Neo4jGraphStore()` / `PredicateNormalizer()` fail to construct (e.g.
  DB unreachable, bad credentials), the failure is caught in `__init__`.

Either failure logs a single warning and flips `self.kg_enabled = False`.
Every KG method (`query_kg`, `query_entity`, `_extract_entities_from_prompt`,
`chat_with_kg`) then short-circuits into a harmless fallback (empty context /
plain chat) instead of raising, so `use_kg_rag=True` callers keep working -
just without KG augmentation - rather than crashing.
"""

from typing import Dict, Optional

import time
import traceback

from src.core.types import Article, GENConfig, GeneralLLMConfig, LLMConfig
from src.modules.llm.llmCore import LLM_Core, logger
from src.modules.llm.llmRouter import APIConfig

# ── optional KG backend import ───────────────────────────────────────────────
try:
    from src.modules.analysis.kg_models import PredicateNormalizer
    from src.modules.store.kg_store import Neo4jGraphStore

    _KG_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # noqa: BLE001 - deliberately broad: any import failure disables KG
    PredicateNormalizer = None  # type: ignore
    Neo4jGraphStore = None  # type: ignore
    _KG_IMPORT_ERROR = e

KG_MODULES_AVAILABLE = _KG_IMPORT_ERROR is None


class LLM_Base(LLM_Core):
    """LLM_Core plus knowledge-graph retrieval-augmented chat, with a
    fail-safe fallback to plain (non-KG) behavior if the KG backend is
    unavailable or fails to initialize."""

    def __init__(
        self,
        llm_config: LLMConfig = LLMConfig(),
        gen_config: GENConfig = GENConfig(),
        general_config: GeneralLLMConfig = GeneralLLMConfig(),
        use_system_role: bool = True,
        schema: dict = {},
        api_config: Optional[APIConfig] = None,
        verbose: bool = True,
    ):
        super().__init__(
            llm_config=llm_config,
            gen_config=gen_config,
            general_config=general_config,
            use_system_role=use_system_role,
            schema=schema,
            api_config=api_config,
            verbose=verbose,
        )

        self.graph_store = None
        self.predicate_normalizer = None
        self._kg_rag_cache = {}
        self.kg_enabled = False
        self._kg_warned = False  # avoid spamming the log on every call
        self._llm = self.router._llm  # for backward compatibility with LLM_Core

        if not KG_MODULES_AVAILABLE:
            logger.debug(
                f"[{self.__class__.__name__}] KG modules unavailable; running in normal chat mode."
            )
            return

        try:
            self.graph_store = Neo4jGraphStore() # type: ignore
            self.predicate_normalizer = PredicateNormalizer() # type: ignore
            self.kg_enabled = True
        except Exception:
            # Expected when Neo4j is not active: cleanly route to normal chat without traceback
            self.graph_store = None
            self.predicate_normalizer = None
            self.kg_enabled = False

    # ── fail-safe helper ─────────────────────────────────────────────────────

    def _warn_kg_disabled_once(self, context: str):
        if not self._kg_warned:
            logger.debug(
                f"[{self.__class__.__name__}] KG not active - {context} "
                "routed to normal chat."
            )
            self._kg_warned = True

    # ── knowledge-graph retrieval / RAG ─────────────────────────────────────

    def _extract_entities_from_prompt(self, prompt: str) -> Dict:
        """Extract entities and the target predicate/relation from the prompt.
        Returns an empty extraction (no entities/predicate) if KG is disabled."""

        if not self.kg_enabled:
            self._warn_kg_disabled_once("_extract_entities_from_prompt()")
            return {"entities": [], "predicate": None}

        SCHEMA = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of corrected, canonical entity names (1-3 words each).",
                },
                "predicate": {
                    "type": "string",
                    "description": "The core relationship or attribute being queried (e.g., 'CEO', 'capital', 'located in', 'founded by'). Return 'null' if asking for a general summary.",
                },
            },
            "required": ["entities", "predicate"],
        }

        SYS_MSG = (
            "You are an expert Knowledge Graph query planner. "
            "Identify the core subjects and the specific relationship/attribute the user is asking about.\n"
            "Rules:\n"
            "1. Extract proper nouns for 'entities'.\n"
            "2. Extract the core relationship/attribute for 'predicate' - do not repeat same meaning words (e.g., 'CEO', 'capital', 'revenue'). Keep it to 1-8 words.\n"
            "3. If the user just wants a general summary (e.g., 'Tell me about Apple'), set 'predicate' to 'null'.\n"
            "4. Auto-correct typos and slang into formal terms."
        )

        USR_MSG = f"Question: {prompt}"

        try:
            response, _ = self._parse_llm_json(
                self.chat(USR_MSG, SYS_MSG, schema_override=SCHEMA)
            )
            raw_predicate = response.get("predicate")

            # The normalizer maps the raw LLM output to the exact canonical DB string
            normalized_predicate = (
                self.predicate_normalizer.normalize(raw_predicate) if raw_predicate else None # type: ignore
            )

            return {
                "entities": list(response.get("entities", [])),
                "predicate": normalized_predicate,
            }
        except Exception as e:
            logger.warning(
                f"[{self.__class__.__name__}] Entity extraction failed ({e}); "
                "treating as KG-disabled for this call."
            )
            traceback.print_exc()
            return {"entities": [], "predicate": None}

    def query_kg(self, prompt: str, limit: int = 20, hops: int = 1) -> str:
        """Query the knowledge graph and return synthesized context.
        Returns an empty string (no context) if KG is disabled or the query
        fails, so callers building a RAG prompt degrade gracefully."""

        if not self.kg_enabled:
            self._warn_kg_disabled_once("query_kg()")
            return ""

        try:
            extraction = self._extract_entities_from_prompt(prompt)
            key_entities = extraction["entities"]
            target_predicate = extraction["predicate"]

            if not key_entities:
                return "[No relevant entities found in knowledge graph]"

            all_facts = []
            for entity in key_entities:
                # Pass the predicate down to filter the graph query
                summary = self.query_entity(
                    entity, limit=limit, hops=hops, predicate=target_predicate
                )
                all_facts.append(summary)

            if not all_facts:
                return f"[No information found in knowledge graph for: {', '.join(key_entities)}]"

            return "\n".join(all_facts)
        except Exception as e:
            logger.warning(
                f"[{self.__class__.__name__}] query_kg() failed ({e}); "
                "returning empty context for this call."
            )
            traceback.print_exc()
            return ""

    def query_entity(
        self, entity_name: str, limit: int = 10, hops: int = 1, predicate: str = ""
    ) -> str:
        """RAG Retrieval: Get facts/paths and generate a strict summary.
        Returns a plain "no information" message if KG is disabled."""

        if not self.kg_enabled:
            self._warn_kg_disabled_once("query_entity()")
            return f"No information found for '{entity_name}' (KG disabled)."

        try:
            # Pass predicate to the graph store
            facts = self.graph_store.get_entity_summary( # type: ignore
                entity_name, limit=limit, hops=hops, predicate=predicate
            )
        except Exception as e:
            logger.warning(
                f"[{self.__class__.__name__}] graph_store lookup failed for "
                f"'{entity_name}' ({e}); treating as no information found."
            )
            traceback.print_exc()
            return f"No information found for '{entity_name}' (KG lookup error)."

        if not facts:
            # If predicate filtering yielded nothing, fallback to general summary if no predicate was asked
            if predicate:
                return f"No specific information found for '{predicate}' regarding '{entity_name}'."
            return f"No information found for '{entity_name}'."

        if facts and facts[0].get("type") == "path":
            facts_text = "\n".join(
                [
                    f"- Path ({f['hops']} hops, avg hit_count: {f['hit_count']:.1f}): {f['path_string']}"
                    for f in facts
                ]
            )
        else:
            facts_text = "\n".join(
                [
                    f"- {f['subject']} {f['predicate'].replace('_', ' ').lower()} {f['object']} (hit_count: {f['hit_count']})"
                    for f in facts
                ]
            )

        SCHEMA = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A concise, direct summary using ONLY the provided facts/paths.",
                }
            },
            "required": ["summary"],
        }

        SYS_MSG = (
            "You are a strict knowledge graph synthesizer. "
            "Write a concise summary using ONLY the provided facts or paths. "
            "Higher hit_count indicates more reliable connections. "
            "Do not add outside information or hallucinate."
        )

        retrieval_type = "paths (multi-hop)" if hops > 1 else "direct facts"
        USR_MSG = (
            f"Query subject: {entity_name}\nTarget relation: {predicate or 'General Summary'}\n"
            f"Retrieval type: {retrieval_type}\n\nProvided Context:\n{facts_text}"
        )

        response, _ = self._parse_llm_json(
            self.chat(USR_MSG, SYS_MSG, schema_override=SCHEMA)
        )
        return response.get("summary", "Could not synthesize.")

    def chat_with_kg(
        self,
        prompt: str,
        sys_prompt: str = "",
        query_kw: str = "",
    ) -> str:
        """Retrieval-augmented chat. Falls back to plain `self.chat(...)`
        (no RAG context) if KG is disabled or retrieval fails for any
        reason - never raises just because KG is unavailable."""

        if not self.kg_enabled:
            self._warn_kg_disabled_once("chat_with_kg()")
            return self.chat(prompt, sys_prompt)

        try:
            # Query KG for context (1 inference for synthesis)
            query_text = query_kw if query_kw else prompt
            context = self.query_kg(query_text)
        except Exception as e:
            logger.warning(
                f"[{self.__class__.__name__}] chat_with_kg() context retrieval "
                f"failed ({e}); falling back to plain chat."
            )
            traceback.print_exc()
            context = ""

        # Final chat with context (1 inference). If context ended up empty
        # (disabled/failed retrieval), this is equivalent to a plain chat call.
        return self.chat(
            prompt,
            sys_prompt + "IMPORTANT: If you do not know answer it directly, do not hallucinate.",
            rag_context=context,
        )

    def process_single(self, article: Article, use_kg_rag: bool = False) -> Article:
        self._llm = self.router._llm
        """Process a single article. Routes generation through either the
        local llama.cpp model or a hosted API provider, depending on
        self.use_api (set via api_config.use_api).

        `use_kg_rag=True` uses `chat_with_kg()`, which transparently falls
        back to plain chat if the KG backend is disabled or unavailable.
        """

        sys_prompt, prompt = self._build_prompt(article)
        try:
            if self.verbose:
                logger.info(f"{article.title[:60] or article.url[:60]!r}")
            if use_kg_rag:
                raw_chat_result = self.chat_with_kg(
                    prompt,
                    sys_prompt,
                    query_kw=f"{article.title} - {article.keywords} - {article.summarized_text}",
                )
            else:
                raw_chat_result = self.chat(prompt, sys_prompt)

            try:
                parsed, think = self._parse_llm_json(raw_chat_result)
                if self.verbose and think:
                    logger.info(f" [THINK]\n {think}\n")
                return self._apply_to_article(article, parsed)
            except ValueError as e:
                if self.verbose:
                    logger.info(f" [WARN] JSON parse failed: {e}")
                return article  # leave unchanged
            except Exception as e:
                logger.info(f" [ERROR] Failed to process article '{article.title}': {e}")
                traceback.print_exc()
                return article  # leave unchanged

        except Exception:
            traceback.print_exc()
            return article  # leave unchanged

    def process_batch(
        self, articles: list[Article], on_progress=None, use_kg_rag: bool = False
    ) -> None:
        """Process a batch of articles."""
        if not self.use_api and self._llm is None:
            self.load()
            self._llm = self.router._llm

        total = len(articles)
        if self.verbose:
            logger.info(f"[{self.__class__.__name__}] Processing {total} article(s)...\n")

        t_batch = time.time()

        for i, article in enumerate(articles, 1):
            if self.verbose:
                logger.info(f" ({i}/{total})")
            try:
                articles[i - 1] = self.process_single(
                    article, use_kg_rag=use_kg_rag
                )  # substitute with the updated article
            except Exception as e:
                logger.info(f" [ERROR] Failed to process article '{article.title}': {e}")
                traceback.print_exc()
                continue

            if on_progress:
                on_progress(i, total, articles[i - 1])

        elapsed = time.time() - t_batch
        if self.verbose:
            avg = elapsed / total if total else 0
            logger.info(
                f"\n[{self.__class__.__name__}] Done. {total} articles in "
                f"{elapsed:.1f}s (avg {avg:.1f}s/article)"
            )
        return articles