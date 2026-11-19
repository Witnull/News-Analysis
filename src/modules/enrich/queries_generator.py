from src.modules.llm.llmBase import LLM_Base
from src.core.types import Article, GENConfig, GeneralLLMConfig, LLMConfig

ASPECTS = ("who", "what", "when", "where", "why", "how")
MAX_PER_ASPECT = 2

SYSTEM_PROMPT = f"""You are a research query generator for an investigative news enrichment pipeline.
Given a news summary and its extracted keywords and additional related information (if applicable), produce targeted web search queries that will
uncover facts still missing about the event, organized under the 5W1H facets: who, what, when,
where, why, how.

Hard rules:
- Ground every query ONLY in entities, places, dates, and terms that literally appear in the
  SUMMARY or KEYWORDS below. Never invent a name, year, date, or location that isn't there.
- Never write placeholder text like "[insert location]" or "[if known]" — if a fact (location,
  date, name) is unknown, write the query to search FOR that missing fact generically instead
  (e.g. "cargo ship grounding location" not "cargo ship grounding [insert location] 2024").
- Each facet gets at most {MAX_PER_ASPECT} queries, and 0 is valid if the summary already covers
  it or there's nothing groundable left to ask.
- Every query across the whole output must probe a genuinely distinct fact — if two facets would
  produce near-identical queries, keep it once under the single best-fitting facet.
- Each query is a short keyword-style search string (not a question, not a full sentence).
- Output strictly the JSON object matching the schema, nothing else, no markdown fences."""

# ── Config ─────────────────────────────────────────────────────────────────────

GENERAL_CONFIG = GeneralLLMConfig().from_dict(
    {"model_path": "./models/Ministral-3-3B-Instruct-2512-Q5_K_M.gguf", "verbose": True}
)

GENERAL_CONFIG.prompt_token_update(system_prompt=SYSTEM_PROMPT, user_prompt="")

LLM_CONFIG = LLMConfig().from_dict(
    {
        "n_gpu_layers": -1,  # full GPU offload (RTX 4060 8GB)
        "n_ctx": 4000,  # MAX TOTAL context window for Mistral Q4_K_M per text.
        "n_batch": 1024,  # prompt eval batch size
        "n_threads": 12,  # CPU threads for prompt eval - more = faster
        "verbose": False,  # llama_cpp verbose logging
    }
)

GEN_CONFIG = GENConfig().from_dict(
    {  # any
        "max_tokens": LLM_CONFIG.n_ctx,  # output Tokens
        "temperature": 0.1,  # low = deterministic / accurate
        "repeat_penalty": 1.0,
        "stop": ["</s>", "[/INST]"],
    }
)

SCHEMA = {
    "type": "object",
    "properties": {
        aspect: {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": MAX_PER_ASPECT,
        }
        for aspect in ASPECTS
    },
    "required": list(ASPECTS),
}


class QueryGenerator(LLM_Base):
    """process_single(article) -> article with .dive_queries populated."""

    def __init__(
        self,
        llm_config: LLMConfig = LLM_CONFIG,
        gen_config: GENConfig = GEN_CONFIG,
        general_config: GeneralLLMConfig = GENERAL_CONFIG,
    ):
        super().__init__(
            llm_config=llm_config,
            gen_config=gen_config,
            general_config=general_config,
            schema=SCHEMA,
        )

    def _build_prompt(self, article: Article) -> tuple[str, str]:
        summary = self._truncate_to_budget(
            article.summarized_text or article.original_text
        )
        prior = article.no_save_pool.get("dive_snippets")

        parts = [
            f"SUMMARY:\n{summary}",
            f"KEYWORDS: {article.keywords}",
        ]
        if prior:
            parts.append(
                "ALREADY FOUND (avoid re-asking these, dig further instead):\n"
                + "\n".join(f"- {s}" for s in prior[:15])
            )
        parts.append(
            "Generate the next round of search queries as a JSON object with keys "
            f"{list(ASPECTS)}, each a list of at most {MAX_PER_ASPECT} query strings."
        )
        return SYSTEM_PROMPT, "\n\n".join(parts)

    @staticmethod
    def _apply_to_article(article: Article, parsed: dict) -> Article:
        by_aspect = {a: parsed.get(a, []) for a in ASPECTS}
        seen_lower: set[str] = set()
        for aspect in ASPECTS:
            deduped = []
            for q in by_aspect[aspect]:
                key = q.lower()
                if key in seen_lower:
                    continue
                seen_lower.add(key)
                deduped.append(q)
            by_aspect[aspect] = deduped
        article.no_save_pool["dive_queries"] = by_aspect
        article.no_save_pool["pending_queries"] = [
            q for qs in by_aspect.values() for q in qs
        ]
        return article


# def demo():
#     """Exercises the real LLM_Base.process_single/process_batch path (not
#     _apply_to_article directly), so this actually catches prompt/schema/parse
#     regressions. Needs a configured local model or API provider — skips
#     (doesn't fail) if neither is available in this environment."""
#     articles = [
#         Article(
#             url="https://example.com/ship-aground",
#             title="Cargo ship runs aground near port",
#             original_text="A cargo ship ran aground near a port, spilling fuel.",
#             summarized_text="A cargo ship ran aground near a port, spilling fuel.",
#             keywords=["cargo ship", "aground", "fuel spill"],
#         )
#     ]

#     try:
#         with QueryGenerator() as qg:
#             qg.process_batch(articles)  # calls process_single per article internally
#     except FileNotFoundError:
#         print("skipped: no local model and no API config found in this environment")
#         return

#     article = articles[0]
#     pending = article.no_save_pool.get("pending_queries")
#     assert pending, f"expected non-empty pending_queries, got {article.no_save_pool!r}"
#     assert len(pending) <= MAX_PER_ASPECT * len(ASPECTS)
#     assert not any("[" in q or "insert" in q.lower() for q in pending)
#     print(f"ok: {len(pending)} queries generated -> {pending}")


# if __name__ == "__main__":
#     demo()
