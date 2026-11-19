"""End-to-end classifier pipeline: runs Articles through all 4 stages,
chaining "with" per step class with LLM calling for immediate load and unload
from VRAM.

Stage 1 (tag)          -> current_classifier.ArticleSummarizer (LLM)
Stage 2 (sentiment)    -> sentiment_extractor.EventSentimentExtractor (LLM)
Stage 3 (impact)       -> impact_duration_scorer.ImpactDurationScorer (Embedding)
Stage 4 (sector+merge) -> sector_mapper.SectorMapper, build_opportunity_record (Embedding)
"""

from typing import Optional, List, Union
import json

from src.core.types import Article
from src.modules.llm.llmEmbed import LLMEmbed

try:
    from .current_classifier import ArticleSummarizer
    from .sentiment_extractor import EventSentimentExtractor
    from .impact_duration_scorer import ImpactDurationScorer
    from .sector_mapper import SectorMapper, build_opportunity_record
    from .fast_filter import ClassifierFilter
except ImportError:
    from current_classifier import ArticleSummarizer
    from sentiment_extractor import EventSentimentExtractor
    from impact_duration_scorer import ImpactDurationScorer
    from sector_mapper import SectorMapper, build_opportunity_record
    from fast_filter import ClassifierFilter


def run_pipeline(
    articles: List[Article],
    embed_model_path: str = "./models/Qwen3-Embedding-0.6B-Q8_0.gguf",
    use_tiebreak: bool = False,
    verbose: bool = False,
    discard_empty: bool = True,
) -> Optional[List[dict]]:
    """Runs Articles through all 4 classifier stages.

    Chains 'with' context managers per step class with LLM calling so that models
    are loaded into memory only during their active step and immediately evicted,
    retaining zero unnecessary VRAM.

    Inter-step ClassifierFilter gates screen out no-value articles between steps:
    - Step 0: Pre-Stage 1 static screening drops chart chatter, pet charity, listicles
    - Step 1: Post-Stage 1 gate drops untagged/non-economic articles before Stage 2 LLM
    - Step 2: Post-Stage 2 gate drops articles lacking directional stakeholder impact
    - Final:  Drops empty records if discard_empty=True
    """
    is_single = isinstance(articles, Article)
    article_list = [articles] if is_single else articles

    # Step 0: Pre-Stage 1 fast deterministic filter (0ms, 0 VRAM)
    active, dropped_pre = ClassifierFilter.filter_pre_stage1(article_list)
    if verbose and dropped_pre:
        print(f"[run_pipeline] Step 0: dropped {len(dropped_pre)} non-economic/duplicate articles.")
    if not active:
        return [] if not is_single else None

    # Stage 1: Tagging & event classification (LLM)
    with ArticleSummarizer(verbose=verbose) as summarizer:
        summarizer.process_batch(active)

    # Step 1: Post-Stage 1 gate (actionable tag & economic sector validation)
    active, dropped_s1 = ClassifierFilter.filter_post_stage1(active)
    if verbose and dropped_s1:
        print(f"[run_pipeline] Step 1: dropped {len(dropped_s1)} untagged/no-impact articles before Stage 2.")

    # Stage 2: Event sentiment & role extraction (LLM)
    if active:
        with EventSentimentExtractor(verbose=verbose) as sentiment_extractor:
            sentiment_extractor.process_batch(active)

    # Step 2: Post-Stage 2 gate (sentiment direction & stakeholder validation)
    active, dropped_s2 = ClassifierFilter.filter_post_stage2(active)
    if verbose and dropped_s2:
        print(f"[run_pipeline] Step 2: dropped {len(dropped_s2)} articles with no direction/roles.")

    # Stage 3 & 4: Impact Duration/Scope and Sector mapping (Embedding)
    if active:
        with LLMEmbed(model_path=embed_model_path) as embedder:
            impact_scorer = ImpactDurationScorer(
                embed_fn=embedder.encode,
                use_tiebreak=use_tiebreak,
            )
            impact_scorer.process_batch(active)

            sector_mapper = SectorMapper(embed_fn=embedder.encode)
            sector_mapper.process_batch(active)

    records = build_opportunity_record(article_list, discard_empty=discard_empty)
    if discard_empty and isinstance(records, list):
        records = ClassifierFilter.filter_records(records)

    if is_single:
        return records[0] if isinstance(records, list) and records else None
    return records


# Kept for callers using process_single
process_single = run_pipeline


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_title = (
        "VietRice Corp announces new 200,000-ton rice processing plant in Mekong Delta"
    )
    sample_text = (
        "VietRice Corp said Monday it will build a new processing facility, "
        "adding 200,000 tons of annual capacity by 2027. Analysts said the "
        "expansion could pressure margins for smaller domestic millers "
        "already facing overcapacity, while helping stabilize rice prices "
        "for exporters buying from the region. The company also said it "
        "will hire 400 new workers for the plant."
    )

    sample_article = Article(
        url="https://example.com/vietrice-plant-announcement",
        title=sample_title,
        original_text=sample_text,
        lang="en",
    )

    records = run_pipeline([sample_article], verbose=False)
    print(json.dumps(records, indent=2, ensure_ascii=False))
