"""Dedicated inter-step filter for the classifier module.

Enforces structural contracts between classifier stages to drop no-value
articles early and prevent wasted LLM and embedding operations.

Zero hard-coded keyword lists: uses structural contracts (cluster tags,
target sectors, affected roles, directional sentiment) produced by the stages.

Inter-step filters:
- Step 0 (filter_pre_stage1): In-batch deduplication and upstream rejection checks.
- Step 1 (filter_post_stage1): Drops articles where Stage 1 identified no actionable
  economic cluster tag or classified the event into non-economic sectors.
- Step 2 (filter_post_stage2): Drops articles where Stage 2 found no affected
  stakeholder roles or economic direction.
- Final (filter_records): Strips empty/unactionable records from output.
"""

import hashlib
import re
from typing import List, Optional, Set, Tuple
from src.core.types import Article

DATA_CARD_TAG_FIELDS = [
    "demand_side_tag",
    "supply_side_tag",
    "objective_event_tag",
    "policy_change_tag",
    "self_entity_tag",
]

NON_ECONOMIC_SECTORS = {
    "Society",
    "Other",
    "Other/Not in scope",
}


def _fingerprint(text: str) -> str:
    cleaned = re.sub(r"\s+", "", text.lower()[:200])
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def get_primary_tag(data_card: dict) -> Optional[Tuple[str, str]]:
    """Finds the primary tag from the 5 cluster fields in data_card."""
    for field in DATA_CARD_TAG_FIELDS:
        val = data_card.get(field, "Other")
        if val and val not in ("Other", "_", "none", "None"):
            return field, val
    return None


class ClassifierFilter:
    """Structural inter-step filter for the 4-stage classifier pipeline."""

    @staticmethod
    def filter_pre_stage1(articles: List[Article]) -> Tuple[List[Article], List[Article]]:
        """Step 0: Fast screening before Stage 1 LLM.
        Drops duplicate articles in the current batch and respects upstream drops.
        """
        kept: List[Article] = []
        dropped: List[Article] = []
        seen_titles: Set[str] = set()
        seen_fps: Set[str] = set()

        for a in articles:
            # Respect upstream collector filter flag
            if (a.no_save_pool or {}).get("keep") is False:
                ClassifierFilter._mark_dropped(a, "Dropped by upstream filter", "pre_stage1")
                dropped.append(a)
                continue

            title = (a.title or "").strip()
            body = (a.original_text or "").strip()

            if not title and not body:
                ClassifierFilter._mark_dropped(a, "Empty article title and body", "pre_stage1")
                dropped.append(a)
                continue

            # In-batch deduplication via normalized title & body fingerprint
            t_norm = re.sub(r"\W+", "", title.lower())
            if t_norm and len(t_norm) > 10:
                if t_norm in seen_titles:
                    ClassifierFilter._mark_dropped(a, "Duplicate title in batch", "pre_stage1")
                    dropped.append(a)
                    continue
                seen_titles.add(t_norm)

            fp = _fingerprint(body if len(body) >= 50 else title)
            if fp in seen_fps:
                ClassifierFilter._mark_dropped(a, "Duplicate content in batch", "pre_stage1")
                dropped.append(a)
                continue
            seen_fps.add(fp)

            kept.append(a)

        return kept, dropped

    @staticmethod
    def filter_post_stage1(articles: List[Article]) -> Tuple[List[Article], List[Article]]:
        """Step 1: Evaluates Stage 1 data_card before Stage 2 LLM.
        Drops articles where Stage 1 found no actionable event tag, where
        the event is in a non-economic sector, or where event summary is missing.
        """
        kept: List[Article] = []
        dropped: List[Article] = []

        for a in articles:
            data_card = (a.no_save_pool or {}).get("data_card") or {}

            # 1. Actionable tag check: at least one cluster tag must be set
            primary = get_primary_tag(data_card)
            if primary is None:
                ClassifierFilter._mark_dropped(a, "No actionable economic event tag", "post_stage1")
                dropped.append(a)
                continue

            # 2. Sector validity: drop non-economic civic/society domains
            sector = data_card.get("target_sector")
            if sector in NON_ECONOMIC_SECTORS:
                ClassifierFilter._mark_dropped(a, f"Non-economic sector ({sector})", "post_stage1")
                dropped.append(a)
                continue

            # 3. Factual summary check
            summary = (data_card.get("main_event_summary") or "").strip()
            if not summary:
                ClassifierFilter._mark_dropped(a, "Missing main event summary", "post_stage1")
                dropped.append(a)
                continue

            kept.append(a)

        return kept, dropped

    @staticmethod
    def filter_post_stage2(articles: List[Article]) -> Tuple[List[Article], List[Article]]:
        """Step 2: Evaluates Stage 2 sentiment_card before Stage 3/4 embeddings.
        Drops articles where no stakeholder roles were affected and direction was null/neutral.
        """
        kept: List[Article] = []
        dropped: List[Article] = []

        for a in articles:
            sentiment_card = (a.no_save_pool or {}).get("sentiment_card") or {}
            roles = sentiment_card.get("roles") or []
            direction = sentiment_card.get("direction")

            # Must affect at least one stakeholder role or indicate market direction
            if not roles and direction in (None, "neutral", "unknown", "_"):
                ClassifierFilter._mark_dropped(a, "No affected stakeholder roles or direction", "post_stage2")
                dropped.append(a)
                continue

            kept.append(a)

        return kept, dropped

    @staticmethod
    def filter_records(records: List[Optional[dict]]) -> List[dict]:
        """Final Output Gate: Drops empty records lacking tag or summary."""
        clean: List[dict] = []
        for r in records:
            if r is None:
                continue
            if not r.get("tag") or not r.get("main_event_summary"):
                continue
            clean.append(r)
        return clean

    @staticmethod
    def _mark_dropped(article: Article, reason: str, stage: str):
        if article.no_save_pool is None:
            article.no_save_pool = {}
        article.no_save_pool["keep"] = False
        article.no_save_pool["drop_reason"] = reason
        article.no_save_pool["drop_stage"] = stage


# ── Self Check (Ponytail verification) ────────────────────────────────────────

if __name__ == "__main__":
    # 1. Pre-stage 1: in-batch deduplication
    a1 = Article(url="u1", title="Steel Tariff Imposed", original_text="US imposes 25% steel tariff.")
    a2 = Article(url="u2", title="Steel Tariff Imposed", original_text="US imposes 25% steel tariff.")
    kept, dropped = ClassifierFilter.filter_pre_stage1([a1, a2])
    assert len(kept) == 1 and len(dropped) == 1
    assert dropped[0].no_save_pool["drop_stage"] == "pre_stage1"

    # 2. Post-stage 1: untagged article dropped
    a_untagged = Article(url="u3", title="No Tag", original_text="General text.")
    a_untagged.no_save_pool = {"data_card": {"demand_side_tag": "Other", "main_event_summary": "Test"}}
    kept, dropped = ClassifierFilter.filter_post_stage1([a_untagged])
    assert len(kept) == 0 and len(dropped) == 1
    assert dropped[0].no_save_pool["drop_stage"] == "post_stage1"

    # 3. Post-stage 1: Society sector dropped
    a_society = Article(url="u4", title="Cat Spaying", original_text="Cat spaying program.")
    a_society.no_save_pool = {
        "data_card": {
            "self_entity_tag": "Regulatory approval/certification",
            "target_sector": "Society",
            "main_event_summary": "Cat spaying restarted",
        }
    }
    kept, dropped = ClassifierFilter.filter_post_stage1([a_society])
    assert len(kept) == 0 and len(dropped) == 1

    # 4. Post-stage 1: valid economic tag passes
    a_valid = Article(url="u5", title="Valid Economic Event", original_text="Valid.")
    a_valid.no_save_pool = {
        "data_card": {
            "supply_side_tag": "Process innovation",
            "target_sector": "Agriculture",
            "main_event_summary": "Biochar furnace launched",
        }
    }
    kept, dropped = ClassifierFilter.filter_post_stage1([a_valid])
    assert len(kept) == 1 and len(dropped) == 0

    # 5. Post-stage 2: directionless & no roles dropped
    a_norole = Article(url="u6", title="No Role", original_text="No role.")
    a_norole.no_save_pool = {"sentiment_card": {"roles": [], "direction": None}}
    kept, dropped = ClassifierFilter.filter_post_stage2([a_norole])
    assert len(kept) == 0 and len(dropped) == 1
    assert dropped[0].no_save_pool["drop_stage"] == "post_stage2"

    print("All ClassifierFilter contract self-checks PASSED!")
