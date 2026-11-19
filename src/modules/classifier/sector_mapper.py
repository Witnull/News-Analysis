import json
import math
from pathlib import Path
from typing import Optional, Union, List

from src.core.types import Article

# ── Stage-4 module ───────────────────────────────────────────────────────────
# Chains off stage 1 (data_card), stage 2 (sentiment_card), stage 3
# (impact_card). Resolves which of the 8 project target sectors an article
# belongs to, then merges all four stages into one flat opportunity record.
#
# Two sector-resolution paths, because the project's 8 target sectors are
# NOT the same shape as GICS's 11 sectors:
#   - Heavy Industry, Mineral, Agriculture, Finance, Computer Science,
#     Logistics map cleanly onto GICS industry groups (company/product
#     based classification) -> resolved via vector match against
#     gics_taxonomy sub-industry definitions.
#   - Geopolitical and Society do NOT exist as GICS sectors at all — they
#     come from gics_taxonomy.json's own "special_cross_cutting_tags" note
#     (Policy/Macro/Geopolitical/Society/N/A) and in practice map straight
#     from the stage-1 tag itself (e.g. "Geopolitical conflict" tag ->
#     Geopolitical sector, no GICS lookup needed).
# Deterministic override table runs first; GICS vector match is the
# fallback for anything product/company based.

DATA_DIR = Path(__file__).parent

# read_text(encoding="utf-8-sig") transparently strips a leading UTF-8 BOM if
# present (common after Windows editors re-save the file) and behaves
# identically to plain UTF-8 if no BOM exists — safe either way.
GICS_TAXONOMY = json.loads((DATA_DIR / "gics_taxonomy.json").read_text(encoding="utf-8-sig"))

TARGET_SECTORS = [
    "Heavy Industry",
    "Mineral",
    "Agriculture",
    "Finance",
    "Computer Science",
    "Logistics",
    "Geopolitical",
    "Society",
    "Policy",
    "Macro",
    "Other/Not in scope",
]

# GICS industry / industry_group name -> project target sector.
# Built from the standard GICS structure (11 sectors, 25 industry groups, 74 industries)
# cross-checked against gics_taxonomy.json.
GICS_INDUSTRY_TO_TARGET_SECTOR = {
    "Metals & Mining": "Mineral",
    "Chemicals": "Heavy Industry",
    "Construction Materials": "Heavy Industry",
    "Containers & Packaging": "Heavy Industry",
    "Paper & Forest Products": "Agriculture",
    "Food Products": "Agriculture",
    "Beverages": "Agriculture",
    "Tobacco": "Agriculture",
}

GICS_INDUSTRY_GROUP_TO_TARGET_SECTOR = {
    "Energy": "Heavy Industry",
    "Capital Goods": "Heavy Industry",
    "Transportation": "Logistics",
    "Banks": "Finance",
    "Financial Services": "Finance",
    "Insurance": "Finance",
    "Software & Services": "Computer Science",
    "Technology Hardware & Equipment": "Computer Science",
    "Semiconductors & Semiconductor Equipment": "Computer Science",
    "Communication Services": "Computer Science",
    "Telecommunication Services": "Computer Science",
    "Media & Entertainment": "Computer Science",
    "Utilities": "Heavy Industry",
    "Real Estate Management & Development": "Finance",
    "Equity Real Estate Investment Trusts (REITs)": "Finance",
}

GICS_SUB_INDUSTRY_OVERRIDE = {
    "Fertilizers & Agricultural Chemicals": "Agriculture",
    "Data Processing & Outsourced Services": "Computer Science",
}

# tag -> target sector, resolved directly with no GICS lookup at all.
# Ported from gics_taxonomy.json's "special_cross_cutting_tags" note:
# these are Policy/Macro/Geopolitical/Society events that aren't tied to
# one company/product and so don't have a natural GICS sub-industry.
CROSS_CUTTING_TAG_OVERRIDE = {
    # Policy (from gics_taxonomy.json special_cross_cutting_tags)
    "Regulation": "Policy",
    "Tax/subsidy": "Policy",
    "Tariff/trade restriction": "Policy",
    "Export/import ban or quota": "Policy",
    "Industrial policy": "Policy",
    "Land use/zoning policy": "Policy",
    "Public health policy": "Policy",
    "Immigration/visa policy": "Policy",
    "Antitrust/competition policy": "Policy",
    "Pending legislation": "Policy",
    "Policy reversal": "Policy",
    # Geopolitical
    "Geopolitical conflict": "Geopolitical",
    "Political instability": "Geopolitical",
    "Sanctions": "Geopolitical",
    "International agreement withdrawal": "Geopolitical",
    "Referendum/election outcome risk": "Geopolitical",
    "Central bank independence risk": "Geopolitical",
    "Terrorism attack": "Geopolitical",
    # Society (demography, culture, social behavior — not one company/product)
    "Social behavior shift": "Society",
    "Population growth/decline": "Society",
    "Age structure shift": "Society",
    "Urban vs rural shift": "Society",
    "Migration": "Society",
    "Household size change": "Society",
    "Gender ratio shift": "Society",
    "Ethnic composition shift": "Society",
    "Disability/health population change": "Society",
    "Cultural/religious event": "Society",
    "Religious/ethical trend": "Society",
    "Generational trend": "Society",
    "Income inequality/distribution shift": "Society",
    # Macro conditions (from gics_taxonomy.json special_cross_cutting_tags)
    "Inflation/deflation": "Macro",
    "Business cycle phase": "Macro",
    "Interest rate environment": "Macro",
    "Currency/exchange rate move": "Macro",
    "Credit crunch/liquidity crisis": "Macro",
    "Sovereign debt crisis": "Macro",
    "Asset bubble/crash": "Macro",
    "Trade balance shift": "Macro",
    "Commodity super-cycle": "Macro",
    "Labor force shrinkage": "Macro",
    "Monetary policy": "Macro",
}


# ── GICS sub-industry vector index (same embed_fn pattern as stage 3) ───────

def _flatten_gics_sub_industries() -> list[dict]:
    flat = []
    for sector in GICS_TAXONOMY.get("sectors", []):
        for ig in sector.get("industry_groups", []):
            for industry in ig.get("industries", []):
                for sub in industry.get("sub_industries", []):
                    flat.append(
                        {
                            "code": sub["code"],
                            "name": sub["name"],
                            "definition": sub.get("definition", ""),
                            "industry": industry["name"],
                            "industry_group": ig["name"],
                            "sector": sector["name"],
                        }
                    )
    return flat


_GICS_SUB_INDUSTRIES = _flatten_gics_sub_industries()
_GICS_EMBEDDING_CACHE: dict[str, list[float]] = {}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_gics_embeddings(embed_fn) -> dict[str, list[float]]:
    if not _GICS_EMBEDDING_CACHE:
        for sub in _GICS_SUB_INDUSTRIES:
            text = f"{sub['name']}: {sub['definition']}"
            _GICS_EMBEDDING_CACHE[sub["code"]] = embed_fn(text)
    return _GICS_EMBEDDING_CACHE


def _sub_to_target_sector(sub: dict) -> str:
    name = sub.get("name")
    if name in GICS_SUB_INDUSTRY_OVERRIDE:
        return GICS_SUB_INDUSTRY_OVERRIDE[name]
    ind = sub.get("industry")
    if ind in GICS_INDUSTRY_TO_TARGET_SECTOR:
        return GICS_INDUSTRY_TO_TARGET_SECTOR[ind]
    ig = sub.get("industry_group")
    if ig in GICS_INDUSTRY_GROUP_TO_TARGET_SECTOR:
        return GICS_INDUSTRY_GROUP_TO_TARGET_SECTOR[ig]
    return "Other/Not in scope"


def _format_sub_industry_name(sub_name: str, event_text: str) -> str:
    """Break down composite GICS sub-industries like 'Financial Exchanges & Data'."""
    if sub_name == "Financial Exchanges & Data":
        trading_terms = {
            "exchange", "exchanges", "trading", "derivatives", "futures", "options",
            "clearing", "clearinghouse", "order book", "matching engine", "settlement",
            "liquidity", "broker", "brokerage", "swap", "cme", "nasdaq", "nyse", "hose", "hnx"
        }
        text_lower = event_text.lower()
        if any(term in text_lower for term in trading_terms):
            return "Financial Exchanges & Trading Venues"
        else:
            return "Financial Market Data & Analytics"
    return sub_name


def _best_gics_match(
    event_text: str, embed_fn, sector_constraint: Optional[str] = None
) -> Optional[dict]:
    embeds = _get_gics_embeddings(embed_fn)
    event_vec = embed_fn(event_text)

    # Filter candidates by sector when a corporate target sector is detected
    candidates = _GICS_SUB_INDUSTRIES
    if (
        sector_constraint
        and sector_constraint in TARGET_SECTORS
        and sector_constraint not in ("Other", "Other/Not in scope", "Society", "Geopolitical", "Policy", "Macro")
    ):
        filtered = [s for s in _GICS_SUB_INDUSTRIES if _sub_to_target_sector(s) == sector_constraint]
        if filtered:
            candidates = filtered

    best_sub, best_score = None, -1.0
    for sub in candidates:
        score = _cosine(event_vec, embeds[sub["code"]])
        if score > best_score:
            best_sub, best_score = sub, score
    if best_sub is None:
        return None
    return {**best_sub, "score": best_score}


def resolve_sector(article: Article, tag: str, embed_fn) -> dict:
    """1. cross-cutting tag override (Policy/Macro/Geopolitical/Society -> no GICS).
    2. LLM cross-cutting sector check (Society/Geopolitical/Policy/Macro -> no GICS).
    3. Sector-constrained GICS sub-industry vector match -> sub-industry breakdown.
    4. "Other/Not in scope" fallback for unmapped corporate sectors."""

    if tag in CROSS_CUTTING_TAG_OVERRIDE:
        return {
            "sector": CROSS_CUTTING_TAG_OVERRIDE[tag],
            "confidence": "tag_override",
            "matched_gics_sub_industry": None,
        }

    data_card = (article.no_save_pool or {}).get("data_card") or {}
    event_text = data_card.get("main_event_summary") or article.title or ""
    llm_sector = data_card.get("target_sector")

    # Non-corporate / cross-cutting civic or macro sectors have no GICS corporate sub-industry
    if llm_sector in ("Society", "Geopolitical", "Policy", "Macro"):
        return {
            "sector": llm_sector,
            "confidence": "llm_classified",
            "matched_gics_sub_industry": None,
        }

    match = _best_gics_match(event_text, embed_fn, sector_constraint=llm_sector)
    matched_sub = match["name"] if match else None
    if matched_sub:
        matched_sub = _format_sub_industry_name(matched_sub, event_text)

    # If LLM classified a concrete target sector, use it
    if llm_sector and llm_sector not in ("Other", "_", "") and llm_sector in TARGET_SECTORS:
        return {
            "sector": llm_sector,
            "confidence": "llm_classified",
            "matched_gics_sub_industry": matched_sub,
        }

    if match is None:
        return {"sector": "Other/Not in scope", "confidence": "no_match", "matched_gics_sub_industry": None}

    sector = _sub_to_target_sector(match)

    return {
        "sector": sector,
        "confidence": "gics_matched" if sector != "Other/Not in scope" else "unmapped_gics",
        "matched_gics_sub_industry": matched_sub,
    }


# ── Orchestrator + final merge ───────────────────────────────────────────────

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


class SectorMapper:
    """Plain orchestrator, same shape as ImpactDurationScorer — no LLM call
    needed at all for this stage, it's a vector match plus a lookup table.
    Writes article.no_save_pool["sector_card"]."""

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn

    def process(self, article: Article) -> Article:
        article.no_save_pool = article.no_save_pool or {}
        data_card = article.no_save_pool.get("data_card")
        if not data_card:
            article.no_save_pool["sector_card"] = None
            return article

        primary = _get_primary_tag(data_card)
        if primary is None:
            article.no_save_pool["sector_card"] = None
            return article

        _, tag = primary
        article.no_save_pool["sector_card"] = resolve_sector(article, tag, self.embed_fn)
        return article

    def process_batch(self, articles: List[Article]) -> List[Article]:
        for article in articles:
            self.process(article)
        return articles


def _build_single_record(article: Article) -> Optional[dict]:
    pool = article.no_save_pool or {}
    data_card = pool.get("data_card")
    if not data_card:
        return None

    sentiment_card = pool.get("sentiment_card") or {}
    impact_card = pool.get("impact_card") or {}
    sector_card = pool.get("sector_card") or {}

    primary = _get_primary_tag(data_card)
    cluster_field, tag = primary if primary else (None, None)

    return {
        "article_type": data_card.get("article_type"),
        "main_event_summary": data_card.get("main_event_summary"),
        "event_status": data_card.get("event_status"),
        "event_date": data_card.get("event_date"),
        "cluster": cluster_field,
        "tag": tag,
        "direction": sentiment_card.get("direction"),
        "roles": sentiment_card.get("roles", []),
        "fixed_sentiment_override": sentiment_card.get("fixed_sentiment_override", False),
        "duration": impact_card.get("duration"),
        "duration_confidence": impact_card.get("duration_confidence"),
        "scope": impact_card.get("scope"),
        "scope_confidence": impact_card.get("scope_confidence"),
        "matched_checkpoint": impact_card.get("matched_checkpoint"),
        "target_sector": sector_card.get("sector"),
        "sector_confidence": sector_card.get("confidence"),
        "matched_gics_sub_industry": sector_card.get("matched_gics_sub_industry"),
    }


def build_opportunity_record(
    articles: Union[Article, List[Article]],
    discard_empty: bool = True,
) -> Union[Optional[dict], List[Optional[dict]]]:
    """Final stage-4 merge: flattens data_card + sentiment_card +
    impact_card + sector_card (stages 1-4) into opportunity record(s).
    Supports either a single Article or a list of Articles.
    When discard_empty=True, filters out articles that had no actionable event tag."""
    if isinstance(articles, list):
        records = [_build_single_record(a) for a in articles]
        if discard_empty:
            return [r for r in records if r is not None and r.get("tag") is not None]
        return records
    rec = _build_single_record(articles)
    if discard_empty and rec and rec.get("tag") is None:
        return None
    return rec
