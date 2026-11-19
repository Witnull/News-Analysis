from src.core.types import Article, GeneralLLMConfig, LLMConfig, GENConfig
from src.modules.llm.llmBase import APIConfig, LLM_Base

from string import Template
import traceback

# ── Stage-2 module ───────────────────────────────────────────────────────────
# Chains directly off ArticleSummarizer (current classifier.py).
# It does NOT re-pick the tag. It reads article.no_save_pool["data_card"]
# (written by stage 1) to find which of the 5 cluster fields is not "Other",
# and only asks the model for direction + role/sentiment/scope on that
# already-known tag. This removes the duplicate tag-extraction that existed
# in the standalone draft (suggested_sentiment_extractor.py).
# One gap I left as-is: DEFAULT_ROLES covers most tags but I approximated a few sparse spots (e.g. "Producer financing/credit access" only lists Producer) rather than inventing roles not clearly stated in the PDF — worth a manual pass against the PDF table before production use.

ROLE_ENUM = [
    "Producer",
    "New entrant",
    "Consumer",
    "Supplier",
    "Labor",
    "Investor",
    "Environment/Third party",
    "Self-entity",
]

DATA_CARD_TAG_FIELDS = [
    "demand_side_tag",
    "supply_side_tag",
    "objective_event_tag",
    "policy_change_tag",
    "self_entity_tag",
]

# tag -> fixed sentiment, ported from the standalone draft's
# FIXED_SENTIMENT_EVENTS idea, but keyed by plain tag string (matching
# classifier.py's SCHEMA enums) instead of a colon-joined event string.
# These are tags where target_role_sentiment_mapping.pdf gives one sentiment
# regardless of direction, so we skip the open role-picking step entirely.
FIXED_SENTIMENT_TAGS = {
    # Objective event / Exogenous shock — mostly fixed negative
    "Natural disaster": "negative",
    "Disease/epidemic": "negative",
    "Weather anomaly": "negative",
    "Accident": "negative",
    "Volcanic/geological event": "negative",
    "Lockdown/movement restriction": "negative",
    "Infrastructure failure": "negative",
    "Space weather": "negative",
    "Terrorism attack": "negative",
    # Self-entity / Ethic problem — always negative
    "Fraud": "negative",
    "Labor violation": "negative",
    "Environmental violation": "negative",
    "Data breach/privacy violation": "negative",
    "Animal welfare violation": "negative",
    "Bribery/corruption": "negative",
    "IP theft/counterfeiting": "negative",
    # Self-entity / New product
    "Launch": "positive",
    "Product recall": "negative",
    "Regulatory approval/certification": "positive",
    "Rebranding/repositioning": "neutral",
    # Self-entity / Default/debt
    "Loan default": "negative",
    "Bond default": "negative",
    "Debt restructuring": "negative",
    "Credit rating change": "negative",
    "Insolvency/liquidation": "negative",
    "Foreign-currency debt mismatch risk": "negative",
    # Self-entity / Drama
    "Leadership change/scandal": "negative",
    "Legal dispute/lawsuit": "negative",
    "Internal conflict": "negative",
    "Cybersecurity breach": "negative",
    "PR crisis": "negative",
    "Succession crisis": "negative",
    "Whistleblower event": "negative",
}

# tag -> default 2-3 roles, ported from target_role_sentiment_mapping.pdf
# section headers ("default roles: ..."). Used as a hint in the prompt so
# the model picks from the right subset instead of guessing across all 8.
DEFAULT_ROLES = {
    # Demand / Trends
    "Consumer trend": ["Producer", "Consumer"],
    "Media-driven trend": ["Producer", "Consumer"],
    "Industry trend": ["Producer", "Consumer"],
    "Seasonal trend": ["Producer", "Consumer"],
    "Generational trend": ["Producer", "Consumer"],
    "Income-class trend": ["Producer", "Consumer"],
    "Religious/ethical trend": ["Producer", "Consumer"],
    "Regional/geographic trend": ["Producer", "Consumer"],
    # Demand / National avg salary
    "Wage growth": ["Labor", "Producer", "Consumer"],
    "Wage decline": ["Labor", "Producer", "Consumer"],
    "Purchasing power change": ["Labor", "Producer", "Consumer"],
    "Income inequality/distribution shift": ["Labor", "Producer", "Consumer"],
    "Informal economy wage": ["Labor", "Producer", "Consumer"],
    "Remittance inflow": ["Labor", "Producer", "Consumer"],
    "Investment/dividend income": ["Labor", "Producer", "Consumer"],
    # Demand / Demography
    "Population growth/decline": ["Producer", "Consumer", "Labor"],
    "Age structure shift": ["Producer", "Consumer", "Labor"],
    "Urban vs rural shift": ["Producer", "Consumer", "Labor"],
    "Migration": ["Producer", "Consumer", "Labor"],
    "Household size change": ["Producer", "Consumer", "Labor"],
    "Gender ratio shift": ["Producer", "Consumer", "Labor"],
    "Ethnic composition shift": ["Producer", "Consumer", "Labor"],
    "Disability/health population change": ["Producer", "Consumer", "Labor"],
    # Demand / Lifestyle/culture
    "Habit change": ["Producer", "Consumer"],
    "Cultural/religious event": ["Producer", "Consumer"],
    "Health/diet shift": ["Producer", "Consumer"],
    "Education level shift": ["Producer", "Consumer"],
    "Digital penetration": ["Producer", "Consumer"],
    "Tourism-driven demand": ["Producer", "Consumer"],
    "E-commerce adoption": ["Producer", "Consumer"],
    "Sharing economy adoption": ["Producer", "Consumer"],
    # Demand / Substitution
    "Direct substitute price change": ["Producer", "New entrant", "Consumer"],
    "Complement price change": ["Producer", "New entrant", "Consumer"],
    "Cross-industry substitute": ["Producer", "New entrant", "Consumer"],
    "Self-production substitute": ["Producer", "Consumer"],
    "Rental vs own substitution": ["Producer", "New entrant", "Consumer"],
    "Import substitution": ["Producer", "New entrant", "Consumer"],
    "Generic vs branded substitution": ["Producer", "New entrant", "Consumer"],
    # Demand / Number of buyers
    "New buyer entry": ["Producer", "Consumer"],
    "Buyer exit": ["Producer", "Consumer"],
    "Buyer concentration": ["Producer", "Consumer"],
    "Buyer bargaining power shift": ["Producer", "Consumer"],
    "Group purchasing/co-op formation": ["Producer", "Consumer"],
    "Institutional buyer entry": ["Producer", "Consumer"],
    # Demand / Expectation
    "Price expectation": ["Producer", "Consumer"],
    "Shortage expectation": ["Producer", "Consumer"],
    "Future demand forecast": ["Producer", "Consumer"],
    "Anticipatory buying before policy change": ["Producer", "Consumer"],
    "Expectation of tech disruption": ["Producer", "Consumer"],
    # Supply / Technology
    "New production method": ["Producer", "New entrant", "Labor"],
    "Automation/efficiency upgrade": ["Producer", "Labor"],
    "Tech breakthrough": ["Producer", "New entrant"],
    "Tech transfer/licensing": ["Producer", "New entrant"],
    "Tech obsolescence": ["Producer", "New entrant", "Labor"],
    "Patent expiry": ["Producer", "New entrant"],
    "R&D investment change": ["Producer", "Labor"],
    # Supply / Productivity-capacity
    "Capacity expansion": ["Producer", "Consumer", "Labor"],
    "Capacity cut/closure": ["Producer", "Consumer", "Labor"],
    "Utilization rate change": ["Producer", "Consumer", "Labor"],
    "Yield change": ["Producer", "Consumer"],
    "Planned maintenance shutdown": ["Producer", "Consumer"],
    "Strategic reserve release/drawdown": ["Producer", "Consumer"],
    "Capacity relocation": ["Producer", "Labor"],
    "Vertical integration": ["Producer", "New entrant"],
    # Supply / Value chain
    "Input price change": ["Producer", "Consumer"],
    "Transportation problem": ["Producer", "Consumer"],
    "Logistics bottleneck": ["Producer", "Consumer"],
    "Labor cost/shortage": ["Producer", "Consumer", "Labor"],
    "Producer financing/credit access": ["Producer"],
    "Single-source supplier risk": ["Producer", "Supplier"],
    "Resource constraint": ["Producer", "Consumer"],
    "FX pass-through on input": ["Producer", "Consumer"],
    "Packaging material cost": ["Producer", "Consumer"],
    "Shipping insurance cost": ["Producer", "Consumer"],
    # Supply / Number of sellers
    "New entrant": ["Producer", "New entrant", "Consumer"],
    "Seller exit/bankruptcy": ["Producer", "Consumer"],
    "Consolidation": ["Producer", "New entrant", "Consumer"],
    "Cartel/coordination": ["Producer", "New entrant", "Consumer"],
    "State-owned enterprise intervention": ["Producer", "Consumer"],
    "Informal/black market supply": ["Producer", "Consumer"],
    "Joint venture formation": ["Producer", "New entrant"],
    # Supply / Expectation
    "Producer price expectation": ["Producer"],
    "Input cost forecast": ["Producer"],
    "Pre-emptive stockpiling by producer": ["Producer"],
    # Objective / Exogenous shock -> handled via FIXED_SENTIMENT_TAGS
    # Objective / Macro condition
    "Inflation/deflation": ["Producer", "Consumer", "Investor"],
    "Business cycle phase": ["Producer", "Consumer", "Investor"],
    "Interest rate environment": ["Producer", "Consumer", "Investor"],
    "Currency/exchange rate move": ["Producer", "Consumer", "Investor"],
    "Social behavior shift": ["Producer", "Consumer", "Investor"],
    "Commodity super-cycle": ["Producer", "Consumer", "Investor"],
    "Credit crunch/liquidity crisis": ["Producer", "Consumer", "Investor"],
    "Sovereign debt crisis": ["Producer", "Consumer", "Investor"],
    "Labor force shrinkage": ["Producer", "Labor"],
    "Asset bubble/crash": ["Producer", "Investor"],
    "Trade balance shift": ["Producer", "Investor"],
    # Policy / New policy
    "Tax/subsidy": ["Producer", "Consumer"],
    "Tariff/trade restriction": ["Producer", "Consumer"],
    "Regulation": ["Producer", "Consumer"],
    "Export/import ban or quota": ["Producer", "Consumer"],
    "Monetary policy": ["Producer", "Consumer"],
    "Industrial policy": ["Producer"],
    "Land use/zoning policy": ["Producer"],
    "Public health policy": ["Producer", "Consumer"],
    "Immigration/visa policy": ["Producer", "Labor"],
    "Antitrust/competition policy": ["Producer", "New entrant", "Consumer"],
    # Policy / Policy risk
    "Pending legislation": ["Producer", "Investor"],
    "Political instability": ["Producer", "Investor"],
    "Sanctions": ["Producer", "Investor"],
    "Geopolitical conflict": ["Producer", "Investor"],
    "Policy reversal": ["Producer", "Investor"],
    "International agreement withdrawal": ["Producer", "Investor"],
    "Referendum/election outcome risk": ["Producer", "Investor"],
    "Central bank independence risk": ["Producer", "Investor"],
}

DEFAULT_ROLES_OTHER = ["Producer", "Consumer"]


def _get_primary_tag(data_card: dict) -> tuple[str, str] | None:
    """Return (cluster_field, tag) for whichever of the 5 tag fields is not
    'Other'/'_' in the stage-1 data_card. None if nothing is tagged yet."""
    for field in DATA_CARD_TAG_FIELDS:
        val = data_card.get(field, "Other")
        if val and val not in ("Other", "_"):
            return field, val
    return None


# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = Template("""You are a News Intelligence sentiment classifier for a trading and investment signal system.

TASK: The article's primary event has already been tagged as "$TAG" (cluster: $CLUSTER). Do NOT re-tag it. Extract only:
1. direction: how the tagged factor moved. Options: increase|decrease|no_change
2. roles: 1-3 roles ONLY from this list: $ROLE_ENUM
   Default roles for this tag are: $DEFAULT_ROLES — use these unless the article text explicitly names a different affected party.
   For each role assign sentiment (positive|negative|neutral) and scope (direct|indirect_1hop).
   direct = same market as the tag. indirect_1hop = nearest substitute/complement named in the article text only. Never chain beyond 1 hop.

## MODEL RULES
- Return ONLY a valid JSON object matching the provided schema. No markdown formatting, no extra text.
- Extract only information explicitly stated in the article. Never infer market impact not stated.
- Never invent a role outside the given list.
- If the tagged event has a fixed/obvious sentiment (e.g. a disaster, fraud case), still assign roles and sentiment as instructed — the fixed-sentiment shortcut is applied outside this step for those cases.

## EXAMPLE
Tagged tag: "Capacity expansion" (cluster: supply_side_tag)
Input text: "VietRice Corp said Monday it will build a new processing facility, adding 200,000 tons of annual capacity by 2027. Analysts said the expansion could pressure margins for smaller domestic millers, while helping stabilize rice prices for exporters. The company will hire 400 new workers."
Output:
{"direction": "increase", "roles": [{"role": "Producer", "sentiment": "negative", "scope": "direct"}, {"role": "Consumer", "sentiment": "positive", "scope": "direct"}, {"role": "Labor", "sentiment": "positive", "scope": "direct"}]}
""")

USER_PROMPT_TEMPLATE = """
Extract direction and role sentiment for the tagged event in this article.

TITLE: {title}

ARTICLE:
{text}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["increase", "decrease", "no_change"],
        },
        "roles": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ROLE_ENUM},
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                    },
                    "scope": {"type": "string", "enum": ["direct", "indirect_1hop"]},
                },
                "required": ["role", "sentiment", "scope"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["direction", "roles"],
    "additionalProperties": False,
}


# ── Config ─────────────────────────────────────────────────────────────────────
# Built once at import time with a placeholder tag/cluster/roles — the real
# per-article system prompt is rendered fresh in _build_prompt() below, since
# it depends on the stage-1 tag. This mirrors classifier.py's config-at-import
# pattern while allowing per-article prompt content.

_PLACEHOLDER_SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.substitute(
    TAG="(resolved per article)",
    CLUSTER="(resolved per article)",
    ROLE_ENUM=", ".join(ROLE_ENUM),
    DEFAULT_ROLES="(resolved per article)",
)

GENERAL_CONFIG = GeneralLLMConfig().from_dict(
    {
        "verbose": True,
    }
)
GENERAL_CONFIG.prompt_token_update(
    system_prompt=_PLACEHOLDER_SYSTEM_PROMPT,
    user_prompt=USER_PROMPT_TEMPLATE,
)

LLM_CONFIG = LLMConfig().from_dict(
    {
        "n_ctx": GENERAL_CONFIG.calc_max_n_ctx(),
        "verbose": False,
    }
)

GEN_CONFIG = GENConfig().from_dict(
    {
        "temperature": 0.1,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }
)

API_CONFIG = APIConfig().from_dict(
    {
        "use_api": False,
    }
)


# ── Extractor ────────────────────────────────────────────────────────────────


class EventSentimentExtractor(LLM_Base):
    def __init__(
        self,
        llm_config: LLMConfig = LLM_CONFIG,
        gen_config: GENConfig = GEN_CONFIG,
        general_config: GeneralLLMConfig = GENERAL_CONFIG,
        api_config: APIConfig = API_CONFIG,
        verbose: bool = True,
    ):
        general_config.verbose = verbose

        super().__init__(
            llm_config=llm_config,
            gen_config=gen_config,
            general_config=general_config,
            schema=SCHEMA,
            api_config=api_config,
        )

    def _build_prompt(self, article: Article) -> tuple[str, str] | None:
        """Build the system and user prompt for one article.

        Returns None if stage 1 (ArticleSummarizer) hasn't tagged this
        article yet, or the tag is a fixed-sentiment shortcut tag — in
        either case there's nothing for the LLM to do here.
        """

        data_card = (article.no_save_pool or {}).get("data_card")
        if not data_card:
            return None

        primary = _get_primary_tag(data_card)
        if primary is None:
            return None

        cluster_field, tag = primary

        if tag in FIXED_SENTIMENT_TAGS:
            # Handled deterministically in process_single's None short-circuit;
            # skip the LLM call entirely for these.
            return None

        default_roles = DEFAULT_ROLES.get(tag, DEFAULT_ROLES_OTHER)

        system = SYSTEM_PROMPT_TEMPLATE.substitute(
            TAG=tag,
            CLUSTER=cluster_field,
            ROLE_ENUM=", ".join(ROLE_ENUM),
            DEFAULT_ROLES=", ".join(default_roles),
        )

        txt = article.original_text
        title = article.title

        if article.lang != "en" or not article.lang:
            translated = article.no_save_pool.get("translated") or {}
            txt = translated.get("body") or article.original_text
            title = translated.get("title") or article.title

        text = self._truncate_to_budget(txt)

        user = USER_PROMPT_TEMPLATE.format(
            title=title or "(no title)",
            text=text,
        )

        return system, user

    def process_single(self, article: Article, use_kg_rag: bool = False) -> Article:
        """Override of LLM_Base.process_single.

        LLM_Base's base implementation unconditionally does
        `sys_prompt, prompt = self._build_prompt(article)`, which crashes
        when _build_prompt returns None (our fixed-sentiment / untagged
        short-circuit). This override checks for that case first and routes
        straight to _apply_to_article(article, None), which already knows
        how to resolve the fixed-sentiment tags deterministically. When a
        real prompt exists, this mirrors LLM_Base.process_single exactly.
        """

        prompt = self._build_prompt(article)
        if prompt is None:
            return self._apply_to_article(article, None)

        sys_prompt, user_prompt = prompt

        try:
            if self.verbose:
                logger_title = article.title[:60] or article.url[:60]
                print(f"[EventSentimentExtractor] {logger_title!r}")

            raw_chat_result = self.chat(user_prompt, sys_prompt)

            try:
                parsed, think = self._parse_llm_json(raw_chat_result)
                return self._apply_to_article(article, parsed)
            except ValueError:
                return self._apply_to_article(article, None)
            except Exception:
                traceback.print_exc()
                return self._apply_to_article(article, None)

        except Exception:
            traceback.print_exc()
        return self._apply_to_article(article, None)

    @staticmethod
    def _apply_to_article(article: Article, parsed: dict | None) -> Article:
        if article.no_save_pool is None:
            article.no_save_pool = {}

        data_card = article.no_save_pool.get("data_card") or {}
        primary = _get_primary_tag(data_card)

        if primary is None:
            article.no_save_pool["sentiment_card"] = None
            return article

        cluster_field, tag = primary

        if tag in FIXED_SENTIMENT_TAGS:
            article.no_save_pool["sentiment_card"] = {
                "tag": tag,
                "cluster": cluster_field,
                "direction": "decrease"
                if FIXED_SENTIMENT_TAGS[tag] == "negative"
                else "increase",
                "fixed_sentiment_override": True,
                "roles": [
                    {
                        "role": "Self-entity",
                        "sentiment": FIXED_SENTIMENT_TAGS[tag],
                        "scope": "direct",
                    }
                ],
            }
            return article

        if not parsed:
            article.no_save_pool["sentiment_card"] = None
            return article

        article.no_save_pool["sentiment_card"] = {
            "tag": tag,
            "cluster": cluster_field,
            "direction": parsed.get("direction", "_"),
            "fixed_sentiment_override": False,
            "roles": parsed.get("roles", []),
        }

        return article
