from src.core.types import Article, GeneralLLMConfig, LLMConfig, GENConfig
from src.modules.llm.llmBase import APIConfig, LLM_Base


# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a News Intelligence article-sorting classifier for a trading and investment signal system.

TASK: Read the article and extract ONE explicit primary event, then tag it under the cluster it belongs to.

## MODEL RULES
- Return ONLY a valid JSON object matching the provided schema. No markdown formatting, no extra text.
- Select exactly one primary event. Do not merge a historical fact, current condition, and future plan into one event.
- Extract only information explicitly stated in the article. Never infer causes, market impacts, strategic importance, or likely outcomes.
- Every field option list is separated by "|" with no space. Pick exactly one option per field, copied exactly as written.
- "Other" means the field's cluster does not apply to this article, or no listed option fits. Set every field that is not the article's main cluster to "Other".
- Only ONE of the five tag fields (demand_side_tag, supply_side_tag, objective_event_tag, policy_change_tag, self_entity_tag) should be a real option; the other four must be "Other".
- Never invent numbers, dates, or entities not present in the text.

## FIELDS (10 total)
- article_type: option, one of the article's format. Options: News report|Official announcement|Regulatory document|Company release|Analysis-opinion|Interview|Research report|Other
- target_sector: option, the primary domain or sector this event impacts directly. Options: Policy|Macro|Finance|Computer Science|Heavy Industry|Mineral|Agriculture|Logistics|Geopolitical|Society|Other
  Sector definitions:
  - Policy: government regulations, decrees, laws, subsidies, tariffs, trade restrictions, import/export quotas, zoning, pending legislation.
  - Macro: economy-wide conditions, inflation/deflation, interest rates, currency/FX moves, sovereign debt, trade balance swings, business cycle phases.
  - Finance: commercial banks, investment banking, stock/commodity exchanges, capital markets, securities, lending, insurance.
  - Computer Science: software, AI platforms/models, cloud computing, semiconductors, data infrastructure, IT services, telecom, cybersecurity.
  - Heavy Industry: manufacturing, industrial machinery, construction materials, chemicals, capital goods, energy (oil/gas/power utilities).
  - Mineral: metals and mining, precious metals, critical minerals, ore exploration and extraction.
  - Agriculture: farming, crops, livestock, fertilizer/soil inputs, food and beverage processing, forestry.
  - Logistics: freight transport, maritime shipping, ports and terminals, trucking, air cargo, rail, warehousing.
  - Geopolitical: interstate conflict, war, border disputes, international sanctions, political instability, elections.
  - Society: demographics, migration, labor and wage trends, public health, education, cultural and social behavior shifts.
  - Other: unlisted or non-applicable sector.
- main_event_summary: one factual sentence with actor, action, object when stated (<=35 words).
- event_status: option, current stage of the event. Options: Proposed|Announced|Approved|Signed|Started|Ongoing|Completed|Delayed|Cancelled|Disputed|Other
- event_date: MM-DD-YYYY, or "_" if no explicit event date is stated. Do not use publication date unless explicitly the event date.
- demand_side_tag: option, tag under the "Demand side" cluster describing the primary opportunity driver. Options: Consumer trend|Media-driven trend|Industry trend|Seasonal trend|Generational trend|Income-class trend|Religious/ethical trend|Regional/geographic trend|Wage growth|Wage decline|Purchasing power change|Income inequality/distribution shift|Informal economy wage|Remittance inflow|Investment/dividend income|Population growth/decline|Age structure shift|Urban vs rural shift|Migration|Household size change|Gender ratio shift|Ethnic composition shift|Disability/health population change|Habit change|Cultural/religious event|Health/diet shift|Education level shift|Digital penetration|Tourism-driven demand|E-commerce adoption|Sharing economy adoption|Direct substitute price change|Complement price change|Cross-industry substitute|Self-production substitute|Rental vs own substitution|Import substitution|Generic vs branded substitution|New buyer entry|Buyer exit|Buyer concentration|Buyer bargaining power shift|Group purchasing/co-op formation|Institutional buyer entry|Price expectation|Shortage expectation|Future demand forecast|Anticipatory buying before policy change|Expectation of tech disruption|Other
  Definition and example for each option (subtag: example):
  Trends:
    - Consumer trend: fashion shift, health/wellness shift, sustainability preference, premiumization vs downgrading
    - Media-driven trend: viral post, influencer endorsement, ad campaign, documentary/expose effect
    - Industry trend: B2B tech adoption, green transition demand, digitalization demand, outsourcing trend
    - Seasonal trend: holiday spike, off-season dip
    - Generational trend: Gen Z vs older generation preference shift
    - Income-class trend: luxury vs budget demand shift
    - Religious/ethical trend: veganism, halal/kosher demand
    - Regional/geographic trend: urban core vs suburb shift
  National avg salary:
    - Wage growth: minimum wage hike, sector wage rise, bonus/overtime increase
    - Wage decline: layoff wave, wage freeze, part-time shift
    - Purchasing power change: real wage vs inflation gap, subsidy/welfare payout effect
    - Income inequality/distribution shift: income gap widening/narrowing
    - Informal economy wage: gig/informal sector wage change
    - Remittance inflow: overseas worker income
    - Investment/dividend income: non-wage income change
  Demography:
    - Population growth/decline: birth rate, death rate, baby boom/bust
    - Age structure shift: aging population, youth bulge, dependency ratio
    - Urban vs rural shift: urbanization rate, rural depopulation
    - Migration: immigration, refugee inflow, labor migration, brain drain
    - Household size change: nuclear family rise, multi-generation household
    - Gender ratio shift: sex ratio imbalance effect on demand
    - Ethnic composition shift: demographic mix change
    - Disability/health population change: aging-related health need shift
  Lifestyle/culture:
    - Habit change: diet shift, work-from-home habit, digital payment adoption
    - Cultural/religious event: festival demand spike, religious dietary restriction
    - Health/diet shift: low-carb trend, organic demand, supplement demand
    - Education level shift: literacy rise, higher education enrollment
    - Digital penetration: mobile/internet adoption rate
    - Tourism-driven demand: visitor inflow/outflow effect
    - E-commerce adoption: online shopping shift
    - Sharing economy adoption: rental/pooled-use platform growth
  Substitution:
    - Direct substitute price change: pork vs chicken, coal vs gas, cotton vs polyester
    - Complement price change: car price vs fuel demand, phone vs accessory demand
    - Cross-industry substitute: plant protein vs animal protein, EV vs ICE, renewable vs fossil
    - Self-production substitute: home solar vs grid power, home garden vs market produce
    - Rental vs own substitution: car rental/share vs ownership, housing rent vs buy
    - Import substitution: domestic vs imported good
    - Generic vs branded substitution: generic vs branded goods
  Number of buyers:
    - New buyer entry: new export market, new trade bloc member
    - Buyer exit: sanctioned country drops out, major client bankruptcy
    - Buyer concentration: single dominant buyer, fragmented small buyers
    - Buyer bargaining power shift: large buyer squeezing price
    - Group purchasing/co-op formation: buyer alliance formed
    - Institutional buyer entry: government stockpiling, sovereign wealth fund
  Expectation:
    - Price expectation: hoarding, panic buying, pre-emptive stocking
    - Shortage expectation: rumor-driven scarcity fear
    - Future demand forecast: industry outlook, government projection
    - Anticipatory buying before policy change: stock-up ahead of tariff/ban
    - Expectation of tech disruption: waiting for next-gen product
  - Other: cluster not applicable to this article

- supply_side_tag: option, tag under the "Supply side" cluster describing the primary opportunity driver. Options: New production method|Automation/efficiency upgrade|Tech breakthrough|Tech transfer/licensing|Tech obsolescence|Patent expiry|R&D investment change|Capacity expansion|Capacity cut/closure|Utilization rate change|Yield change|Planned maintenance shutdown|Strategic reserve release/drawdown|Capacity relocation|Vertical integration|Input price change|Transportation problem|Logistics bottleneck|Labor cost/shortage|Producer financing/credit access|Single-source supplier risk|Resource constraint|FX pass-through on input|Packaging material cost|Shipping insurance cost|New entrant|Seller exit/bankruptcy|Consolidation|Cartel/coordination|State-owned enterprise intervention|Informal/black market supply|Joint venture formation|Producer price expectation|Input cost forecast|Pre-emptive stockpiling by producer|Other
  Definition and example for each option (subtag: example):
  Technology:
    - New production method: automation line, precision farming, gene-edited crop, new smelting process
    - Automation/efficiency upgrade: robotics, AI-assisted production, process digitization
    - Tech breakthrough: new material, battery tech, new vaccine/drug production
    - Tech transfer/licensing: technology sold/licensed to new producer
    - Tech obsolescence: old method phased out
    - Patent expiry: opens supply to generic/new producers
    - R&D investment change: funding increase/decrease affecting future supply
  Productivity/capacity:
    - Capacity expansion: new factory, new mine, farmland expansion
    - Capacity cut/closure: plant shutdown, mine depletion, farmland conversion
    - Utilization rate change: overtime added/removed, idle capacity
    - Yield change: harvest yield, ore grade, livestock productivity
    - Planned maintenance shutdown: scheduled downtime
    - Strategic reserve release/drawdown: oil SPR release, grain reserve sale
    - Capacity relocation: offshoring/reshoring
    - Vertical integration: backward/forward integration changing capacity control
  Value chain:
    - Input price change: raw material, energy, feed, fertilizer cost
    - Transportation problem: freight rate spike, route block, port congestion, trucking shortage
    - Logistics bottleneck: container shortage, warehouse limit, cold-chain failure
    - Labor cost/shortage: strike, wage impact on cost, skilled labor shortage
    - Producer financing/credit access: loan availability for producers
    - Single-source supplier risk: dependency on one supplier
    - Resource constraint: water shortage, land availability limit
    - FX pass-through on input: imported input price change from currency move
    - Packaging material cost: packaging input price change
    - Shipping insurance cost: marine/cargo insurance premium change
  Number of sellers/market structure:
    - New entrant: new competitor, new exporting country
    - Seller exit/bankruptcy: producer shutdown, bankruptcy filing
    - Consolidation: M&A, monopoly/oligopoly forming
    - Cartel/coordination: OPEC-style cut, price-fixing
    - State-owned enterprise intervention: government-run producer entering/exiting market
    - Informal/black market supply: unregulated production or smuggling
    - Joint venture formation: partnership between producers
  Expectation:
    - Producer price expectation: sell-now vs hold-inventory
    - Input cost forecast: expected raw material trend
    - Pre-emptive stockpiling by producer: ahead of expected regulation
  - Other: cluster not applicable to this article

- objective_event_tag: option, tag under the "Objective event" cluster describing the primary opportunity driver. Options: Natural disaster|Disease/epidemic|Weather anomaly|Accident|Volcanic/geological event|Lockdown/movement restriction|Infrastructure failure|Space weather|Terrorism attack|Inflation/deflation|Business cycle phase|Interest rate environment|Currency/exchange rate move|Social behavior shift|Commodity super-cycle|Credit crunch/liquidity crisis|Sovereign debt crisis|Labor force shrinkage|Asset bubble/crash|Trade balance shift|Other
  Definition and example for each option (subtag: example):
  Exogenous shock:
    - Natural disaster: flood, earthquake, drought, typhoon, wildfire
    - Disease/epidemic: human pandemic, livestock disease (ASF, avian flu, FMD), crop disease/pest
    - Weather anomaly: El Nino/La Nina, unseasonal frost, heatwave, unusual rainfall
    - Accident: factory fire, chemical spill, mine collapse, ship collision
    - Volcanic/geological event: eruption, tectonic event affecting mining/aviation
    - Lockdown/movement restriction: quarantine, border closure
    - Infrastructure failure: power grid failure, dam failure
    - Space weather: solar flare affecting electronics/satellite
    - Terrorism attack: distinct from state war
  Macro condition:
    - Inflation/deflation: CPI trend, hyperinflation, deflation spiral
    - Business cycle phase: recession signal, expansion, stagflation
    - Interest rate environment: rate hike/cut cycle, credit tightening
    - Currency/exchange rate move: devaluation, appreciation, peg break
    - Social behavior shift: protest, strike wave, boycott, consumer confidence shift
    - Commodity super-cycle: multi-year boom/bust in commodity class
    - Credit crunch/liquidity crisis: bank lending freeze, market liquidity dry-up
    - Sovereign debt crisis: country default risk, debt crisis contagion
    - Labor force shrinkage: aging-driven macro labor supply drop
    - Asset bubble/crash: stock market crash, real estate bubble burst
    - Trade balance shift: current account deficit/surplus swing
  - Other: cluster not applicable to this article

- policy_change_tag: option, tag under the "Policy change" cluster describing the primary opportunity driver. Options: Tax/subsidy|Tariff/trade restriction|Regulation|Export/import ban or quota|Monetary policy|Industrial policy|Land use/zoning policy|Public health policy|Immigration/visa policy|Antitrust/competition policy|Pending legislation|Political instability|Sanctions|Geopolitical conflict|Policy reversal|International agreement withdrawal|Referendum/election outcome risk|Central bank independence risk|Other
  Definition and example for each option (subtag: example):
  New policy:
    - Tax/subsidy: VAT change, export tax, farm subsidy, green subsidy
    - Tariff/trade restriction: import tariff, anti-dumping duty, quota
    - Regulation: environmental rule, safety standard, labor law, data privacy law
    - Export/import ban or quota: rice export ban, mineral quota, tech export control
    - Monetary policy: rate change, reserve requirement, QE/QT
    - Industrial policy: strategic sector support, national champion buildup
    - Land use/zoning policy: rezoning, farmland protection, industrial zone designation
    - Public health policy: vaccination mandate, food safety standard
    - Immigration/visa policy: labor supply/demand effect
    - Antitrust/competition policy: explicit market-structure regulation
  Policy risk:
    - Pending legislation: bill under debate, election-driven uncertainty
    - Political instability: coup, government collapse, leadership transition
    - Sanctions: trade sanction, financial sanction, blacklist
    - Geopolitical conflict: war, border dispute, embargo, military standoff
    - Policy reversal: new administration undoes prior policy
    - International agreement withdrawal: leaving trade pact/climate accord
    - Referendum/election outcome risk: referendum or election outcome risk
    - Central bank independence risk: central bank independence risk
  - Other: cluster not applicable to this article

- self_entity_tag: option, tag under the "Self-entity" cluster describing the primary opportunity driver. Options: Fraud|Labor violation|Environmental violation|Data breach/privacy violation|Animal welfare violation|Bribery/corruption|IP theft/counterfeiting|Launch|Product recall|Regulatory approval/certification|Rebranding/repositioning|Loan default|Bond default|Debt restructuring|Credit rating change|Insolvency/liquidation|Foreign-currency debt mismatch risk|Leadership change/scandal|Legal dispute/lawsuit|Internal conflict|Cybersecurity breach|PR crisis|Succession crisis|Whistleblower event|Other
  Definition and example for each option (subtag: example):
  Ethic problem:
    - Fraud: accounting fraud, Ponzi scheme, falsified data
    - Labor violation: forced labor, unsafe working condition, child labor
    - Environmental violation: illegal dumping, deforestation, emissions violation
    - Data breach/privacy violation: leaked user data, unauthorized data use
    - Animal welfare violation: livestock/poultry mistreatment, illegal wildlife trade
    - Bribery/corruption: distinct from fraud
    - IP theft/counterfeiting: by entity itself
  New product:
    - Launch: new product/service release
    - Product recall: safety recall, quality defect recall
    - Regulatory approval/certification: drug approval, patent grant, license issued
    - Rebranding/repositioning: rebranding/repositioning
  Default/debt:
    - Loan default: bank loan, trade credit default
    - Bond default: corporate or sovereign bond default
    - Debt restructuring: rollover, haircut, bailout
    - Credit rating change: downgrade/upgrade
    - Insolvency/liquidation: company wound down
    - Foreign-currency debt mismatch risk: foreign-currency debt mismatch risk
  Drama:
    - Leadership change/scandal: CEO resignation, insider trading, corruption
    - Legal dispute/lawsuit: patent case, antitrust case, contract dispute
    - Internal conflict: strike, management infighting, mass resignation
    - Cybersecurity breach: hack, ransomware, system outage
    - PR crisis: social media backlash, reputation damage event
    - Succession crisis: family-run business leadership transition
    - Whistleblower event: internal exposure of wrongdoing
  - Other: cluster not applicable to this article

## EXAMPLE
Input title: "Vietnam approves $2B port expansion loan"
Input text: "[1] The Vietnamese government approved a $2 billion government-backed loan on 2025-03-01 to expand Cai Mep port capacity, aiming to increase container throughput after months of congestion caused by rising export demand."
Output:
{
  "article_type": "News report",
  "target_sector": "Logistics",
  "main_event_summary": "Vietnamese government approves $2B loan to expand Cai Mep port capacity",
  "event_status": "Approved",
  "event_date": "03-01-2025",
  "demand_side_tag": "Other",
  "supply_side_tag": "Capacity expansion",
  "objective_event_tag": "Other",
  "policy_change_tag": "Other",
  "self_entity_tag": "Other"
}
"""

USER_PROMPT_TEMPLATE = """
Classify and extract the primary event from this article.

TITLE: {title}

ARTICLE:
{text}

"""

SCHEMA = {
    "type": "object",
    "properties": {
        "article_type": {
            "type": "string",
            "enum": [
                "News report",
                "Official announcement",
                "Regulatory document",
                "Company release",
                "Analysis-opinion",
                "Interview",
                "Research report",
                "Other",
            ],
        },
        "target_sector": {
            "type": "string",
            "enum": [
                "Policy",
                "Macro",
                "Finance",
                "Computer Science",
                "Heavy Industry",
                "Mineral",
                "Agriculture",
                "Logistics",
                "Geopolitical",
                "Society",
                "Other",
            ],
        },
        "main_event_summary": {"type": "string"},
        "event_status": {
            "type": "string",
            "enum": [
                "Proposed",
                "Announced",
                "Approved",
                "Signed",
                "Started",
                "Ongoing",
                "Completed",
                "Delayed",
                "Cancelled",
                "Disputed",
                "Other",
            ],
        },
        "event_date": {"type": "string"},
        "demand_side_tag": {
            "type": "string",
            "enum": [
                "Consumer trend",
                "Media-driven trend",
                "Industry trend",
                "Seasonal trend",
                "Generational trend",
                "Income-class trend",
                "Religious/ethical trend",
                "Regional/geographic trend",
                "Wage growth",
                "Wage decline",
                "Purchasing power change",
                "Income inequality/distribution shift",
                "Informal economy wage",
                "Remittance inflow",
                "Investment/dividend income",
                "Population growth/decline",
                "Age structure shift",
                "Urban vs rural shift",
                "Migration",
                "Household size change",
                "Gender ratio shift",
                "Ethnic composition shift",
                "Disability/health population change",
                "Habit change",
                "Cultural/religious event",
                "Health/diet shift",
                "Education level shift",
                "Digital penetration",
                "Tourism-driven demand",
                "E-commerce adoption",
                "Sharing economy adoption",
                "Direct substitute price change",
                "Complement price change",
                "Cross-industry substitute",
                "Self-production substitute",
                "Rental vs own substitution",
                "Import substitution",
                "Generic vs branded substitution",
                "New buyer entry",
                "Buyer exit",
                "Buyer concentration",
                "Buyer bargaining power shift",
                "Group purchasing/co-op formation",
                "Institutional buyer entry",
                "Price expectation",
                "Shortage expectation",
                "Future demand forecast",
                "Anticipatory buying before policy change",
                "Expectation of tech disruption",
                "Other",
            ],
        },
        "supply_side_tag": {
            "type": "string",
            "enum": [
                "New production method",
                "Automation/efficiency upgrade",
                "Tech breakthrough",
                "Tech transfer/licensing",
                "Tech obsolescence",
                "Patent expiry",
                "R&D investment change",
                "Capacity expansion",
                "Capacity cut/closure",
                "Utilization rate change",
                "Yield change",
                "Planned maintenance shutdown",
                "Strategic reserve release/drawdown",
                "Capacity relocation",
                "Vertical integration",
                "Input price change",
                "Transportation problem",
                "Logistics bottleneck",
                "Labor cost/shortage",
                "Producer financing/credit access",
                "Single-source supplier risk",
                "Resource constraint",
                "FX pass-through on input",
                "Packaging material cost",
                "Shipping insurance cost",
                "New entrant",
                "Seller exit/bankruptcy",
                "Consolidation",
                "Cartel/coordination",
                "State-owned enterprise intervention",
                "Informal/black market supply",
                "Joint venture formation",
                "Producer price expectation",
                "Input cost forecast",
                "Pre-emptive stockpiling by producer",
                "Other",
            ],
        },
        "objective_event_tag": {
            "type": "string",
            "enum": [
                "Natural disaster",
                "Disease/epidemic",
                "Weather anomaly",
                "Accident",
                "Volcanic/geological event",
                "Lockdown/movement restriction",
                "Infrastructure failure",
                "Space weather",
                "Terrorism attack",
                "Inflation/deflation",
                "Business cycle phase",
                "Interest rate environment",
                "Currency/exchange rate move",
                "Social behavior shift",
                "Commodity super-cycle",
                "Credit crunch/liquidity crisis",
                "Sovereign debt crisis",
                "Labor force shrinkage",
                "Asset bubble/crash",
                "Trade balance shift",
                "Other",
            ],
        },
        "policy_change_tag": {
            "type": "string",
            "enum": [
                "Tax/subsidy",
                "Tariff/trade restriction",
                "Regulation",
                "Export/import ban or quota",
                "Monetary policy",
                "Industrial policy",
                "Land use/zoning policy",
                "Public health policy",
                "Immigration/visa policy",
                "Antitrust/competition policy",
                "Pending legislation",
                "Political instability",
                "Sanctions",
                "Geopolitical conflict",
                "Policy reversal",
                "International agreement withdrawal",
                "Referendum/election outcome risk",
                "Central bank independence risk",
                "Other",
            ],
        },
        "self_entity_tag": {
            "type": "string",
            "enum": [
                "Fraud",
                "Labor violation",
                "Environmental violation",
                "Data breach/privacy violation",
                "Animal welfare violation",
                "Bribery/corruption",
                "IP theft/counterfeiting",
                "Launch",
                "Product recall",
                "Regulatory approval/certification",
                "Rebranding/repositioning",
                "Loan default",
                "Bond default",
                "Debt restructuring",
                "Credit rating change",
                "Insolvency/liquidation",
                "Foreign-currency debt mismatch risk",
                "Leadership change/scandal",
                "Legal dispute/lawsuit",
                "Internal conflict",
                "Cybersecurity breach",
                "PR crisis",
                "Succession crisis",
                "Whistleblower event",
                "Other",
            ],
        },
    },
    "required": [
        "article_type",
        "target_sector",
        "main_event_summary",
        "event_status",
        "event_date",
        "demand_side_tag",
        "supply_side_tag",
        "objective_event_tag",
        "policy_change_tag",
        "self_entity_tag",
    ],
    "additionalProperties": False,
}


# ── Config ─────────────────────────────────────────────────────────────────────

GENERAL_CONFIG = GeneralLLMConfig().from_dict(
    {
        "verbose": True,
    }
)
GENERAL_CONFIG.prompt_token_update(
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT_TEMPLATE,
)

GENERAL_CONFIG.prompt_token_update(
    system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT_TEMPLATE
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


# ── Summarizer ─────────────────────────────────────────────────────────────────


class ArticleSummarizer(LLM_Base):
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

    def _build_prompt(self, article: Article) -> tuple[str, str]:
        """Build the system and user prompt for one article."""

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

        return SYSTEM_PROMPT, user

    @staticmethod
    def _apply_to_article(article: Article, parsed: dict) -> Article:
        keys = [
            "article_type",
            "target_sector",
            "main_event_summary",
            "event_status",
            "event_date",
            "demand_side_tag",
            "supply_side_tag",
            "objective_event_tag",
            "policy_change_tag",
            "self_entity_tag",
        ]

        if article.no_save_pool is None:
            article.no_save_pool = {}

        # Extract values, defaulting to '_' if the LLM omitted a key
        values = []
        for key in keys:
            val = parsed.get(key, "_")
            if not isinstance(val, str):
                val = "_"
            # Strip whitespace and ensure empty strings become '_'
            values.append(val.strip() or "_")

        article.no_save_pool["data_card"] = dict(zip(keys, values))

        return article
