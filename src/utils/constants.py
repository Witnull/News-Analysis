ENTITY_LABELS = {
    # ── Universal ────────────────────────────────────────────────────────────
    "person":               "Named individual: researcher, politician, executive, author, or scientist",
    "organization":         "Company, university, government body, NGO, institute, or international body",
    "location":             "Country, city, region, continent, ocean, or geographic place",
    "event":                "Conference, election, war, summit, announcement, or named milestone",
    "law or policy":        "Legislation, regulation, treaty, executive order, or government policy",
    "metric":               "Measured outcome, statistic, index, KPI, percentage, or numeric finding",
    "concept":              "Abstract idea, theory, methodology, doctrine, or named framework",

    # ── AI / Technology ──────────────────────────────────────────────────────
    "ai model":             "Machine learning model, LLM, neural network, or AI system",
    "technology":           "Software system, platform, programming language, framework, or technical tool",
    "dataset":              "Named dataset, benchmark, corpus, or data collection",
    "algorithm":            "Named computational method, mathematical procedure, or optimization technique",

    # ── Science / Research ───────────────────────────────────────────────────
    "research study":       "Published paper, clinical trial, survey, experiment, or systematic review",
    "scientific theory":    "Named hypothesis, model, or established scientific principle",
    "chemical compound":    "Named molecule, drug, element, polymer, or chemical substance",
    "biological entity":    "Gene, protein, organism, species, microbe, virus, or cell type",
    "medical condition":    "Disease, disorder, syndrome, symptom, or clinical diagnosis",
    "medical treatment":    "Drug, therapy, surgical procedure, vaccine, or clinical intervention",

    # ── Finance / Crypto / Stock ─────────────────────────────────────────────
    "financial instrument": "Stock, bond, ETF, derivative, option, or tradable security",
    "cryptocurrency":       "Digital currency, token, blockchain asset, or DeFi protocol",
    "financial metric":     "Price, market cap, revenue, earnings, interest rate, or valuation figure",
    "financial institution":"Bank, hedge fund, exchange, brokerage, central bank, or payment network",
    "economic indicator":   "GDP, inflation rate, unemployment rate, trade balance, or macro index",

    # ── Agriculture ──────────────────────────────────────────────────────────
    "crop or plant":        "Named agricultural crop, plant species, or cultivar",
    "agricultural input":   "Fertilizer, pesticide, herbicide, seed variety, or farming technology",
    "agricultural metric":  "Yield, harvest volume, acreage, crop price, or food security index",

    # ── Mining / Minerals ────────────────────────────────────────────────────
    "mineral or resource":  "Named mineral, metal, ore, fossil fuel, or natural resource",
    "mining operation":     "Named mine, extraction site, refinery, or resource project",

    # ── Geopolitics / Politics ───────────────────────────────────────────────
    "political party":      "Named political party, coalition, or electoral bloc",
    "geopolitical actor":   "Nation-state, alliance, bloc, or supranational body acting politically",
    "conflict":             "Named war, military operation, territorial dispute, or armed conflict",
    "sanction or tariff":   "Named economic sanction, trade tariff, embargo, or restriction",

    # ── Environment / Climate ─────────────────────────────────────────────────
    "climate phenomenon":   "Named climate event, weather pattern, or atmospheric condition",
    "environmental metric": "Carbon level, temperature anomaly, sea level, deforestation rate, or emission figure",
    "energy source":        "Named energy type: solar, wind, nuclear, coal, oil, gas, or hydrogen",
    "environmental policy": "Climate accord, carbon market, emissions target, or green regulation",
}


RELATION_LABELS = {
    # ── Causality & Impact ───────────────────────────────────────────────────
    "causes":               "One entity directly produces or leads to another",
    "improves":             "One entity increases performance, value, or capability of another",
    "reduces":              "One entity decreases, mitigates, or lowers another",
    "depends_on":           "One entity requires or relies on another to function",

    # ── Creation & Ownership ─────────────────────────────────────────────────
    "developed_by":         "Model, tool, drug, or technology created or invented by an entity",
    "owned_by":             "Asset, company, or resource owned or controlled by an entity",
    "funded_by":            "Entity receives financial or institutional support from another",
    "acquired_by":          "Entity purchased, merged with, or absorbed by another",

    # ── Publication & Research ───────────────────────────────────────────────
    "published_in":         "Research, finding, or article appears in a journal, venue, or outlet",
    "found_that":           "Study, experiment, or report produced a specific result or conclusion",
    "evaluated_on":         "Model, drug, or system tested against a dataset, trial, or benchmark",
    "collaborated_with":    "Entities worked jointly on a study, project, or initiative",

    # ── Usage & Application ──────────────────────────────────────────────────
    "used_in":              "Entity is applied or deployed within a specific domain or context",
    "treats":               "Drug, therapy, or intervention applied to a medical condition",
    "regulates":            "Law, policy, or body governs or controls another entity",
    "produces":             "Entity manufactures, extracts, grows, or generates another",

    # ── Comparison & Classification ──────────────────────────────────────────
    "compared_to":          "Entities are evaluated or contrasted against each other",
    "part_of":              "Entity is a component, member, or subset of a larger entity",
    "located_in":           "Entity resides in, originates from, or operates within a location",
    #"instance_of":          "Entity is a specific example or case of a broader category",

    # ── Finance & Markets ────────────────────────────────────────────────────
    "traded_on":            "Financial instrument listed or traded on an exchange or market",
    "invested_in":          "Entity allocates capital into another entity or asset",
    "affects_price_of":     "Event, policy, or entity influences the market price of another",
    # "pegged_to":            "Currency, token, or asset value tied to another reference asset",

    # ── Geopolitics & Environment ────────────────────────────────────────────
    "signed":               "Actor or party agreed to or ratified a treaty, accord, or policy",
    "imposed_on":           "Sanction, tariff, or restriction applied by one entity to another",
    # "emits":                "Entity produces a pollutant, greenhouse gas, or emission",
    "threatens":            "Entity poses a risk to another entity, ecosystem, or stability",
    
    # ── Knowledge & Definition ───────────────────────────────────────────────
    "defines":              "Entity formally specifies the meaning or scope of another",
    #"describes":            "Entity provides a characterization or explanation of another",
    "classifies":           "Entity assigns another to a category, taxonomy, or type system",
    # "measures":             "Entity quantifies, scores, or evaluates another",
    # "demonstrates":         "Entity shows evidence or proof of another entity or claim",
    # "contradicts":          "Entity opposes, refutes, or is in conflict with another finding",
    # "supports":             "Entity provides evidence, backing, or validation for another",
    "extends":              "Entity builds upon, generalizes, or expands the scope of another",
    # "derived_from":         "Entity originates, inherits, or is adapted from another",
    # "named_after":          "Entity received its name in reference to another person or entity",
    #"synonym_of":           "Entity refers to the same concept under a different name",
    "example_of":           "Entity is a concrete illustration or instance of a broader concept",
}


TYPE_COLORS = {
    # Universal
    "person":               "#AED6F1",   # light blue
    "organization":         "#A9DFBF",   # light green
    "location":             "#FDEBD0",   # peach
    "event":                "#FADBD8",   # rose
    "law or policy":        "#D5D8DC",   # cool gray
    "metric":               "#FAD7A0",   # amber
    "concept":              "#E8DAEF",   # lavender

    # AI / Tech
    "ai model":             "#F9E79F",   # yellow
    "technology":           "#FCF3CF",   # pale yellow
    "dataset":              "#A8D8EA",   # sky blue
    "algorithm":            "#D6EAF8",   # ice blue

    # Science / Research
    "research study":       "#D2B4DE",   # purple
    "scientific theory":    "#C9B1D0",   # mauve
    "chemical compound":    "#A9CCE3",   # steel blue
    "biological entity":    "#A2D9CE",   # teal
    "medical condition":    "#F1948A",   # salmon red
    "medical treatment":    "#F8C471",   # warm orange

    # Finance / Crypto
    "financial instrument": "#ABEBC6",   # mint
    "cryptocurrency":       "#82E0AA",   # emerald
    "financial metric":     "#F7DC6F",   # gold
    "financial institution":"#A9DFBF",   # green
    "economic indicator":   "#D5F5E3",   # pale mint

    # Agriculture
    "crop or plant":        "#ABEBC6",   # leaf green
    "agricultural input":   "#D5F5E3",   # pale green
    "agricultural metric":  "#F9E79F",   # harvest yellow

    # Mining / Minerals
    "mineral or resource":  "#CCD1D1",   # stone gray
    "mining operation":     "#AAB7B8",   # dark gray

    # Geopolitics / Politics
    "political party":      "#F1948A",   # warm red
    "geopolitical actor":   "#EC7063",   # bold red
    "conflict":             "#E74C3C",   # alert red
    "sanction or tariff":   "#F0B27A",   # burnt orange

    # Environment / Climate
    "climate phenomenon":   "#85C1E9",   # sky
    "environmental metric": "#7FB3D3",   # ocean blue
    "energy source":        "#A9CCE3",   # blue-gray
    "environmental policy": "#A2D9CE",   # seafoam
}
DEFAULT_COLOR = "#EAECEE"