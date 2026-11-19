from src.core.types import Article, GeneralLLMConfig, LLMConfig, GENConfig
from src.modules.llm.llmBase import LLM_Base

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a multi-language strict news-value filter for economic and financial analysis . 
You MUST output ONLY a valid JSON object. No markdown formatting. No extra text.

# OUTPUT FORMAT (Exactly this):
{
  "keep": false,
  "need_cleaning": false,
  "reason": "Short reason if false"
}

# STEP 1: Decide "keep" (true or false)
Set "keep": true ONLY if the article contains valuable information for economic, financial, market, policy, research, large-scale weather impact, improvement, trends, business analysis or news that impacts the economic landscape.
Valuable topics include: market/stock moves, company earnings, supply chain, macroeconomics, regulation, infrastructure,  tech breakthroughs with clear commercial impact, potential medium to large disasters/plague/pandemics that will or may impact the economic landscape, impending crises or potential trends that may impact the economic landscape.

Set "keep": false if the text is:
- Junk: headline only, login walls, cookie notices, raw HTML, CAPTCHA.
- Low-value: isolated local accidents, crime, sports, entertainment, gossip, memes.
- Fluff: pure opinion, marketing, listicles, generic advice.
(Rule: If unsure, set "keep": false).

# STEP 2: Decide "need_cleaning" (true or false)
- If "keep" is false, "need_cleaning" MUST be false.
- If "keep" is true, set "need_cleaning": true if the text contains web noise OR unrelated text.

Triggers for "need_cleaning": true:
- Web noise: HTML tags (<div>, <script>), cookie banners, "Subscribe", comments, footer boilerplate.
- "Read Next" blocks: Phrases like "Read Next:", "You may also like:", "Related Articles:", "More from our network:", "Up Next:", or lists of unrelated headlines at the end or middle of the text.
- Unrelated text: Paragraphs about a completely different topic, random ads, or off-topic stories injected into the article.

# STEP 3: Decide "reason" (string)
- If "keep": false, write a short 3-6 word reason (e.g., "Headline only, no body", "Isolated local accident", "Raw HTML navigation").
- If "keep": true, set "reason": "".

# EXAMPLES:

Input: "Breaking: Stock Market Hits Record High"
Output: {"keep": false, "need_cleaning": false, "reason": "Headline only, no body"}

Input: "<div class='menu'><a href='/home'>Home</a></div>"
Output: {"keep": false, "need_cleaning": false, "reason": "Raw HTML navigation"}

Input: "The company reported a 12% increase in quarterly revenue, driven by strong cloud demand. Executives expect continued growth next year."
Output: {"keep": true, "need_cleaning": false, "reason": ""}

Input: "The city council approved a new infrastructure bill, allocating $2 million for road repairs. Read Next: 5 Tips for Better Sleep Tonight."
Output: {"keep": true, "need_cleaning": true, "reason": ""}

Input: "Two cars collided on a highway, leaving one person dead. Police are investigating."
Output: {"keep": false, "need_cleaning": false, "reason": "Isolated accident, no economic impact"}

Input: "A major port suspended operations for 48 hours after an accident, delaying container flows and affecting regional exporters."
Output: {"keep": true, "need_cleaning": false, "reason": ""}

Input: "Apple reported a 5% drop in iPhone sales due to supply chain issues. You may also like: Taylor Swift announces new tour dates. Analysts expect Apple's stock to recover by Q4."
Output: {"keep": true, "need_cleaning": true, "reason": ""}

# FINAL INSTRUCTION:
Analyze the input text and output ONLY the JSON object. Do not wrap the output in ```json ``` code blocks.
"""
# You are a strict news-value filter for economic and financial analysis.
# Output valid JSON only. No markdown. No explanation outside the JSON.

# # Your job:
# Decide whether the article text is worth keeping for downstream economic, financial, policy, business, market, industry, infrastructure, technology, science, or operational analysis.
# Be conservative.
# Newsworthy does NOT mean valuable.

# A single isolated event is usually NOT valuable unless it has broader economic, policy, market, legal, operational, scientific, or infrastructure implications stated in the text.

# # Return exactly this JSON schema:

# {
#   "keep": true,
#   "need_cleaning": false,
#   "reason": ""
# }

# ## Field rules:
# - keep: true or false.
# - need_cleaning: true or false.
# - reason: required only when keep = false; otherwise "".

# Use the title only as reference for matching the correct main article body.
# Do not invent facts not present in the text.

# # STEP 1 — Determine whether the text contains a real article body.
# Set keep = false if the text is primarily or only:
# - headline with no body
# - cookie/privacy notice
# - login wall, menu, navigation, footer, category links
# - raw HTML, CSS, or JavaScript
# - CAPTCHA or bot check text
# - machine-readable junk with no readable article
# - broken extraction with only fragments and no coherent article

# If keep = false here:
# - set need_cleaning = false
# - set reason to one short phrase
# - stop

# # STEP 2 — Determine whether the article has economic-analysis value.
# Keep the article only if it contains information useful for at least one of these:
# - prices updates, commodities, stock, bond, or currency market movements
# - company performance, earnings, revenue, margin, costs, financing
# - industry conditions, supply chain, pricing, demand, capacity, competition
# - macroeconomics, policy, regulation, trade, taxation, public spending
# - infrastructure, energy, transport, logistics, real estate, labor markets
# - science, technology, health, or legal developments with broader commercial, operational, regulatory, or adoption implications
# - events, trends that can support a broader forward-looking hypothesis beyond the single incident

# Set keep = false for content that is mainly:
# - entertainment, celebrity, lifestyle, fashion, beauty, gossip
# - sports, gaming, astrology, memes, viral content
# - purely emotional or sensational stories
# - crime blotter, local accidents, isolated disasters, isolated deaths, isolated fires, isolated human-interest stories
# - opinion, rant, commentary, or reaction with little new factual information
# - heavy promotional or brand-marketing content
# - listicles, trivia, quizzes, generic advice
# - local event coverage with no broader economic or policy relevance


# If the article is concrete but only about one isolated event, set keep = false.

# If keep = false here:
# - set need_cleaning = false
# - set reason to one short phrase explaining why it lacks downstream value
# - stop

# # STEP 3 — Detect cleaning need.
# After identifying the main article body, inspect the full text.
# Set need_cleaning = true if any junk, extraction noise, or off-topic material appears anywhere, especially after the main article begins.

# Trigger need_cleaning = true if any of these appear:
# - HTML tags like <div>, <script>, <style>, <a>, <iframe>
# - CSS, JavaScript, JSON blobs, tracking code
# - cookie/privacy/subscription banners
# - CAPTCHA or bot-verification language
# - login/register prompts
# - newsletter prompts
# - hashtags or emoji-heavy blocks
# - "Read more", "Related", "Recommended", "You may also like", "Next", "More stories", "In other news"
# - gallery/photo/video instructions unrelated to the article body
# - duplicated paragraphs
# - long author bio, site boilerplate, copyright/footer text
# - a block about a different event, person, company, or subject
# - comment-section text
# - transcript fragments or broken extraction artifacts

# If none of the above are present and the text stays on the same main topic:
# - need_cleaning = false

# # STEP 4 — Final consistency rules.
# - If keep = false, then need_cleaning must be false and reason must not be empty.
# - If keep = true, reason must be "".
# - If uncertain whether an article has real downstream value, choose keep = false.
# - Do not keep articles just because they contain names, numbers, or a dramatic event.
# - Do not confuse public-interest value with economic-analysis value.

# # Examples:

# Text: "Breaking: Stock Market Hits Record High"
# Output:
# "keep":false,"need_cleaning":false,"reason":"Headline only, no body text"}

# Text: "<div class='menu'><a href='/home'>Home</a></div>"
# Output:
# {"keep":false,"need_cleaning":false,"reason":"Raw HTML/navigation text"}

# Text: "The company reported a 12% increase in quarterly revenue, driven by strong cloud demand. Executives expect continued growth next year."
# Output:
# {"keep":true,"need_cleaning":false,"reason":""}

# Text: "Subscribe to our newsletter! <script>trackClick()</script> The city council approved a new infrastructure bill, allocating $2 million for road repairs. Accept Cookies"
# Output:
# {"keep":true,"need_cleaning":true,"reason":""}

# Text: "Scientists announced a battery breakthrough, a 40% gain in energy density using solid-state design. Read more: 5 Tips for Better Sleep Tonight."
# Output:
# {"keep":true,"need_cleaning":true,"reason":""}

# Text: "Two cars collided on a highway, leaving one person dead and several injured. Police are investigating the cause."
# Output:
# {"keep":false,"need_cleaning":false,"reason":"Isolated accident with no broader economic implication"}

# Text: "A major port suspended operations for 48 hours after an accident, delaying container flows and affecting regional exporters."
# Output:
# {"keep":true,"need_cleaning":false,"reason":""}

# Text: "Lawmakers proposed new trucking safety rules after a series of highway crashes, with compliance requirements for fleet operators starting next year."
# Output:
# {"keep":true,"need_cleaning":false,"reason":""}

# Only output the JSON object. No extra text.
# """

USER_PROMPT_TEMPLATE = """
# Title:
{title}

# Text:
{text}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        # "main_topic": {"type": "string"},
        "keep": {"type": "boolean"},
        "need_cleaning": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["keep", "need_cleaning"],
}

# ── Config ─────────────────────────────────────────────────────────────────────

GENERAL_CONFIG = GeneralLLMConfig().from_dict(
    {"model_path": "./models/Ministral-3-3B-Instruct-2512-Q5_K_M.gguf", "verbose": True}
)
GENERAL_CONFIG.prompt_token_update(
    system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT_TEMPLATE
)

LLM_CONFIG = LLMConfig().from_dict(
    {
        "n_ctx": GENERAL_CONFIG.calc_max_n_ctx(),  # MAX TOTAL context window for Mistral Q4_K_M per text.
        "verbose": False,  # llama_cpp verbose logging
    }
)

GEN_CONFIG = GENConfig().from_dict(
    {  # any
        "max_tokens": LLM_CONFIG.n_ctx,  # output Tokens
        "temperature": 0.1,  # low = deterministic / accurate
        "repeat_penalty": 1.05,
    }
)


# ── Filter ─────────────────────────────────────────────────────────────────
# region Class
class ArticleFilter(LLM_Base):
    """
    # Context manager (auto-unloads model on exit):
    with ArticleFilter() as s:
        results = s.filter_batch(articles)

    # Manual lifecycle:
    s = ArticleFilter(n_parallel=4)
    s.load()
    results = s.filter_batch(articles)
    s.unload()
    """

    def __init__(
        self,
        llm_config: LLMConfig = LLM_CONFIG,
        gen_config: GENConfig = GEN_CONFIG,
        general_config: GeneralLLMConfig = GENERAL_CONFIG,
        verbose: bool = False,
    ):
        general_config.verbose = verbose
        super().__init__(
            llm_config=llm_config,
            gen_config=gen_config,
            general_config=general_config,
            schema=SCHEMA,
        )

    @staticmethod
    def _apply_to_article(article: Article, parsed: dict) -> Article:
        """Write parsed LLM fields back onto the Article, leaving originals intact."""
        # article.no_save_pool["main_topic"] = parsed.get("main_topic", "")
        article.no_save_pool["keep"] = parsed.get("keep", False)
        article.no_save_pool["need_cleaning"] = parsed.get("need_cleaning", False)
        article.no_save_pool["filter_hint"] = parsed.get("hint", "")
        article.no_save_pool["filter_reason"] = parsed.get("reason", "")
        article.processed_stage = "filtered"  # mark as filtered
        return article

    def _build_prompt(self, article: Article) -> tuple[str, str]:
        text = self._truncate_to_budget(article.original_text)
        user = USER_PROMPT_TEMPLATE.format(
            title=article.title,
            text=text,
        )
        # Mistral instruct format
        return SYSTEM_PROMPT, user  # f"[INST] {SYSTEM_PROMPT}\n\n{user} [/INST]"
