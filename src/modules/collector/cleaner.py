from typing import List
import re
from src.modules.collector.cleaner_junkDB import JunkCache
from src.core.types import Article, GeneralLLMConfig, LLMConfig, GENConfig
from src.modules.llm.llmBase import LLM_Base

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a text-cleaning assistant.
The article text is split into numbered lines (0-indexed).
Your task: identify which line numbers contain "junk" (not part of the main article).

# Junk includes:
- Cookie/privacy notices, consent dialogs
- Newsletter signup prompts, ads, sponsored content
- Navigation menus, breadcrumbs, footer/header text
- Login/register prompts, "Read more", "Related articles"
- Social sharing prompts, image credits, copyright notices
- UI labels, user comments, disclaimers, legal notices
- Author bios, sidebars, unrelated captions/headings
- "Member-only story", paywall notices
- Website footers with unrelated news links
- Unrelated news snippets appended to articles

# DO NOT remove:
- Main article title/subtitle/body
- Quotes, image captions, publication date
- Section headings belonging to the main article

# Output format:
Return an array of LINE NUMBERS (0-indexed) that should be removed.
- Empty array [] if no junk detected
- Numbers must be in ascending order
- Only include numbers that actually exist in the text

# Examples:

Input:
Title: Breaking news
[0] The government announced measures.
[1] These cookies are set by us to enhance website performance.
[2] Officials say the plan takes effect next month.
[3] Bình luận
[4] Xin vui lòng gõ tiếng Việt có dấu

Output:
{"remove_lines": [1, 3, 4]}

Input:
Title: Quarterly results
[0] The quarterly results exceeded expectations.
[1] Revenue grew 12% year over year.
[2] Profit margins improved significantly.

Output:
{"remove_lines": []}

# Output ONLY valid JSON:
{"remove_lines": [0, 1, 2, ...]}
"""

USER_PROMPT_TEMPLATE = """
# Title:
{title}
# Text:
{numbered_text}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "remove_lines": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0}
        }
    },
    "required": ["remove_lines"],
}


# ── Config ─────────────────────────────────────────────────────────────────────

GENERAL_CONFIG = GeneralLLMConfig().from_dict(
    {"model_path": "./models/Ministral-3-8B-Instruct-2512-Q5_K_M.gguf", "max_input_text_chars": 50_000, "verbose": True}
)

GENERAL_CONFIG.prompt_token_update(
    system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT_TEMPLATE
)

LLM_CONFIG = LLMConfig().from_dict(
    {
        "n_gpu_layers": -1,
        "n_ctx": GENERAL_CONFIG.calc_max_n_ctx(),
        "n_batch": 1024,
        "n_threads": 12,
        "verbose": False,
    }
)

GEN_CONFIG = GENConfig().from_dict(
    {
        "max_tokens": LLM_CONFIG.n_ctx,
        "temperature": 0.0,
        "repeat_penalty": 1.0,
        "stop": ["</s>", "[/INST]"],
    }
)

# ── Helpers ────────────────────────────────────────────────────────────────
def split_text_to_sentences(text: str) -> list[str]:
    """
    1. Replace all newlines/carriage returns with a single space.
    2. Split by sentence-ending punctuation: ..., ., ?, !
    3. Keep punctuation attached to the sentence.
    """
    # 1. Normalize all newlines to spaces
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # 2. Split after '...' OR after '.', '?', '!' followed by whitespace
    # Order matters: '...' must be checked before '.' so the dots inside '...' don't trigger early splits
    raw_sentences = re.split(r'(?<=\.\.\.)\s+|(?<=[.?!])\s+', text)
    
    # 3. Clean and filter out empty strings
    return [s.strip() for s in raw_sentences if s.strip()]
def number_lines(lines: list[str]) -> str:
    """Convert text to numbered format: [0] line1\n[1] line2\n..."""
    return '\n'.join(f"[{i}] {line}" for i, line in enumerate(lines))


# ── Filter ─────────────────────────────────────────────────────────────────

class ArticleCleaner(LLM_Base):
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
        self.junk_cache = JunkCache()

    
    def _apply_to_article(self, article: Article, parsed: dict) -> Article:
        """Remove junk lines from article based on line indices."""
        remove_indices = parsed.get("remove_lines", [])
        text = article.original_text
        
        if not remove_indices:
            return article

        # Split text into lines
        lines = article.no_save_pool["residual_lines"]
        total_lines = len(lines)
        if total_lines == 0:
            print("[ArticleCleaner] No lines to process.")
            return article
        # Validate and sanitize indices
        valid_indices = set()
        for idx in remove_indices:
            if not isinstance(idx, int):
                continue
            if 0 <= idx < total_lines:
                valid_indices.add(idx)
            else:
                print(f"[ArticleCleaner] WARN: Index {idx} out of range [0, {total_lines})")
        
        if not valid_indices:
            return article

        # Log what's being removed
        sorted_indices = sorted(valid_indices)
        print(f"[ArticleCleaner] Removing {len(sorted_indices)} line(s): {sorted_indices}")
        for idx in sorted_indices: 
            preview = lines[idx][:80] + "..." if len(lines[idx]) > 80 else lines[idx]
            print(f"  [{idx}] {preview!r}")
        self.junk_cache.add_lines([lines[idx] for idx in sorted_indices])
            
        # Filter lines
        cleaned_lines = [line for i, line in enumerate(lines) if i not in valid_indices]
        
        # Rejoin
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = cleaned_text.strip()

        print(f"[clean_article] {len(text)} chars -> {len(cleaned_text)} chars ({len(text) - len(cleaned_text)} removed)")
        
        article.original_text = cleaned_text
        article.processed_stage = "cleaned"
        return article

    def _build_prompt(self, article: Article) -> tuple[str, str]:
        text = self._truncate_to_budget(article.original_text)
        lines = split_text_to_sentences(text)
        cache_junk = self.junk_cache.classify_lines(lines)
        if cache_junk:
            print(f"[Cache] Flagged {len(cache_junk)} line(s) as junk")
            for idx in sorted(cache_junk):
                preview = lines[idx][:80] + "..." if len(lines[idx]) > 80 else lines[idx]
                print(f"  [{idx}] {preview!r}")

        residual_lines = []
        residual_to_orig = {}
        for i, line in enumerate(lines):
            if i not in cache_junk:
                residual_to_orig[len(residual_lines)] = i
                residual_lines.append(line)

        article.no_save_pool["residual_lines"] = residual_lines

        numbered_text = number_lines(residual_lines)
        user = USER_PROMPT_TEMPLATE.format(
            title=article.title,
            numbered_text=numbered_text,
        )
        return SYSTEM_PROMPT, user
