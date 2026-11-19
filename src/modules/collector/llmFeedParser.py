from dataclasses import dataclass
import json
from pathlib import Path
import time


from bs4 import BeautifulSoup, Comment, Tag
from llama_cpp import LlamaGrammar
import trafilatura

from src.modules.llm.llmBase import LLM_Base
from src.core.types import GENConfig, GeneralLLMConfig, LLMConfig

JUNK_TAGS = [
    # scripting / styling / metadata
    "script",
    "style",
    "meta",
    "link",
    "noscript",
    "template",
    # embedded / interactive media
    "svg",
    "canvas",
    "iframe",
    "embed",
    "object",
    "param",
    "video",
    "audio",
    "source",
    "track",
    "picture",
    "img",
    "map",
    "area",
    # forms / inputs
    "form",
    "input",
    "button",
    "select",
    "option",
    "optgroup",
    "textarea",
    "label",
    "fieldset",
    "legend",
    "datalist",
    "output",
    "progress",
    "meter",
    # layout chrome (non-article regions)
    "nav",
    "header",
    "footer",
    "aside",
    # deprecated / obsolete tags
    "applet",
    "basefont",
    "bgsound",
    "blink",
    "marquee",
    "frame",
    "frameset",
    "noframes",
    "noembed",
    "isindex",
    "keygen",
    "dir",
    "xmp",
    "listing",
    "plaintext",
    # web components / misc
    "slot",
    "portal",
    "menuitem",
    "menu",
    "dialog",
    # ruby annotation (rarely article content, optional)
    "rp",
    "rt",
    "ruby",
]

TITLE_SELECTORS = [
    # class-based, common patterns
    '[class*="title"]',
    '[class*="Title"]',
    '[class*="headline"]',
    '[class*="Headline"]',
    '[class*="heading"]',
    '[class*="Heading"]',
    '[class*="entry-title"]',
    '[class*="post-title"]',
    '[class*="article-title"]',
    '[class*="page-title"]',
    '[class*="story-title"]',
    '[class*="content-title"]',
    '[class*="hero-title"]',
    '[class*="main-title"]',
    # id-based
    '[id*="title"]',
    '[id*="Title"]',
    '[id*="headline"]',
    '[id*="Headline"]',
    '[id*="heading"]',
    '[id*="Heading"]',
    '[id*="entry-title"]',
    '[id*="post-title"]',
    '[id*="article-title"]',
    '[id*="page-title"]',
    # itemprop / schema.org microdata
    '[itemprop="headline"]',
    '[itemprop="name"]',
    # ARIA role-based
    '[role="heading"]',
    # data-* attributes (JS-driven sites, React/Vue components)
    '[data-testid*="headline"]',
    '[data-testid*="title"]',
    '[data-component*="headline"]',
    '[data-role*="title"]',
    # semantic tags
    "h1",
    "header h1",
    "article h1",
]

CONTENT_SELECTORS = [
    # class-based, common patterns
    '[class*="content"]',
    '[class*="Content"]',
    '[class*="article"]',
    '[class*="Article"]',
    '[class*="post"]',
    '[class*="Post"]',
    '[class*="entry"]',
    '[class*="Entry"]',
    '[class*="body"]',
    '[class*="Body"]',
    '[class*="story"]',
    '[class*="Story"]',
    '[class*="main"]',
    '[class*="Main"]',
    '[class*="text"]',
    '[class*="Text"]',
    '[class*="detail"]',
    '[class*="Detail"]',
    '[class*="post-body"]',
    '[class*="post-content"]',
    '[class*="article-body"]',
    '[class*="article-content"]',
    '[class*="entry-content"]',
    '[class*="story-body"]',
    '[class*="content-body"]',
    '[class*="page-content"]',
    '[class*="rich-text"]',
    '[class*="prose"]',
    '[class*="markdown-body"]',
    '[class*="wysiwyg"]',
    # id-based
    '[id*="content"]',
    '[id*="Content"]',
    '[id*="article"]',
    '[id*="Article"]',
    '[id*="post"]',
    '[id*="Post"]',
    '[id*="entry"]',
    '[id*="Entry"]',
    '[id*="body"]',
    '[id*="Body"]',
    '[id*="main"]',
    '[id*="Main"]',
    '[id*="story"]',
    '[id*="Story"]',
    # itemprop / schema.org microdata
    '[itemprop="articleBody"]',
    '[itemprop="text"]',
    '[itemprop="mainEntity"]',
    # ARIA / role-based
    '[role="article"]',
    '[role="main"]',
    '[role="document"]',
    # data-* attributes (React/Vue/JS-driven sites)
    '[data-testid*="article-body"]',
    '[data-testid*="content"]',
    '[data-component*="article"]',
    '[data-role*="content"]',
    '[data-module*="article"]',
    # semantic tags
    "article",
    "main",
    "article section",
]

TEXT_BLOCK_TAGS = [
    # core prose
    "p",
    "blockquote",
    "pre",
    # list content (how-to guides, ranked lists, explainers)
    "li",
    "dd",
    "dt",
    # headings (structure long-form articles into sections)
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    # tabular / data content
    "td",
    "th",
    "caption",
    # media captions and figure text
    "figcaption",
    # inline-but-substantial text containers
    "summary",
    "details",
    # definition/description blocks
    "dl",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "title_idx": {"type": "integer"},
        "content_idx": {"type": "integer"},
    },
    "required": ["title_idx", "content_idx"],
}

MAX_CANDIDATES = 12
MAX_CANDIDATE_PREVIEW = 160
MIN_CONTENT_LEN = 200
MAX_RETRIES = 1
MIN_BLOCKS = 3
PARENT_DECAY_TOLERANCE = (
    0.85  # lower for more aggressive parent climbing, higher for more conservative
)

SYSTEM_PROMPT = (
    "You choose the correct webpage title and main content container from "
    "numbered candidate lists extracted from the page's DOM. Pick the TITLE "
    "index matching the article headline (not site name/nav/ads). Pick the "
    "CONTENT index that is the container wrapping the full article body "
    "(not a single paragraph, not sidebar/comment/ad). Use -1 if none fits. "
    "Respond with indices only."
)

# ── Config ─────────────────────────────────────────────────────────────────────

GENERAL_CONFIG = GeneralLLMConfig().from_dict({
    "model_path": "./models/Ministral-3-3B-Instruct-2512-Q5_K_M.gguf",
    "verbose": True
})

GENERAL_CONFIG.prompt_token_update(system_prompt=SYSTEM_PROMPT, user_prompt="{title_cands}\n{content_cands}")

LLM_CONFIG = LLMConfig().from_dict(
    {
        "n_ctx": GENERAL_CONFIG.calc_max_n_ctx(),  # MAX TOTAL context window for Mistral Q4_K_M per text.
        "verbose": False,  # llama_cpp verbose logging
    }
)

GEN_CONFIG = GENConfig().from_dict(
    {  # any
        "max_tokens": 50,  # small int json
        "temperature": 0.0,  # only int output
        "top_p": 1.0,  # irrelevant for deterministic output
        "repeat_penalty": 1.0,  # irrelevant for deterministic output
        # "stop":           ["</s>", "[/INST]", "###"],
    }
)


# --------------------------------------------------------------------------- #
# Fetch + clean
# --------------------------------------------------------------------------- #


def clean_html(raw_html: str) -> BeautifulSoup:
    soup = BeautifulSoup(raw_html, "lxml")

    for tag_name in JUNK_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    allowed_attrs = {"id", "class", "itemprop", "role"}
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr not in allowed_attrs:
                del tag.attrs[attr]

    for tag in soup.find_all(True):
        if not tag.get_text(strip=True) and not tag.find_all(True):
            tag.decompose()

    return soup


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _text_density_score(tag: Tag) -> float:
    text_len = len(tag.get_text(" ", strip=True))
    if text_len < 200:
        return 0.0
    blocks = tag.find_all(TEXT_BLOCK_TAGS)
    if not blocks:
        return 0.0
    para_like = [b for b in blocks if b.name in ("p", "li", "blockquote")]
    n_para = len(para_like) if para_like else len(blocks)
    block_text_len = sum(len(b.get_text(" ", strip=True)) for b in blocks)
    avg_block_len = block_text_len / len(blocks)
    coverage = min(block_text_len / text_len, 1.0)
    link_text_len = sum(len(a.get_text(" ", strip=True)) for a in tag.find_all("a"))
    link_penalty = 1 - min(link_text_len / text_len, 0.9)
    return n_para * min(avg_block_len, 200) / 100.0 * coverage * link_penalty


def _snap_to_parent(tag: Tag, decay_tolerance: float = PARENT_DECAY_TOLERANCE) -> Tag:
    """Climb ancestors while the parent absorbs more real content without diluting density."""
    current = tag
    current_score = _text_density_score(current)
    current_blocks = len(current.find_all(TEXT_BLOCK_TAGS))

    while current.parent and isinstance(current.parent, Tag):
        parent = current.parent
        p_score = _text_density_score(parent)
        p_blocks = len(parent.find_all(TEXT_BLOCK_TAGS))

        if p_blocks <= current_blocks or p_score < current_score * decay_tolerance:
            break
        current, current_score, current_blocks = parent, p_score, p_blocks

    return current


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    kind: str
    tag: Tag
    score: float
    preview: str

def build_title_candidates(soup: BeautifulSoup) -> list:
    cands, seen_text = [], set()

    def add(tag, score):
        if tag is None:
            return
        text = tag.get_text(" ", strip=True)
        if not text or text in seen_text or len(text) > 300:
            return
        seen_text.add(text)
        cands.append(Candidate("title", tag, score, text[:MAX_CANDIDATE_PREVIEW]))

    for h1 in soup.find_all("h1"):
        add(h1, 10)
    for sel in TITLE_SELECTORS:
        for tag in soup.select(sel)[:3]:
            if tag.name not in ("html", "body"):
                add(tag, 6)
    if soup.title:
        add(soup.title, 3)
    for h2 in soup.find_all("h2")[:2]:
        add(h2, 2)

    cands.sort(key=lambda c: -c.score)
    return cands[:MAX_CANDIDATES]


def build_content_candidates(soup: BeautifulSoup) -> list:
    raw_hits = set()

    for tag in soup.find_all(["article", "main"]):
        raw_hits.add(tag)
    for sel in CONTENT_SELECTORS:
        for tag in soup.select(sel):
            raw_hits.add(tag)
    for tag in soup.find_all(["div", "section"]):
        if _text_density_score(tag) > 0:
            raw_hits.add(tag)

    # snap every hit to its true parent container, dedupe by identity
    snapped = {}
    for tag in raw_hits:
        parent = _snap_to_parent(tag)
        snapped[id(parent)] = parent

    cands = []
    for tag in snapped.values():
        score = _text_density_score(tag)
        blocks = len(tag.find_all(TEXT_BLOCK_TAGS))
        if score <= 0 or blocks < MIN_BLOCKS:
            continue
        text = tag.get_text(" ", strip=True)
        cands.append(Candidate("content", tag, score, text[:MAX_CANDIDATE_PREVIEW]))

    cands.sort(key=lambda c: -c.score)

    # drop candidates nested inside an already-kept (larger/better) candidate
    filtered = []
    for c in cands:
        if any(c.tag in list(kept.tag.parents) for kept in filtered):
            continue
        filtered.append(c)

    return filtered[:MAX_CANDIDATES]


def extract_with_trafilatura(raw_html: str, content_tag) -> str:
    focused_html = str(content_tag) if content_tag is not None else raw_html
    extracted = trafilatura.extract(
        focused_html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        favor_recall=True,
    )
    if not extracted or len(extracted) < 40:
        extracted = trafilatura.extract(
            raw_html, include_comments=False, include_tables=False
        )
    return extracted or ""


def _resolve_tag(idx, cands):
    if 0 <= idx < len(cands):
        return cands[idx].tag
    return cands[0].tag if cands else None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

class llmFeedParser(LLM_Base):
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

    def process_single(
        self, title_cands, content_cands, exclude_content_idx=None
    ) -> dict:
        """Process a single article."""
        if self._llm is None:
            self.load()
            self._llm = self.router._llm
        if not title_cands and not content_cands:
            return {"title_idx": -1, "content_idx": -1}

        sys_prompt, prompt = self._build_prompt(title_cands, content_cands)
        if exclude_content_idx is not None:
            sys_prompt += (
                f" Candidate [{exclude_content_idx}] for CONTENT already failed "
                "extraction; pick a different content_idx."
            )

        messages = [{"role": "user", "content": prompt}]
        if self._use_system_role:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
        if self.schema != {}:
            grammar = LlamaGrammar.from_json_schema(json.dumps(self.schema))
        #print(f"Prompt: {prompt}")

        t0 = time.time()
        response = self._llm.create_chat_completion(  # type: ignore
            messages=messages,  # type: ignore
            grammar=grammar,
            **self.gen_config.to_dict(),
        )  # type: ignore
        elapsed = time.time() - t0

        raw = response["choices"][0]["message"]["content"].strip()  # type: ignore
        tok_out = response["usage"]["completion_tokens"]  # type: ignore

        if self.verbose:
            print(f"  [{elapsed:.1f}s | {tok_out} tok] ")

        print(f"  [RAW OUTPUT]\n {raw}\n")
        try:
            parsed = self._parse_llm_json(raw)
            t_idx = int(parsed.get("title_idx", -1)) # type: ignore
            c_idx = int(parsed.get("content_idx", -1)) # type: ignore
        except Exception:
            t_idx = 0 if title_cands else -1
            c_idx = 0 if content_cands else -1

        return {"title_idx": t_idx, "content_idx": c_idx}

    # --------------------------------------------------------------------------- #
    # Extraction
    # --------------------------------------------------------------------------- #

    def _build_prompt(self, title_cands, content_cands):
        lines = ["TITLE CANDIDATES:"]
        lines += [
            f"[{i}] <{c.tag.name}> {c.preview}" for i, c in enumerate(title_cands)
        ]
        lines.append("\nCONTENT CANDIDATES:")
        lines += [
            f"[{i}] <{c.tag.name}> {c.preview}" for i, c in enumerate(content_cands)
        ]
        return SYSTEM_PROMPT, "\n".join(lines)

    def parse_article(self, raw_html: str) -> dict:
        soup = clean_html(raw_html)

        title_cands = build_title_candidates(soup)
        content_cands = build_content_candidates(soup)

        choice = self.process_single(title_cands, content_cands)

        title_tag = _resolve_tag(choice["title_idx"], title_cands)
        content_tag = _resolve_tag(choice["content_idx"], content_cands)
        content_text = extract_with_trafilatura(raw_html, content_tag)

        tried_idx, retries = choice["content_idx"], 0
        while (
            len(content_text) < MIN_CONTENT_LEN
            and retries < MAX_RETRIES
            and len(content_cands) > 1
        ):
            choice = self.process_single(
                title_cands, content_cands, exclude_content_idx=tried_idx
            )
            tried_idx = choice["content_idx"]
            content_tag = _resolve_tag(choice["content_idx"], content_cands)
            content_text = extract_with_trafilatura(raw_html, content_tag)
            retries += 1

        title_text = (
            title_tag.get_text(" ", strip=True) if title_tag is not None else None
        )
        return {
            "title": title_text,
            "content": content_text,
        }
