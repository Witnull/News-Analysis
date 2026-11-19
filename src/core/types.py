"""Shared types for the news intelligence pipeline."""

from dataclasses import dataclass, field, asdict
from math import ceil
from typing import List, Literal
import json
import llama_cpp


# ---------------------------------------------------------------------------
# Web data classes
# ---------------------------------------------------------------------------
@dataclass
class Feed:
    url: str  # fetch url
    publisher: str  # publisher name
    category: str  # primary category - the web main topic /General if unknown
    type: Literal["rss", "atom", "html", "json", "api", "other", "video", ""] = (
        "other"  # feed type - rss, atom, json, html
    )
    rss_type: Literal["google_news", "custom", "sitemap"] = (
        "google_news"  # rss type - google_news, custom, sitemap
    )
    require_custom_parser: bool = False  # whether to use a custom parser for this feed
    paid: bool = False  # whether the feed is paid or free
    lang: str = "en"  # language code, e.g. "en", "zh", "es", etc.
    country: str = ""  # country code, e.g. "US", "CN", "ES", etc.
    tags: List[str] = field(default_factory=list)  # additional tags for the feed
    bozo_ignore: bool = False  # whether to ignore bozo exceptions when parsing the feed
    # manual verify
    checked: bool = False  # whether the feed has been manually checked for quality
    working: bool = False  # whether the feed is working or broken
    custom_parser_id: str = ""  # ID of the custom parser for this feed, if applicable
    may_block_access: bool = (
        False  # whether the feed may block access due to rate limiting or other reasons
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def json(self, **kwargs) -> str:
        """Serialize this article to a JSON string. Pass indent=2 etc. via kwargs."""
        return json.dumps(asdict(self), ensure_ascii=False, **kwargs)


# ---------------------------------------------------------------------------
# Article data classes
# ---------------------------------------------------------------------------


@dataclass
class Article:
    # Basic fields
    url: str
    title: str
    original_text: str
    tags: List[str] = field(default_factory=list)  # category
    pub_date: str = ""
    #desc: str = ""
    cat1: str = ""  # primary category - the web main topic - manual set
    publisher: str = ""  # source - the web site name - manual set
    # additional fields
    lang: str = "en"  # language code, e.g. "en", "zh", "es", etc. - auto/manual set
    country: str = "US"  # country code, e.g. "US", "CN", "ES", etc.  - auto/manual set

    # GENERATED FIELDS ######################################
    summarized_text: str = ""
    #llm_tags: List[str] = field(default_factory=list)  # llm generated tags
    keywords: List[str] = field(default_factory=list)
    #concepts: List[str] = field(default_factory=list)
    one_line_summary: str = ""  # one line summary
    #wii: str = ""  # why it is important

    yield_score: int = 0  # secondary category - "important" score
    yield_reason: dict = field(
        default_factory=dict
    )  # secondary category - "important" reason
    half_life_est: Literal["short", "medium", "long", "none"] = (
        "none"  # estimated half-life of the article
    )
    half_life_est_reason: dict = field(
        default_factory=dict
    )  # reason for half-life estimation

    is_reviewed: bool = False  # whether the article has been reviewed
    processed_stage: str = "none"  # processing stage - 0 = not processed, 1 = filtered, 2 = summarized, 3 = concept extracted, 4 = worthy eval
    entity: str = ""  # entity extracted from the article
    ticker: str = ""  # ticker extracted from the article
    log: str = "none"  # log of processing stages and errors
    no_save_pool: dict = field(
        default_factory=dict
    )  # no save pool - for temporary data that should not be saved to the database

    def json(self, **kwargs) -> str:
        """Serialize this article to a JSON string. Pass indent=2 etc. via kwargs."""
        return json.dumps(asdict(self), ensure_ascii=False, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Article":
        """Build an Article from a dict, ignoring any unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})



@dataclass
class LLMConfig:
    n_gpu_layers: int = -1  # full GPU
    n_ctx: int = 8000
    n_batch: int = 1024
    n_threads: int = 8
    verbose: bool = False
    flash_attn: bool = True  # Enforce native Flash Attention
    seed: int = 42  # default seed llama_cpp.LLAMA_DEFAULT_SEED, -1 for random
    offload_kqv: bool = True
    logits_all: bool = False
    type_k: int = llama_cpp.GGML_TYPE_Q8_0
    type_v: int = llama_cpp.GGML_TYPE_Q8_0
    n_ubatch: int = 512  # physical batch size, pairs with n_batch=1024
    use_mlock: bool = True  # prevents model from being swapped out of RAM

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        """Build an LLMConfig from a dict, ignoring any unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class GENConfig:
    max_tokens: int = 8000
    temperature: float = 0.2
    top_k: int = 50
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    stop: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GENConfig":
        """Build a GENConfig from a dict, ignoring any unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})


@dataclass
class GeneralLLMConfig:
    model_path: str = "./models/Ministral-3-8B-Instruct-2512-Q5_K_M.gguf"#Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    n_ctx: int = 0
    max_input_text_chars: int = 20_000
    chars_per_token_ratio: float = 3.5
    special_chars_token_headroom: int = 40
    headroom_tokens: int = 200
    system_prompt_chars: int = 2000
    user_prompt_chars: int = 2000
    verbose: bool = False

    def prompt_token_update(self, system_prompt: str = "", user_prompt: str = ""):
        if system_prompt != "":
            self.system_prompt_chars = len(system_prompt)
        if user_prompt != "":
            self.user_prompt_chars = len(user_prompt)

    def ceil_k(self, value: float, k: int = 5000) -> int:
        """Ceil a value to the nearest multiple of k."""
        return int(ceil(value / k) * k)

    def calc_max_n_ctx(self) -> int:
        """Calculate the maximum context window size for the LLM based on the max_input_text_chars and chars_per_token_ratio."""
        return (
            self.n_ctx
            if self.n_ctx != 0
            else self.ceil_k(
                ceil(self.max_input_text_chars / self.chars_per_token_ratio)
                + ceil(
                    (self.system_prompt_chars + self.user_prompt_chars)
                    / self.chars_per_token_ratio
                )
                + self.headroom_tokens
                + self.special_chars_token_headroom
            )
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "GeneralLLMConfig":
        """Build a GeneralLLMConfig from a dict, ignoring any unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})
