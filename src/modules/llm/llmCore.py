"""
llm_base.py

Core article-processing LLM layer built on top of `llm_router.LLMRouter`,
which owns the actual local-vs-API backend dispatch (see llm_router.py).
`LLM_Base` is responsible for:

- Prompt construction hooks (subclasses implement `_build_prompt` /
  `_apply_to_article`)
- Chat orchestration, concurrency locking, and token-usage accounting
- Single/batch article processing

This module has NO knowledge-graph dependency. For KG-aware retrieval and
chat (`query_kg`, `query_entity`, `chat_with_kg`), see `llm_infer.py`,
which subclasses `LLM_Base` and adds those components on top.
"""

from abc import abstractmethod
import ast
import json
import re
import threading
import time
import traceback
from typing import Optional

from src.core.types import Article, GENConfig, GeneralLLMConfig, LLMConfig
from src.modules.llm.llmRouter import APIConfig, LLMRouter, logger


class LLM_Core:
    def __init__(
        self,
        llm_config: LLMConfig = LLMConfig(),
        gen_config: GENConfig = GENConfig(),
        general_config: GeneralLLMConfig = GeneralLLMConfig(),
        use_system_role: bool = True,
        schema: dict = {},
        api_config: Optional[APIConfig] = None,
        verbose: bool = True,
    ):
        self._infer_lock = (
            threading.Lock()
        )  # prevent concurrent calls to self.chat() on the same instance

        self.llm_config = llm_config
        self.gen_config = gen_config
        self.general_config = general_config

        # Backend dispatch (local llama.cpp vs hosted API providers) lives in
        # LLMRouter; api_config.use_api is the single switch between the two.
        self.router = LLMRouter(
            llm_config=llm_config,
            gen_config=gen_config,
            general_config=general_config,
            api_config=api_config,
            verbose=verbose,
            owner_name=self.__class__.__name__,
        )
        self.api_config = self.router.api_config
        self.use_api = self.router.use_api
        self.verbose = self.router.verbose
        # Resolved even in API mode: needed if fallback_to_local kicks in.
        self.model_path = self.router.model_path

        self._use_system_role = use_system_role  # whether to use system role in prompts
        self.schema = schema
        self._token_usage = {
            "t_prompt_tokens": 0,
            "t_completion_tokens": 0,
            "t_total_tokens": 0,
            "m_prompt_tokens": 0,
            "m_completion_tokens": 0,
            "m_total_tokens": 0,
            "entries": 0,
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> "LLM_Core":
        """Load the model into VRAM. Called automatically by the context manager."""
        self.router.load()
        return self

    def unload(self):
        self.analyze_token_usage()
        self.router.unload()

    def __enter__(self) -> "LLM_Core":
        return self.load()

    def __exit__(self, *_):
        self.unload()

    # ── Main methods ──────────────────────────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        sys_prompt: str = "",
        rag_context: str = "",
        gen_overrides: dict | None = None,
        schema_override: dict | None = None,
    ) -> str:
        raw = ""
        if rag_context and len(rag_context.strip()) > 0:
            if self._use_system_role:
                sys_prompt = f"{sys_prompt}\n\nKNOWLEDGE CONTEXT:\n{rag_context}"
            else:
                prompt = f"{prompt}\n\nKNOWLEDGE CONTEXT:\n{rag_context}"

        messages = [{"role": "user", "content": prompt}]
        if self._use_system_role:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
        if self.verbose:
            logger.info(f"###### Prompt: \n\t{messages}\n")
        active_schema = schema_override if schema_override is not None else self.schema
        if self.verbose:
            logger.info(f"###### Active Schema: \n\t{active_schema}\n")

        t0 = time.time()  # begin inference
        try:
            with self._infer_lock:
                raw, usage, used_provider, used_model = self.router.chat(
                    messages, active_schema, gen_overrides
                )
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Generation failed: {e}")
            traceback.print_exc()
            return ""

        elapsed = time.time() - t0  # end inference

        tok_out = usage.get("completion_tokens", 0)

        if self.verbose:
            logger.info(
                f" [{elapsed:.1f}s | {tok_out} tok | {used_provider}/{used_model}] "
            )
            logger.info(f"PROMPT TOK: {usage.get('prompt_tokens', 0)}")
            logger.info(f"TOTAL TOK: {usage.get('total_tokens', 0)}")
        self._token_usage["t_prompt_tokens"] += usage.get("prompt_tokens", 0)
        self._token_usage["t_completion_tokens"] += usage.get("completion_tokens", 0)
        self._token_usage["t_total_tokens"] += usage.get("total_tokens", 0)
        self._token_usage["m_prompt_tokens"] = max(
            self._token_usage["m_prompt_tokens"], usage.get("prompt_tokens", 0)
        )
        self._token_usage["m_completion_tokens"] = max(
            self._token_usage["m_completion_tokens"],
            usage.get("completion_tokens", 0),
        )
        self._token_usage["m_total_tokens"] = max(
            self._token_usage["m_total_tokens"], usage.get("total_tokens", 0)
        )
        self._token_usage["entries"] += 1

        logger.info(f" [RAW OUTPUT]\n {raw}\n")
        return raw

    def analyze_token_usage(self):
        if not self.verbose:
            return
        logger.info("=" * 50)
        logger.info(f"[{self.__class__.__name__}] Token usage analysis:")
        logger.info(f"Total prompt tokens: {self._token_usage['t_prompt_tokens']}")
        logger.info(f"Total completion tokens: {self._token_usage['t_completion_tokens']}")
        logger.info(f"Total tokens: {self._token_usage['t_total_tokens']}")
        logger.info(
            f"Max prompt tokens in a single request: {self._token_usage['m_prompt_tokens']}"
        )
        logger.info(
            f"Max completion tokens in a single request: {self._token_usage['m_completion_tokens']}"
        )
        logger.info(
            f"Max total tokens in a single request: {self._token_usage['m_total_tokens']}"
        )
        logger.info(f"Total entries processed: {self._token_usage['entries']}")
        logger.info("=" * 50)
        logger.info(
            f"Avg. prompt tokens per entry: {self._token_usage['t_prompt_tokens'] / self._token_usage['entries'] if self._token_usage['entries'] > 0 else 0:.2f}"
        )
        logger.info(
            f"Avg. completion tokens per entry: {self._token_usage['t_completion_tokens'] / self._token_usage['entries'] if self._token_usage['entries'] > 0 else 0:.2f}"
        )
        logger.info(
            f"Avg. total tokens per entry: {self._token_usage['t_total_tokens'] / self._token_usage['entries'] if self._token_usage['entries'] > 0 else 0:.2f}"
        )
        logger.info("=" * 50)
    
    
    # ── helpers ──────────────────────────────────────────────────────────────

    def _truncate_to_budget(self, text: str) -> str:
        """Truncate text so prompt + reserved system/output tokens fit n_ctx."""
        if not text:
            return ""
        budget_chars = max(5000, self.general_config.max_input_text_chars)
        if len(text) > budget_chars:
            logger.info(
                f"[WARN] Truncating content to {budget_chars} chars (got {len(text)} chars) to fit context window"
            )
            return text[:budget_chars] + "\n...[truncated]"
        return text

    @abstractmethod
    def _build_prompt(self, article: Article) -> tuple[str, str]:
        """Build a prompt for a single article."""
        raise NotImplementedError("Subclasses must implement _build_prompt()")

    def _parse_llm_json(self, raw: str) -> tuple[dict, str | None]:
        think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
        think = think_match.group(1).strip() if think_match else ""
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        text_fixed = cleaned.replace("\n", " ").replace("\r", "")
        text = text_fixed.strip()
        try:
            return json.loads(text), think
        except json.JSONDecodeError:
            traceback.print_exc()
            try:
                return ast.literal_eval(text), think  # fallback to ast.literal_eval for non-standard JSON
            except Exception:
                traceback.print_exc()
                return {}, None

    @staticmethod
    @abstractmethod
    def _apply_to_article(article: Article, parsed: dict) -> Article:
        """Apply parsed LLM output to an Article object."""
        raise NotImplementedError("Subclasses must implement _apply_to_article()")
