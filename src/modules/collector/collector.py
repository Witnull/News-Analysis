import asyncio
import logging
import os

import html
import random
from src.utils.logger import setup_logger
import re
import traceback
from urllib.parse import urljoin, urlparse
import json

import feedparser
import trafilatura
from tqdm.asyncio import tqdm as atqdm

from scrapling.fetchers import (
    AsyncFetcher,  # replaces AsyncFetcherSession — static async HTTP methods
    StealthyFetcher,  # sync .fetch() / async .async_fetch()
)
from src.modules.store.store import ArticleStore
from src.core.types import Article, Feed
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from src.modules.collector.llmFeedParser import llmFeedParser
from src.modules.llm.llmPool import LLMPool

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.environ.get("LOG_DIR", f"Logs/Cltr_{timestamp}")
os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
logger = setup_logger(
    f"{log_dir}/feedCollector.log", log_level=logging.DEBUG
)  # Adjust log level as needed

# ====================================================================
# region Settings
# ====================================================================
MIN_TEXT_LEN = 100

FLARESOLVERR_URL = "http://localhost:8191/v1"
FLARESOLVERR_TIMEOUT_MS = 60000  # 60s for Cloudflare JS challenges

# ====================================================================
# Url redirect
_META_REFRESH_RE = re.compile(
    r'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+content=["\'][^;]+;\s*url=([^"\'> ]+)',
    re.IGNORECASE,
)
_JS_REDIRECT_RE = re.compile(
    r'(?:window\.location(?:\.href)?\s*=\s*|window\.location\.replace\()\s*["\'](https?://[^"\']+|/[^"\']*)["\']',
    re.IGNORECASE,
)
_CANONICAL_LINK_RE = re.compile(
    r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_SINGLE_ANCHOR_RE = re.compile(
    r'<a[^>]+href=["\'](https?://[^"\']+)["\']', re.IGNORECASE
)
# Interstitial content
_INTERSTITIAL_MAX_LEN = 2000
# ====================================================================
# XML recover
_XML_DECL_RE = re.compile(r"<\?xml[^>]*\?>")
_XML_VIEWER_PRE_RE = re.compile(r"<pre[^>]*>(.*)", re.DOTALL | re.IGNORECASE)

# ====================================================================
# Valid URL schemes
_VALID_SCHEMES = ("http://", "https://")

# ====================================================================
STEALTH_HARD_TIMEOUT_S = 90  # must exceed timeout*retries budget with margin
BASIC_TIMEOUT_S = 60  # for plain fetches

GG_NEWS_DECODE_POLLING_S = 2  # poll for Google News decode results every N seconds

GENERAL_CD_PER_REQUESTS_S = (
    2  # general cooldown between requests to avoid rate-limiting / 429s
)

MAX_CONCURRENT_ENTRIES = (
    5  # hard cap on concurrent entry processing + headroom for stealth timeout block
)
MAX_CONCURRENT_FETCH = 8  # hard cap on concurrent fetches (plain)
MAX_CONCURRENT_STEALTH = 3  # hard cap on real Chrome instances
MAX_GG_NEWS_DECODE_CONCURRENT = 2  # hard cap on concurrent Google News decode requests
# ====================================================================
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/",
}
BATCHEXECUTE_URL = "" 
# ====================================================================
COMMON_ERR_PHRASES = [
    "Please enable JavaScript to continue",
    "Please enable cookies to continue",
    "Access Denied",
    "You are being rate limited",
    "You have been temporarily blocked",
    "Your request was blocked",
    "Cloudflare Ray ID",
    "Catch up on global daily news",  # Economist
]
# ====================================================================
# Filter static
_HTML_TAG_RE = re.compile(
    r"<(?:html|head|body|script|style|div|span|a|img|svg|nav|footer|header|form|table|iframe|meta|link)\b",
    re.I,
)

_XML_RE = re.compile(
    r"<\?xml|<(?:rss|feed|channel|entry)\b",
    re.I,
)


def is_useful_text(text: str | None) -> tuple[bool, str]:
    if not text:
        return False, "empty"
    text = text.strip()
    if len(text) < 100:
        return False, "too short"
    lower = text.lower()
    # Entire HTML/XML documents
    if lower.startswith("<!doctype html"):
        return False, "html document"
    if lower.startswith("<html"):
        return False, "html document"
    if _XML_RE.search(text):
        return False, "xml/rss"
    # Lots of HTML tags
    tags = len(_HTML_TAG_RE.findall(text))
    if tags >= 3:
        return False, f"contains {tags} html tags"
    # Looks like mostly HTML
    if text.count("<") > 15 and text.count(">") > 15:
        return False, "too many angle brackets"
    # JS placeholders
    js_markers = (
        "__next",
        "__nuxt",
        "window.__",
        "document.getelementbyid",
        "javascript",
        "function(",
        "onclick=",
    )
    if any(x in lower for x in js_markers):
        return False, "javascript"
    # Cookie banners
    cookie_markers = (
        "accept cookies",
        "cookie policy",
        "privacy policy",
        "manage cookies",
        "consent",
    )

    if sum(m in lower for m in cookie_markers) >= 2:
        return False, "cookie page"
    return True, ""


# Google News decode helpers
async def decode_url(base64_id, signature, timestamp):
    """Decode a single article id to its real URL."""
    pass 


def extract_base64_id(google_news_link):
    parsed = urlparse(google_news_link)
    path_parts = parsed.path.split("/")
    if (
        parsed.hostname == "news.google.com"
        and len(path_parts) > 1
        and path_parts[-2] in ("articles", "read")
    ):
        return path_parts[-1]
    return None


async def get_signing_params(base64_id):
    url = f"https://news.google.com/rss/articles/{base64_id}"
    resp = await AsyncFetcher.get(
        url, timeout=BASIC_TIMEOUT_S * 1000, stealthy_headers=True
    )
    div = resp.css("c-wiz > div").first
    if div is None:
        logger.info("[RED] Could not locate signing attributes on article page")
    signature = div.attrib.get("data-n-a-sg")
    timestamp = div.attrib.get("data-n-a-ts")
    if not signature or not timestamp:
        logger.info("[RED] Missing data-n-a-sg / data-n-a-ts attributes")
    return signature, timestamp


def _extract_body(resp):
    for attr in ("text", "body", "content"):
        val = getattr(resp, attr, None)
        if val is None:
            continue
        if callable(val):
            try:
                val = val()
            except TypeError:
                continue
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        if isinstance(val, str) and val.strip():
            return val
    return ""


# ==================================================================
#  Helpers recovering xml format
def _unwrap_browser_xml_viewer(raw: str) -> Optional[str]:
    """When a fetch falls back to a real browser (StealthyFetcher),
    navigating straight to an .xml/.rss URL makes Chromium render its
    own built-in XML-source viewer instead of handing back raw bytes:
    the actual feed ends up HTML-entity-escaped inside a
    <pre style="white-space: pre-wrap;">...</pre> block of a synthetic
    HTML page (recognizable by the injected
    <meta name="color-scheme" content="light dark"> tag). feedparser
    then sees a normal HTML page with no real <item>/<entry> tags and
    returns zero entries. Pull the real XML back out of that <pre>
    block and HTML-unescape it.
    """
    match = _XML_VIEWER_PRE_RE.search(raw)
    if not match:
        return None
    inner = html.unescape(match.group(1))
    inner = re.sub(
        r"</pre>\s*(</body>\s*</html>\s*)?$", "", inner, flags=re.IGNORECASE
    ).strip()
    if not (
        inner.startswith("<?xml")
        or inner.startswith("<rss")
        or inner.startswith("<feed")
    ):
        return None
    return inner


def _recover_feed_body(raw: str) -> Optional[str]:
    """Called only when the first feedparser.parse() pass comes back
    with zero entries. Tries, in order:
    1. Unwrapping a browser's native XML-viewer HTML page (the
        confirmed, common cause — see _unwrap_browser_xml_viewer).
    2. A generic fallback for servers that prepend junk (a WordPress
        PHP notice/warning, a proxy/CDN error wrapper, or a duplicate
        leading '<?xml ...?>' document) before the real feed body:
        walk every '<?xml ...?>' occurrence (plus a bare <rss>/<feed>
        root) from the end backwards and use the first slice that
        actually parses with entries.
    """
    unwrapped = _unwrap_browser_xml_viewer(raw)
    if unwrapped is not None and feedparser.parse(unwrapped).entries:
        logger.info(
            "[GREEN] Recovered feed body from browser's native XML-viewer wrapper for a feed"
        )
        return unwrapped

    starts = [m.start() for m in _XML_DECL_RE.finditer(raw)]
    root_match = re.search(r"<(rss|feed)[\s>]", raw, re.IGNORECASE)
    if root_match:
        starts.append(root_match.start())
    starts = sorted(set(s for s in starts if s > 0))
    for start in reversed(starts):
        candidate = raw[start:]
        if feedparser.parse(candidate).entries:
            logger.info(
                f"[GREEN] Recovered feed body by trimming {start} bytes of leading junk / duplicate declaration for a feed"
            )
            return candidate
    return None


# ==================================================================
# region Collector
# ==================================================================
class FeedCollector:
    def __init__(
        self,
        feeds: List[Feed],
        max_age_days: int = 7,
        db: Optional[ArticleStore] = None,
        save_to_db: bool = True,
        force_refetch: bool = False,
        limit_per_feed: Optional[int] = None,
        proxy: Optional[str] = None,
        use_flaresolverr: bool = True,
        flaresolverr_url: str = FLARESOLVERR_URL,
        flaresolverr_timeout_ms: int = FLARESOLVERR_TIMEOUT_MS,
        max_concurrent_entries: int = MAX_CONCURRENT_ENTRIES,  #
        max_concurrent_stealth: int = MAX_CONCURRENT_STEALTH,  # hard cap on real Chrome instances
        max_gg_news_decode_concurrent: int = MAX_GG_NEWS_DECODE_CONCURRENT,  # hard cap on concurrent Google News decode requests
        max_concurrent_fetch: int = MAX_CONCURRENT_FETCH,  # hard cap on concurrent fetches (plain)
        strict_mode: bool = False,  # if True, fail on any feed/entry error; if False, log and continue
    ):
        self.feeds = feeds
        self.max_age_days = max_age_days
        self.force_refetch = force_refetch
        self.limit_per_feed = limit_per_feed
        self.imm_articles: List[Article] = []
        self.store = db if db is not None else ArticleStore("sqlite:///articles.db")
        self.save_to_db = save_to_db
        self.proxy = proxy
        self.use_flaresolverr = use_flaresolverr
        self.flaresolverr_url = flaresolverr_url
        self.flaresolverr_timeout_ms = flaresolverr_timeout_ms
        # Flipped to True after the first connection-level failure so we stop
        # wasting time retrying a solver that isn't reachable this run.
        self._flaresolverr_down = False
        # Semaphore set in run() so it lives inside the event loop
        self.max_concurrent_entries = max_concurrent_entries
        self.max_concurrent_stealth = max_concurrent_stealth
        self.max_gg_news_decode_concurrent = max_gg_news_decode_concurrent
        self.max_concurrent_fetch = max_concurrent_fetch
        self._entry_sem: Optional[asyncio.Semaphore] = None
        self._stealth_sem: Optional[asyncio.Semaphore] = None
        self._gg_news_decode_sem: Optional[asyncio.Semaphore] = None
        self._fetch_sem: Optional[asyncio.Semaphore] = None
        self.strict = strict_mode

        self.require_check = []  # list of feeds that require a manual check
        self.failed_feed_entries = []  # list of {feed, entry, url, err} dict for failed entries - will be retry once at the end
        self._llmFeedparser = LLMPool(
            llmFeedParser, n_workers=2, verbose=True
        ).load()  # lshould be 2 - leave headroom to prevent freeze

    # ------------------------------------------------------------------
    # Entry point — call this instead of anything in __init__
    # ------------------------------------------------------------------

    async def run(self):
        """Public entry point. Awaitable replacement for the old sync constructor side-effect."""
        # self._entry_sem = asyncio.Semaphore(MAX_CONCURRENT_ENTRIES)
        # self._entry_sem = asyncio.Semaphore(self.max_concurrent_entries)
        # self._stealth_sem = asyncio.Semaphore(self.max_concurrent_stealth)
        # self._gg_news_decode_sem = asyncio.Semaphore(self.max_gg_news_decode_concurrent)
        # self._fetch_sem = asyncio.Semaphore(self.max_concurrent_fetch)
        self._ensure_semaphores()
        await self._batch_fetch_and_process()

    def _ensure_semaphores(self) -> None:
        """Lazily create semaphores so any entry point (run(), fetch_article_from_url(),
        fetch_articles_from_urls(), etc.) works even if run() was never called."""
        if self._entry_sem is None:
            self._entry_sem = asyncio.Semaphore(self.max_concurrent_entries)
        if self._stealth_sem is None:
            self._stealth_sem = asyncio.Semaphore(self.max_concurrent_stealth)
        if self._gg_news_decode_sem is None:
            self._gg_news_decode_sem = asyncio.Semaphore(
                self.max_gg_news_decode_concurrent
            )
        if self._fetch_sem is None:
            self._fetch_sem = asyncio.Semaphore(self.max_concurrent_fetch)

    # ------------------------------------------------------------------
    # Pure helpers (sync — no I/O)
    # ------------------------------------------------------------------

    def _entry_timestamp(self, entry) -> int:
        for attr in ("published_parsed", "updated_parsed"):
            time_struct = getattr(entry, attr, None)
            if time_struct:
                dt = datetime(*time_struct[:6], tzinfo=timezone.utc)
                return int(dt.timestamp())
        return -1

    def _is_entry_within_age(self, entry) -> bool:
        if self.max_age_days is None:
            return True
        timestamp = self._entry_timestamp(entry)
        if timestamp == -1:
            return True
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self.max_age_days)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc) >= cutoff

    # ==================================================================
    # region 5. Static helpers
    # ==================================================================

    @staticmethod
    def _is_fetchable_url(url: Optional[str]) -> bool:
        return bool(url) and url.strip().lower().startswith(_VALID_SCHEMES)

    def _get_content_from_entry(self, entry) -> str | None:
        content_list = getattr(entry, "content", [])

        if not content_list:
            logger.info("[YELLOW] No content field in entry...")
            return None

        if len(content_list) == 1:
            item = content_list[0]
            if item.get("type", "") in ("text/html", "application/xhtml+xml"):
                logger.info(
                    "[GREEN] Single content item is HTML/XHTML, extracting text..."
                )
                return self._extract_or_clean(item["value"]) or item["value"]
            else:
                return item.get("value", "")
        else:
            logger.warning("Multiple content items found in entry, selecting best...")
            # Multiple content items — prefer HTML/XHTML, else pick the longest # mostly not gonna hit
            for item in content_list:
                if item.get("type", "") in ("text/html", "application/xhtml+xml"):
                    return self._extract_or_clean(item["value"]) or item["value"]

            best = max(content_list, key=lambda x: len(x.get("value", "")))
            return self._extract_or_clean(best.get("value", "")) or best.get(
                "value", ""
            )
        return None

    @staticmethod
    def _looks_like_interstitial(html: str) -> bool:
        if not html:
            return False
        stripped = html.strip()
        if len(stripped) < _INTERSTITIAL_MAX_LEN:
            return True
        if _META_REFRESH_RE.search(html) or _JS_REDIRECT_RE.search(html):
            return True
        return False

    @staticmethod
    def _unwrap_redirect(url: str, html: str) -> Optional[str]:
        for pattern in (_META_REFRESH_RE, _JS_REDIRECT_RE, _CANONICAL_LINK_RE):
            match = pattern.search(html)
            if match:
                candidate = match.group(1).strip()
                resolved = urljoin(url, candidate)
                if resolved != url and FeedCollector._is_fetchable_url(resolved):
                    return resolved
        anchors = {
            m.group(1)
            for m in _SINGLE_ANCHOR_RE.finditer(html)
            if FeedCollector._is_fetchable_url(m.group(1))
        }
        anchors.discard(url)
        if len(anchors) == 1:
            return anchors.pop()
        return None

    @staticmethod
    def _extract_url_from_a_tag(entry) -> Optional[str]:
        summary = getattr(entry, "summary", "")
        match = re.search(r'<a[^>]+href=["\'](https?://[^"\']+)["\']', summary)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _extract_url_from_entry(
        feed: Feed, entry, attempt: int = 0
    ) -> tuple[str | None, int]:
        curr_atmpt = attempt
        match attempt:
            case 0:
                url = getattr(entry, "link", None)
                if FeedCollector._is_fetchable_url(url):
                    return (url, curr_atmpt)
            case 1:
                links = getattr(entry, "links", [])
                for link in links:
                    if link.get("rel") == "alternate" and link.get("type") in (
                        "text/html",
                        "application/xhtml+xml",
                    ):
                        url = link.get("href")
                        if FeedCollector._is_fetchable_url(url):
                            return (url, curr_atmpt)
            case 2:
                url = FeedCollector._extract_url_from_a_tag(entry)
                if FeedCollector._is_fetchable_url(url):
                    return (url, curr_atmpt)
        logger.info("[RED] NO URL FOUND")
        return (None, curr_atmpt)

    @staticmethod
    def _extract_or_clean(text_or_html: str) -> str:
        if not text_or_html:
            return ""
        return (
            trafilatura.extract(
                text_or_html,
                favor_precision=True,
                favor_recall=True,
                include_comments=False,
                deduplicate=True,
            )
            or text_or_html.strip()
        )

    async def _extract_article(self, html_text: str) -> tuple[str, Optional[str]]:
        """Extract (content, title) from article HTML.

        Uses feed.custom_parser_id via feedParser when set; falls back to the
        default trafilatura-based extraction if there's no parser id or the
        custom parser errors out.
        """

        try:
            result = await self._llmFeedparser.call("parse_article", html_text)  # type: ignore

            content = (result.get("content") or "").strip()
            if content:
                return content, result.get("title")

            logger.info("[YELLOW] Parser returned empty content, falling back")
        except Exception:
            logger.exception("[RED] Parser failed, falling back to default extraction")
            traceback.print_exc()
        return self._extract_or_clean(html_text), None

    @staticmethod
    def _is_generated_shell_page(url: str, html: str) -> bool:
        lowered_head = html.strip()[:300].lower()
        if "news.google.com" in url and lowered_head.startswith("<!doctype html>"):
            return True
        # SPA markers specific to Google News' shell (safe to extend)
        if "window.wiz_global_data" in html or "boq-dots" in html:
            return True
        if "<html dir=" in html:
            return True
        return False

    # ------------------------------------------------------------------
    # Async fetch layer
    # ------------------------------------------------------------------

    def _fetch_tiers(self) -> list[str]:
        """Ordered list of fetch strategies to try, cheapest-bypass-first."""
        tiers = []
        if self.use_flaresolverr and not self._flaresolverr_down:
            tiers.append("flaresolverr")
        tiers.append("plain")
        tiers.append("stealth")
        return tiers

    async def _fetch_via_flaresolverr(self, url: str) -> tuple[str, str]:
        payload = {
            "cmd": "request.get",
            "url": url,
            "maxTimeout": self.flaresolverr_timeout_ms,
        }
        try:
            resp = await AsyncFetcher.post(
                self.flaresolverr_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=(self.flaresolverr_timeout_ms / 1000) + 10,
            )
        except Exception:
            # Couldn't even reach the solver — don't keep retrying it this run.
            self._flaresolverr_down = True
            logger.info(f"[RED] FlareSolverr unreachable for {url}")

        try:
            data = resp.json()
        except Exception:
            logger.info(f"[RED] Failed to parse FlareSolverr response for {url}")
            return "", ""

        if data.get("status") != "ok":
            logger.info(f"[RED] FlareSolverr error for {url}: {data.get('message')}")

        solution = data["solution"]
        html = solution.get("response", "")
        content_type = (solution.get("headers") or {}).get("content-type", "")
        return html, content_type

    async def _fetch_raw(self, url: str, tier: str = "plain") -> tuple[str, str]:
        self._ensure_semaphores()
        if not self._is_fetchable_url(url):
            return "", ""
        resp = ""
        if tier == "flaresolverr":
            return await self._fetch_via_flaresolverr(url)

        if tier == "stealth":
            async with self._stealth_sem:  # type: ignore
                try:
                    resp = await asyncio.wait_for(
                        StealthyFetcher.async_fetch(
                            url,
                            headless=True,
                            network_idle=True,
                            solve_cloudflare=True,
                            proxy=self.proxy,
                            timeout=BASIC_TIMEOUT_S * 1000,
                        ),
                        timeout=STEALTH_HARD_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        f"[RED] Stealth/Cloudflare solve exceeded hard timeout ({STEALTH_HARD_TIMEOUT_S}s) for [{url}]"
                    )
                    return resp, ""

                except Exception:
                    logger.info(f"[RED] Stealth fetch failed for [{url}]")
                    traceback.print_exc()
                    return resp, ""
        else:  # plain
            async with self._fetch_sem:  # type: ignore
                try:
                    resp = await AsyncFetcher.get(
                        url, stealthy_headers=True, proxy=self.proxy
                    )
                except Exception:
                    logger.info(f"[RED] Plain fetch failed for [{url}]")
                    traceback.print_exc()
                    return resp, ""

        if resp is None:
            logger.info(f"[RED] Fetch tier '{tier}' produced no response for [{url}]")
            return resp, ""

        html = resp.body.decode("utf-8", errors="ignore")
        content_type = resp.headers.get("Content-Type", "")
        return html, content_type

    async def _resolve_real_url(
        self, url: str, html: str, depth: int = 0
    ) -> tuple[str, str]:
        if depth >= 3 or not self._looks_like_interstitial(html):
            return url, html
        # if the page is not an interstitial, return as is
        # Check for meta-refresh, JS redirect, canonical link, or single anchor
        next_url = self._unwrap_redirect(url, html)
        if not next_url or next_url == url:
            return url, html

        next_html = None
        for tier in self._fetch_tiers():
            try:
                next_html, _ = await self._fetch_raw(next_url, tier=tier)
                if next_html:
                    break
                else:
                    continue
            except Exception:
                logger.info(
                    f"[RED] Redirect unwrap fetch tier '{tier}' failed for {url} -> {next_url}"
                )
                await asyncio.sleep(
                    random.uniform(
                        GENERAL_CD_PER_REQUESTS_S / 2, GENERAL_CD_PER_REQUESTS_S * 2
                    )
                )  # brief pause before next tier
                continue

        if not next_html:
            return url, html
        return await self._resolve_real_url(next_url, next_html, depth=depth + 1)

    async def _fetch_text(self, url: str) -> tuple[str, Optional[str], Optional[str]]:
        logger.info(f"[YELLOW] Fetching text from HTML: {url}")
        last_html = ""
        text, title, ts = "", None, None

        for tier in self._fetch_tiers():
            try:
                html, _ = await self._fetch_raw(url, tier=tier)
                if not html: 
                    continue
            except Exception:
                logger.info(f"[YELLOW] Fetch tier '{tier}' failed for {url}")
                continue

            resolved_url, resolved_html = await self._resolve_real_url(url, html)
            # NEW: detect known SPA-shell / non-article pages before extraction
            if self._is_generated_shell_page(resolved_url, resolved_html):
                logger.info(
                    f"[RED] Detected auto-generated shell page for {resolved_url}, skipping tier '{tier}'"
                )
                # with open(
                #     f"{log_dir}/shell_page_{md5(resolved_url.encode()).hexdigest()}.html",
                #     "w",
                #     encoding="utf-8",
                # ) as f:
                #     f.write(resolved_html)
                continue

            last_html = resolved_html
            text, title = await self._extract_article(resolved_html)
            meta = trafilatura.extract_metadata(last_html)
            if meta:
                ts = meta.date

            if len(text) >= MIN_TEXT_LEN:
                logger.info(
                    f"[GREEN] [HIT] Fetch tier '{tier}' succeeded for {url[:20]} with sufficient text: {text[:100]}..."
                )
                return text, title, str(ts)

            logger.info(
                f"[RED] Fetch tier '{tier}' yielded insufficient text: {text[:100]}...",
            )
        if last_html:
            text, title = await self._extract_article(last_html)
            meta = trafilatura.extract_metadata(last_html)
            if meta:
                ts = meta.date

        return text, title, str(ts)

    async def decode_gnews_url(self, entry: dict):
        url = entry["link"]
        base64_id = extract_base64_id(url)
        if base64_id is None:
            logger.warning(
                f"[YELLOW] Skipping URL with no base64_id in Google News link: {url}"
            )
            return None
        # PASS 1: Get signing params
        async with self._gg_news_decode_sem:  # type: ignore
            try:
                signature, timestamp = await get_signing_params(base64_id)
            except Exception:
                logger.info(
                    f"[RED] Error occurred while fetching signing params for URL: {url}"
                )
                traceback.print_exc()
                return None

        await asyncio.sleep(
            random.uniform(
                GG_NEWS_DECODE_POLLING_S / 2,
                GG_NEWS_DECODE_POLLING_S * 2,
            )
        )  # brief pause before next request

        # PASS 2: Decode the URL
        async with self._gg_news_decode_sem:  # type: ignore
            try:
                real_url = await decode_url(base64_id, signature, timestamp)
                if not real_url:
                    logger.warning(
                        f"[YELLOW] Skipping entry with no base64_id in Google News link: {entry['link']}"
                    )
                    return None
            except Exception:
                logger.info(f"[RED] Error occurred while decoding URL: {entry['link']}")
                traceback.print_exc()
                return None

        entry["link"] = real_url
        logger.info(f"[GREEN][HIT] Decoded link: {entry['link']}")
        await asyncio.sleep(
            random.uniform(
                GG_NEWS_DECODE_POLLING_S / 2,
                GG_NEWS_DECODE_POLLING_S * 2,
            )
        )  # brief pause before next request

        # ------------------------------------------------------------------

    # Article Building Helper
    # ------------------------------------------------------------------
    def _build_article(
        self,
        url: str,
        raw_html: str,
        extracted_text: str,
        title: str = "unknown title",
        pub_date: str = "",
        tags: list[str] | None = None,
        publisher: str = "",
        lang: str = "en",
        country: str = "US",
        cat1: str = "",
    ) -> Article:
        """Centralized Article factory to prevent field mismatches."""
        meta = None  # Initialize meta to None
        # Fallback: try to get title/metadata from raw HTML if not provided
        if title == "unknown title" and raw_html:
            meta = trafilatura.extract_metadata(raw_html)
            if meta and meta.title:
                title = meta.title

        if not pub_date and raw_html:
            if not meta:  # Only run trafilatura again if we didn't run it above
                meta = trafilatura.extract_metadata(raw_html)
            if meta and meta.date:
                pub_date = str(meta.date)

        return Article(
            url=url,
            title=title,
            original_text=extracted_text,
            tags=tags or [],
            pub_date=pub_date,
            cat1=cat1,
            publisher=publisher,
            lang=lang,
            country=country,
            # All other fields use your dataclass defaults
        )

    # ------------------------------------------------------------------
    # Entry / feed processing
    # ------------------------------------------------------------------

    # ==================================================================
    # region 4. RSS entry
    # ==================================================================

    # ------------------------------------------------------------------
    # Updated RSS Entry Processor
    # ------------------------------------------------------------------
    async def _process_rss_entry(self, feed: Feed, entry: dict) -> Optional[Article]:
        async with self._entry_sem:  # type: ignore
            cache_html_hit = ""
            for curr_attempt in range(3):
                url, curr_attempt = self._extract_url_from_entry(
                    feed, entry, curr_attempt
                )

                if not url:
                    logger.warning(
                        "Retrying with alternative URL extraction for entry with no link"
                    )
                    continue

                if not self.force_refetch and self.store.exists(url):
                    logger.info(f"[YELLOW] Cache hit, skipping fetch: {url}")
                    article = self.store.load_by_url(url)
                    if article:
                        self.imm_articles.append(article)
                    return None

                title = self._extract_or_clean(getattr(entry, "title", ""))
                source_name = self._extract_or_clean(
                    entry.get("source", {}).get("title")
                )

                if source_name and source_name != feed.publisher:
                    logger.info(
                        f"[YELLOW] Publisher mismatch: feed '{feed.publisher}' vs entry '{source_name}' for URL {url}"
                    )

                pub_ts = self._entry_timestamp(entry)
                tags = [
                    t.get("term", "")
                    for t in getattr(entry, "tags", [])
                    if isinstance(t, dict)
                ]

                try:
                    # Get inline content from RSS
                    inline = self._get_content_from_entry(entry)

                    # Fetch full text from URL
                    original_text, parsed_title, parsed_ts = await self._fetch_text(url)
                    final_title = title or parsed_title or "unknown title"

                    if not original_text or len(original_text) <= 50:
                        if inline and len(inline) > 50:
                            logger.info(
                                f"[GREEN] [HIT] Using inline content: {inline[:100]}..."
                            )
                            original_text = inline

                    if (
                        self.strict
                        and original_text
                        and (
                            "<html" in original_text
                            or "<!doctype html>" in original_text
                        )
                    ):
                        logger.info(
                            f"[RED] Fetched content looks like HTML, failed for URL: {url}"
                        )
                        cache_html_hit = original_text
                        continue

                except Exception:
                    logger.exception(
                        f"Failed to fetch/extract article text for URL: {url}"
                    )
                    continue

                if not original_text or len(original_text) < MIN_TEXT_LEN:
                    logger.warning(
                        f"Extraction failed or too short ({len(original_text or '')} chars), not caching."
                    )
                    continue

                # Strict mode filtering
                if self.strict:
                    for phrase in COMMON_ERR_PHRASES:
                        if phrase.lower() in original_text.lower():
                            logger.warning(
                                f"Extraction yielded known error page content, abandoning: {phrase}"
                            )
                            cache_html_hit = original_text
                            break
                    else:
                        is_useful, reason = is_useful_text(original_text)
                        if not is_useful:
                            logger.warning(
                                f"Extraction yielded non-useful text ({reason}), abandoning."
                            )
                            cache_html_hit = original_text
                            break

                # Use the unified builder
                article = self._build_article(
                    url=url,
                    raw_html="",  # We don't pass raw_html here to avoid double-running trafilatura metadata extraction
                    extracted_text=original_text,
                    title=final_title,
                    pub_date=(
                        datetime.fromtimestamp(pub_ts, tz=timezone.utc).strftime(
                            "%Y-%m-%d"
                        )
                        if pub_ts != -1
                        else (parsed_ts or "")
                    ),
                    tags=tags,
                    publisher=feed.publisher,
                    lang=feed.lang,
                    country=feed.country,
                    cat1=feed.category,
                )

                await asyncio.get_running_loop().run_in_executor(
                    None, self.store.save_one, article
                )  # type: ignore
                self.imm_articles.append(article)
                return article

            logger.warning(
                f"Skipping entry after 3 attempts: {getattr(entry, 'title', '')}"
            )
            self.failed_feed_entries.append(
                {
                    "feed": feed.to_dict(),
                    "entry": entry,
                    "url": entry.get("link", "no link"),
                    "err": cache_html_hit or "No valid URL or content after 3 attempts",
                }
            )
            return None

    # ==================================================================
    # region 3. RSS: Feed lvl
    # ==================================================================
    async def _process_rss_feed(self, feed: Feed, raw: str):
        parsed = feedparser.parse(raw)
        if not parsed.entries:  # skip Google News feeds for now
            recovered = _recover_feed_body(raw)
            if recovered is not None:
                raw = recovered
                parsed = feedparser.parse(raw)

        if parsed.bozo and not feed.bozo_ignore:
            logger.warning(f"Got BOZO for {feed.url}: {parsed.bozo_exception}")

        entries_sorted = sorted(parsed.entries, key=self._entry_timestamp, reverse=True)
        logger.info(
            f"[PURPLE] Valid {len(entries_sorted)} (Got {len(parsed.entries)}) entries from feed: {feed.publisher}"
        )
        entries_filtered = [e for e in entries_sorted if self._is_entry_within_age(e)]
        if self.limit_per_feed is not None:
            batch = entries_filtered[: min(len(entries_filtered), self.limit_per_feed)]
        else:
            batch = entries_filtered

        if "news.google.com" in feed.url:
            decode_tasks = [self.decode_gnews_url(entry) for entry in batch]
            await asyncio.gather(*decode_tasks, return_exceptions=True)

        ## NORMAL PATH

        logger.info(
            f"[PURPLE] Processing {len(batch)} entries from feed: {feed.publisher}"
        )
        if len(entries_filtered) == 0:
            self.require_check.append(
                (feed, len(parsed.entries), len(entries_filtered))
            )
            return []  # no entries

        # Confirm feed data
        langcc = parsed.feed.get("language", "").split("-")
        lang = langcc[0] if len(langcc) > 0 else None
        cc = langcc[1] if len(langcc) > 1 else None

        if feed.lang and lang and feed.lang != lang:
            logger.warning(
                f"[YELLOW] Feed language mismatch: expected {feed.lang}, got {lang}"
            )
        if feed.country and cc and feed.country != cc:
            logger.warning(
                f"[YELLOW] Feed country mismatch: expected {feed.country}, got {cc}"
            )

        ######################
        # new data
        tags = parsed.feed.get("subtitle", None)
        if tags:
            logger.info(f"[GREEN] Interesting feed tags: {tags}")
        ######################

        tasks = [self._process_rss_entry(feed, e) for e in batch]
        results = await atqdm.gather(*tasks, desc="Processing feed entries")
        return [r for r in results if r]

    # ==================================================================
    # region 3. HTML
    # ==================================================================
    # async def _process_html_page(self, feed: Feed, url: str, html: str):
    #     if not self.force_refetch and self.store.exists(url):
    #         article = self.store.load_by_url(url)
    #         if article:
    #             self.imm_articles.append(article)
    #         return

    #     resolved_url, resolved_html = await self._resolve_real_url(url, html)
    #     text, parsed_title = await self._extract_article(feed, resolved_html)
    #     if len(text) < MIN_TEXT_LEN:
    #         logger.warning(f"HTML page extraction failed or too short for URL {url}")
    #         return

    #     metadata = trafilatura.extract_metadata(html)
    #     article = Article(
    #         url=url,
    #         title=parsed_title or (metadata.title if metadata else ""),  # type: ignore
    #         original_text=text,
    #         tags=[],
    #         pub_date=metadata.date if metadata else "",  # type: ignore
    #         desc=metadata.description if metadata else "",  # type: ignore
    #         keywords=[],
    #         publisher=feed.publisher,
    #         lang=feed.lang,
    #         country=feed.country,
    #         cat1=feed.category,
    #     )
    #     await asyncio.get_event_loop().run_in_executor(
    #         None, self.store.save_one, article, feed.url
    #     )
    #     self.imm_articles.append(article)

    # ------------------------------------------------------------------
    # Updated HTML Page Processor
    # ------------------------------------------------------------------
    async def _process_html_page(self, feed: Feed, url: str, html: str):
        if not self.force_refetch and self.store.exists(url):
            article = self.store.load_by_url(url)
            if article:
                self.imm_articles.append(article)
            return

        resolved_url, resolved_html = await self._resolve_real_url(url, html)
        text, parsed_title = await self._extract_article(resolved_html)

        if len(text) < MIN_TEXT_LEN:
            logger.warning(f"HTML page extraction failed or too short for URL {url}")
            return

        # Use the unified builder
        article = self._build_article(
            url=url,
            raw_html=resolved_html,  # Pass raw HTML so builder can extract missing title/date
            extracted_text=text,
            title=parsed_title or "unknown title",
            publisher=feed.publisher,
            lang=feed.lang,
            country=feed.country,
            cat1=feed.category,
        )

        await asyncio.get_running_loop().run_in_executor(
            None, self.store.save_one, article
        )  # type: ignore
        self.imm_articles.append(article)

    # ==================================================================
    # region 2. Fetch and process a single feed
    # ==================================================================
    async def _fetch_and_process(self, feed: Feed):
        raw, content_type = None, ""
        for tier in self._fetch_tiers():
            try:
                raw, content_type = await self._fetch_raw(feed.url, tier=tier)
                if not raw:
                    continue
                logger.info(
                    f"[GREEN] [HIT] Fetched feed via tier '{tier}': {feed.url} (content-type: {content_type})"
                )
                break
            except Exception:
                # logger.exception(f"[{tier}] Failed to fetch feed: {feed.url}")
                await asyncio.sleep(
                    random.uniform(
                        GENERAL_CD_PER_REQUESTS_S / 2, GENERAL_CD_PER_REQUESTS_S * 2
                    )
                )  # brief pause before next tier
                continue

        if raw is None:
            logger.info(f"[RED] Failed to fetch feed on all tiers: {feed.url}")
            return
        if not feed.type:
            logger.warning(f"[SKIP] Feed type not specified: {feed.url}")
            return
        if feed.type in ("rss", "xml-rss", "atom"):
            # logger.info("URL looks like a feed, processing as RSS/Atom: %s, %s", feed.publisher, feed.category)
            # with open(f"fetched_feed_{feed.publisher}_{feed.category}.xml", "w", encoding="utf-8") as f:
            #     f.write(raw)
            await self._process_rss_feed(feed, raw)
        elif feed.type in ("html"):
            # logger.info("Treating as HTML page: %s", feed.url)
            await self._process_html_page(feed, feed.url, raw)
        else:
            logger.warning(
                f"[SKIP] Unknown feed type '{feed.type}' for URL: {feed.url}"
            )
            return

    # ==================================================================
    # region 1. Batch processing
    # ==================================================================
    async def _batch_fetch_and_process(self):
        async def _safe(feed):
            try:
                await self._fetch_and_process(feed)
                # json_articles = [article.to_dict() for article in self.imm_articles]
                # with open("fetched_articles.json", "w", encoding="utf-8") as f:
                #     json.dump(json_articles, f)

            except Exception:
                logger.exception(f"Failed to process feed: {feed.url}")
                traceback.print_exc()

        tasks = [_safe(feed) for feed in self.feeds]
        await atqdm.gather(*tasks, desc="Processing feeds")
        # print all feeds that require a manual check
        if self.require_check:
            logger.info("[YELLOW] The following feeds require a manual check:")
            for feed, len1, len2 in self.require_check:
                logger.info(
                    f"[YELLOW] - {feed.publisher} ({feed.category}) - {feed.url} with {len1} entries (filtered: {len2})"
                )
        if self.failed_feed_entries:
            logger.info("[YELLOW] The following feed entries failed after 3 attempts:")
            for item in self.failed_feed_entries:
                logger.info(
                    f"[YELLOW] - Feed: {item['feed'].url if isinstance(item['feed'], Feed) else str(item['feed'])}, Entry: {getattr(item['entry'], 'title', 'no title')}, URL: {item['url']}, Error: {item['err']}"
                )
            with open("failed_feed_entries.json", "w", encoding="utf-8") as f:
                json.dump(self.failed_feed_entries, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Fetch from URL directly (No Feed required)
    # ------------------------------------------------------------------
    async def _fetch_article_from_url(self, url: str, **kwargs) -> Article | None:
        """Fetch and extract a single article without a Feed object."""
        if not self.force_refetch and self.store.exists(url):
            article = self.store.load_by_url(url)
            if article:
                self.imm_articles.append(article)
            return article

        if not self._is_fetchable_url(url):
            return None

        raw_html = ""
        for tier in self._fetch_tiers():
            try:
                html, _ = await self._fetch_raw(url, tier=tier)
                if not html: 
                    continue
                resolved_url, resolved_html = await self._resolve_real_url(url, html)

                if self._is_generated_shell_page(resolved_url, resolved_html):
                    continue

                raw_html = resolved_html
                break
            except Exception:
                await asyncio.sleep(GENERAL_CD_PER_REQUESTS_S)
                traceback.print_exc()

        if not raw_html:
            return None

        text, parsed_title = await self._extract_article(raw_html)
        if not text or len(text) < MIN_TEXT_LEN:
            return None

        article = self._build_article(
            url=url,
            raw_html=raw_html,
            extracted_text=text,
            title=parsed_title or "unknown title",
            publisher=kwargs.get("publisher", ""),
            lang=kwargs.get("lang", "en"),
            country=kwargs.get("country", "US"),
            cat1=kwargs.get("cat1", ""),
        )

        await asyncio.get_running_loop().run_in_executor(
            None, self.store.save_one, article
        )  # type: ignore
        self.imm_articles.append(article)
        return article

    async def fetch_articles_from_urls(
        self, urls: list[str], **kwargs
    ) -> list[Article]:
        """Fetch and extract a batch of articles without Feed objects."""
        self._ensure_semaphores()

        async def _safe_fetch(url: str) -> Article | None:
            try:
                return await self._fetch_article_from_url(url, **kwargs)
            except Exception:
                logger.exception(f"Failed to fetch standalone URL: {url}")
                traceback.print_exc()

        # Reuse existing semaphores if run() was called, else no cap
        sem = self._entry_sem

        async def _guarded_fetch(url: str) -> Article | None:
            if sem:
                async with sem:
                    return await _safe_fetch(url)
            return await _safe_fetch(url)

        tasks = [_guarded_fetch(u) for u in urls]

        xresults = await asyncio.gather(*tasks, return_exceptions=True)
        for r in xresults:
            if isinstance(r, BaseException):
                logger.exception("[RED] Task failed", exc_info=r)
        return [r for r in xresults if r and not isinstance(r, BaseException)]
