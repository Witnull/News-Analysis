import time
from dataclasses import dataclass, asdict
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException

@dataclass(frozen=True)
class Hit:
    title: str
    href: str
    body: str
    query: str  # which query surfaced this, for traceability in the graph step

def search(
    query: str,
    max_results: int = 8,
    kind: str = "text",       # "text" or "news"
    timelimit: str | None = None,  # d/w/m/y
    region: str = "wt-wt",
    retries: int = 3,
    backoff: float = 2.0,
) -> list[Hit]:
    """One query -> list[Hit]. Retries on rate limit, gives up quietly on other errors."""
    fn_name = "news" if kind == "news" else "text"
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                fn = getattr(ddgs, fn_name)
                raw = fn(query, region=region, max_results=max_results, timelimit=timelimit)
                return [
                    Hit(r.get("title", ""), r.get("url", r.get("href", "")), r.get("body", ""), query)
                    for r in raw
                ]
        except RatelimitException:
            time.sleep(backoff * (attempt + 1))
        except DDGSException:
            return []
    return []


def batch_search(
    queries: list[str],
    max_results: int = 8,
    kind: str = "text",
    timelimit: str | None = None,
    delay: float = 1.0,
) -> list[Hit]:
    """Run several queries, dedup hits by URL. This is what the recursive-dive
    step calls each round with its expanded keyword/question set."""
    seen: dict[str, Hit] = {}
    for i, q in enumerate(queries):
        for hit in search(q, max_results=max_results, kind=kind, timelimit=timelimit):
            seen.setdefault(hit.href, hit)
        if i < len(queries) - 1:
            time.sleep(delay)  # ponytail: flat delay, swap for token-bucket if volume grows
    return list(seen.values())


# def demo():
#     hits = batch_search(["duckduckgo search python library"], max_results=3)
#     assert hits, "expected at least one hit"
#     assert all(h.href for h in hits)
#     print(f"ok: {len(hits)} hits, e.g. {hits[0].title!r}")


# if __name__ == "__main__":
#     demo()