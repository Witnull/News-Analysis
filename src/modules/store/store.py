from dataclasses import MISSING, fields
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List, Optional
import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from src.core.types import Article
from src.modules.store.models import (
    ArticleModel,
    ArticleReviewModel,
    Base,
    _SHARED_COLUMNS,
)


class ArticleStore:
    def __init__(self, db_url: str = "sqlite:///articles.db"):
        # Switch to Postgres later: "postgresql+psycopg2://user:pass@host/db"
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)  # auto-create tables

    @staticmethod
    def _hash(url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    def exists(self, url: str) -> bool:
        with Session(self.engine) as session:
            return session.get(ArticleModel, url) is not None

    def save_one(
        self,
        article: Article,
        to_review: bool = False,
        reason: str = "",
    ):
        self.save_many([article], to_review=to_review, reason=reason)

    # -- Bulk save (list of Articles -> main or review table) ------------------
    def save_many(
        self,
        articles: List[Article],
        to_review: bool = False,
        reason: str = "",
    ):
        """
        Saves a list of Articles in one transaction (single commit, not one
        commit per article — much faster for batches).

        to_review=False (default): saves into the main `articles` table.
        to_review=True: saves into the `articles_review` cache instead,
                        tagged with `reason` (e.g. "low yield score, needs human check").
        """
        model_cls = ArticleReviewModel if to_review else ArticleModel
        now = self._now()

        with Session(self.engine) as session:
            for article in articles:
                kwargs = dict(
                    url=article.url,
                    url_hash=self._hash(article.url),
                    title=article.title,
                    pub_date=article.pub_date,
                    publisher=article.publisher,
                    tags=",".join(article.tags)
                    if isinstance(article.tags, list)
                    else (article.tags or ""),
                    original_text=article.original_text,
                    summarized_text=article.summarized_text,
                    keywords=",".join(article.keywords)
                    if isinstance(article.keywords, list)
                    else (article.keywords or ""),
                    fetched_at=getattr(article, "fetched_at", now) or now,
                    cat1=article.cat1,
                    yield_score=str(article.yield_score) if article.yield_score is not None else "0",
                    yield_reason=article.yield_reason,
                    half_life_est=article.half_life_est,
                    half_life_est_reason=article.half_life_est_reason,
                    one_line_summary=article.one_line_summary,
                    lang=article.lang,
                    country=article.country,
                    is_reviewed=article.is_reviewed,
                    processed_stage=article.processed_stage,
                    entity=article.entity,
                    ticker=article.ticker,
                    log=article.log,
                )
                if to_review:
                    kwargs["review_reason"] = reason
                    kwargs["moved_at"] = now

                session.merge(model_cls(**kwargs))  # merge = insert or update

            session.commit()

    def load_by_url(self, url: str, to_review: bool = False) -> Article | None:
        model_cls = ArticleReviewModel if to_review else ArticleModel

        with Session(self.engine) as session:
            row = session.get(model_cls, url)
            if row is None:
                return None
            return self._row_to_article(row)

    def load_all(
        self,
        to_review: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **filters,
    ) -> List[Article]:
        """
        Returns articles from `articles` (or `articles_review` if to_review=True).
        Any keyword arg is treated as an equality filter on that column, e.g.
        load_all(cat1="livestock", source="reuters") or load_all(cat1=["a", "b"])
        for an IN filter.
        """
        model_cls = ArticleReviewModel if to_review else ArticleModel
        with Session(self.engine) as session:
            stmt = select(model_cls)
            for field, value in filters.items():
                column = getattr(model_cls, field, None)
                if column is None:
                    raise ValueError(
                        f"'{field}' is not a valid column on {model_cls.__name__}"
                    )
                if isinstance(value, (list, tuple, set)):
                    stmt = stmt.where(column.in_(value))
                elif value is None:
                    stmt = stmt.where(column.is_(None))
                else:
                    stmt = stmt.where(column == value)

            if limit is not None:
                stmt = stmt.limit(limit)
            if offset is not None:
                stmt = stmt.offset(offset)

            rows = session.scalars(stmt).all()
            return [self._row_to_article(row) for row in rows]

    def get_recent_articles(
        self,
        limit: int = 10,
        to_review: bool = False,
    ) -> List[Article]:
        """Returns the most recent articles by publication date or fetched_at."""
        model_cls = ArticleReviewModel if to_review else ArticleModel
        with Session(self.engine) as session:
            stmt = (
                select(model_cls)
                .order_by(model_cls.pub_date.desc(), model_cls.fetched_at.desc())
                .limit(limit)
            )
            rows = session.scalars(stmt).all()
            return [self._row_to_article(row) for row in rows]

    def load_all_by_pubdate(
        self, date: int = 7, to_review: bool = False
    ) -> List[Article]:
        """
        Returns every article currently in the main `articles` table
        (or the `articles_review` table if to_review=True), regardless of feed.
        """
        model_cls = ArticleReviewModel if to_review else ArticleModel
        with Session(self.engine) as session:
            rows = session.scalars(
                select(model_cls).where(
                    model_cls.pub_date
                    >= (
                        datetime.now(tz=timezone.utc) - timedelta(days=date)
                    ).isoformat()
                )
            ).all()
            return [self._row_to_article(row) for row in rows]

    # -- Remove ---------------------------------------------------------------
    def remove(self, url: str, to_review: bool = False) -> bool:
        """Permanently deletes an article from the main table. Returns True if a row was deleted."""
        model_cls = ArticleReviewModel if to_review else ArticleModel
        with Session(self.engine) as session:
            row = session.get(model_cls, url)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    # -- Move (main -> review cache) -------------------------------------------
    def move(self, url: str, reason: str = "", to_review: bool = True) -> bool:
        """
        Moves an article from the main `articles` table into the
        `articles_review` holding table for manual review, removing it from
        the main table. Atomic: both the insert and delete happen in one
        transaction, so a failure can't leave the row duplicated or lost.
        Returns True if the row existed and was moved.
        """
        src_cls = ArticleModel if to_review else ArticleReviewModel
        target_cls = ArticleReviewModel if to_review else ArticleModel
        with Session(self.engine) as session:
            row = session.get(src_cls, url)
            if row is None:
                return False

            # Copy every shared column dynamically rather than hand-listing
            # them again — stays correct if ArticleModel gains new fields.
            values = {col: getattr(row, col) for col in _SHARED_COLUMNS}

            if to_review:
                values["review_reason"] = reason
                values["moved_at"] = self._now()

            session.merge(target_cls(**values))
            session.delete(row)
            session.commit()
            return True

    # -- Shared row -> Article mapping ------------------------------------------
    @staticmethod
    def _row_to_article(row, exclude: set[str] = set(["no_save_pool"])) -> Article:
        exclude = exclude or set()
        kwargs = {}

        for f in fields(Article):
            if f.name in exclude:
                continue

            orm_name = f.metadata.get("orm_name", f.name)

            if not hasattr(row, orm_name):
                value = (
                    f.default
                    if f.default is not MISSING
                    else (
                        f.default_factory()
                        if f.default_factory is not MISSING
                        else None
                    )
                )
            else:
                value = getattr(row, orm_name)
                # Handle CSV-stored list fields (tags, keywords, concepts, llm_tags)
                if f.name in ("tags", "keywords", "concepts", "llm_tags") or f.metadata.get("csv"):
                    if isinstance(value, str):
                        value = [v.strip() for v in value.split(",") if v.strip()]
                    elif value is None:
                        value = []
                # Handle numeric fields
                elif f.name == "yield_score":
                    if value is not None:
                        try:
                            value = int(value)
                        except (ValueError, TypeError):
                            value = 0
                    else:
                        value = 0
                # Handle JSON dict fields
                elif f.name in ("yield_reason", "half_life_est_reason"):
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except Exception:
                            value = {}
                    elif value is None:
                        value = {}

            kwargs[f.name] = value

        return Article(**kwargs)



