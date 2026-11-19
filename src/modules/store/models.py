
import hashlib
from datetime import datetime, timezone


from sqlalchemy import Boolean, String, Text, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, declarative_base
from dataclasses import fields , is_dataclass
from sqlalchemy import inspect as sa_inspect
import inspect as py_inspect

from src.core.types import Article

Base = declarative_base()

class ArticleColumnsMixin:
    __tablename__ = "articles"

    url_hash: Mapped[str] = mapped_column(String, index=True)
    publisher: Mapped[str | None] = mapped_column(String, index=True)

    title: Mapped[str | None] = mapped_column(Text)
    pub_date: Mapped[str | None] = mapped_column(String, index=True)
    tags: Mapped[str] = mapped_column(Text)
    original_text: Mapped[str | None] = mapped_column(Text)

    summarized_text: Mapped[str | None] = mapped_column(Text)
    one_line_summary: Mapped[str | None] = mapped_column(Text)

    entity: Mapped[str| None] = mapped_column(Text)
    ticker: Mapped[str| None] = mapped_column(Text)

    keywords: Mapped[str | None] = mapped_column(Text)

    cat1: Mapped[str | None] = mapped_column(String)  # legacy free-text, kept as fallback

    yield_score: Mapped[int | None] = mapped_column(String)
    yield_reason: Mapped[dict | None] = mapped_column(JSON)
    half_life_est: Mapped[str | None] = mapped_column(String)
    half_life_est_reason: Mapped[dict | None] = mapped_column(JSON)

    lang: Mapped[str | None] = mapped_column(String)
    country: Mapped[str | None] = mapped_column(String)

    fetched_at: Mapped[str] = mapped_column(String)
    is_reviewed: Mapped[bool | None] = mapped_column(Boolean, default=False)

    processed_stage: Mapped[str | None] = mapped_column(String, default="none")  # processing stage - 0 = not processed,
    entity: Mapped[str | None] = mapped_column(String, default="")
    ticker: Mapped[str | None] = mapped_column(String, default="")

    log: Mapped[str | None] = mapped_column(String, default="none")

class ArticleModel(ArticleColumnsMixin, Base):
    __tablename__ = "articles"
    url: Mapped[str] = mapped_column(String, primary_key=True)


class ArticleReviewModel(ArticleColumnsMixin, Base):
    __tablename__ = "articles_review"
    url: Mapped[str] = mapped_column(String, primary_key=True)

    review_reason: Mapped[str | None] = mapped_column(Text)
    moved_at: Mapped[str] = mapped_column(String)


_SHARED_COLUMNS = [c.name for c in ArticleModel.__table__.columns]

    
def md5(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()

def assert_fields_match(orm_obj, dataclass_obj, exclude: set[str] = set(["no_save_pool"])):
    dc_cls = dataclass_obj if py_inspect.isclass(dataclass_obj) else dataclass_obj.__class__
    orm_cls = orm_obj if py_inspect.isclass(orm_obj) else orm_obj.__class__

    assert is_dataclass(dc_cls), f"{dc_cls} is not a dataclass"

    dc_fields = {f.name for f in fields(dc_cls)} - exclude
    orm_dict = {c.key for c in sa_inspect(orm_cls).mapper.column_attrs}

    missing = dc_fields - orm_dict
    assert not missing, f"Missing in ORM: {missing}"

print("BEGIN CHECKING....")
assert_fields_match(ArticleModel, Article)
print("Done check.")