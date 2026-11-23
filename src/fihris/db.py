"""Fihris's database: documents and their metadata, page text, chunks per index build, builds,
jobs and the extraction review queue. PostgreSQL schema ``fihris`` in production; SQLite, with
the schema mapped away, in tests and on a laptop."""

from __future__ import annotations

import datetime as dt
import re
import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    event,
)
from sqlalchemy import Text as _Text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import TypeDecorator

SCHEMA = "fihris"
_ESC, _ESC_SELF, _ESC_NUL = "\ue000", "\ue001", "\ue002"
_UNESCAPE = re.compile(f"{_ESC}([{_ESC_SELF}{_ESC_NUL}])")


class Text(TypeDecorator):
    """Text that round-trips exactly, NUL characters included: PostgreSQL text cannot hold
    U+0000, and extracted text can contain it. Stored with a reversible escape in the
    private-use area; read back unchanged."""

    impl = _Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return value.replace(_ESC, _ESC + _ESC_SELF).replace("\x00", _ESC + _ESC_NUL)

    def process_result_value(self, value, dialect):
        if value is None or _ESC not in value:
            return value
        return _UNESCAPE.sub(lambda m: _ESC if m.group(1) == _ESC_SELF else "\x00", value)


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": SCHEMA}

    doc_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    file_name: Mapped[str] = mapped_column(String(300), default="")
    relative_path: Mapped[str] = mapped_column(String(600), default="")
    code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    doc_type: Mapped[str] = mapped_column(String(40), default="unknown")
    language: Mapped[str] = mapped_column(String(12), default="unknown")
    sector: Mapped[str] = mapped_column(String(120), default="")
    department: Mapped[str] = mapped_column(String(200), default="")
    section: Mapped[str] = mapped_column(String(200), default="")
    file_format: Mapped[str] = mapped_column(String(10), default="pdf")
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paged: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    access_level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    issue_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    owner_department: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(20), default="unset")
    superseded_by: Mapped[str | None] = mapped_column(String(40), nullable=True)
    duplicate_of: Mapped[str | None] = mapped_column(String(40), nullable=True)
    edition_of: Mapped[str | None] = mapped_column(String(40), nullable=True)
    groups: Mapped[list] = mapped_column(JSON, default=list)
    visible_to_all: Mapped[bool] = mapped_column(Boolean, default=False)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    index_state: Mapped[str] = mapped_column(String(20), default="not_indexed")
    checksum_sha256: Mapped[str] = mapped_column(String(64), default="")
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    original_filename: Mapped[str] = mapped_column(String(300), default="")
    source_path: Mapped[str] = mapped_column(String(800), default="")
    removed: Mapped[bool] = mapped_column(Boolean, default=False)
    last_job_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_by: Mapped[str | None] = mapped_column(String(40), nullable=True)

    @property
    def searchable(self) -> bool:
        return not self.removed and self.status not in ("superseded", "duplicate")


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = {"schema": SCHEMA}

    doc_id: Mapped[str] = mapped_column(String(40), ForeignKey(f"{SCHEMA}.documents.doc_id"),
                                        primary_key=True)
    page: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="ok")
    text: Mapped[str] = mapped_column(Text, default="")
    md: Mapped[str] = mapped_column(Text, default="")
    text_source: Mapped[str] = mapped_column(String(12), default="extracted")
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = {"schema": SCHEMA}

    build_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    page: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk: Mapped[int] = mapped_column(Integer, primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    headings: Mapped[list] = mapped_column(JSON, default=list)
    lang: Mapped[str] = mapped_column(String(2))


class Build(Base):
    __tablename__ = "builds"
    __table_args__ = {"schema": SCHEMA}

    build_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    state: Mapped[str] = mapped_column(String(12), default="building")
    collection: Mapped[str] = mapped_column(String(80))
    config_sha256: Mapped[str] = mapped_column(String(64))
    units_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    vectors_source: Mapped[str] = mapped_column(String(300), default="embedded")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)
    activated_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True),
                                                             nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(40), nullable=True)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = {"schema": SCHEMA}

    job_id: Mapped[str] = mapped_column(String(40), primary_key=True, default=lambda: new_id("job"))
    type: Mapped[str] = mapped_column(String(20))
    doc_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    state: Mapped[str] = mapped_column(String(12), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    steps: Mapped[list] = mapped_column(JSON, default=list)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    pages_sent_to_review: Mapped[int] = mapped_column(Integer, default=0)
    target_build: Mapped[str | None] = mapped_column(String(40), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)
    created_by: Mapped[str | None] = mapped_column(String(40), nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True),
                                                            nullable=True)


class ExtractionItem(Base):
    __tablename__ = "extraction_items"
    __table_args__ = {"schema": SCHEMA}

    item_id: Mapped[str] = mapped_column(String(40), primary_key=True,
                                         default=lambda: new_id("ext"))
    doc_id: Mapped[str] = mapped_column(String(40), ForeignKey(f"{SCHEMA}.documents.doc_id"))
    page: Mapped[int] = mapped_column(Integer)
    gate_reasons: Mapped[list] = mapped_column(JSON, default=list)
    ocr_passed_check: Mapped[bool] = mapped_column(Boolean, default=False)
    repaired_text: Mapped[str] = mapped_column(Text, default="")
    repaired_md: Mapped[str] = mapped_column(Text, default="")
    ocr_text: Mapped[str] = mapped_column(Text, default="")
    ocr_md: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(10), default="pending")
    decision: Mapped[str | None] = mapped_column(String(20), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    decided_by: Mapped[str | None] = mapped_column(String(40), nullable=True)
    decided_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True),
                                                           nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=now)


def engine_for(url: str) -> Engine:
    """An engine for ``url``. SQLite has no schemas, so ``fihris`` (and Dalil's ``dalil``) map to
    the default one, and foreign keys are switched on."""
    if url.startswith("sqlite"):
        if url.startswith("sqlite:///") and ":memory:" not in url:
            from pathlib import Path

            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        base = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(base, "connect")
        def _pragma(conn, _record):
            conn.execute("PRAGMA foreign_keys=ON")

        return base.execution_options(schema_translate_map={SCHEMA: None, "dalil": None})
    return create_engine(url, pool_pre_ping=True)


def create_schema(engine: Engine) -> None:
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    Base.metadata.create_all(engine)


def sessions(engine: Engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False)
