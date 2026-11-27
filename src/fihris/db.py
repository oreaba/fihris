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
    # Refreshed by the engine that owns a queued or running job; a stale one means its process
    # died (Engine.recover_interrupted). Null on jobs written before fihris 0.8.
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True),
                                                             nullable=True)


class Stamp(Base):
    """Counters that triggers raise, whoever writes the rows, so an engine in one process sees
    what another process changed with one small read before each retrieval:

    - ``access``: any document's groups, visibility, status or removal changed. The engine
      reloads its access mask.
    - ``live``: chunks were added to or removed from the live build, or a build changed state
      (activated, retired). The engine reloads the whole build. Chunks written into a build that
      is not live do not count, so a corpus re-index in progress costs nothing."""

    __tablename__ = "stamps"
    __table_args__ = {"schema": SCHEMA}

    name: Mapped[str] = mapped_column(String(20), primary_key=True)
    value: Mapped[int] = mapped_column(Integer, default=0)


STAMPS = ("access", "live")


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


ACCESS_COLUMNS = ("groups", "visible_to_all", "status", "removed")

# The stamps' triggers, per dialect. Every statement is idempotent, so ``upgrade_schema`` can
# run on any database, old or new. Dalil's Alembic migration 0002 carries the same DDL.
_S = SCHEMA
_PG_STAMPS = [
    f"CREATE TABLE IF NOT EXISTS {_S}.stamps "
    "(name varchar(20) PRIMARY KEY, value integer NOT NULL DEFAULT 0)",
    f"INSERT INTO {_S}.stamps (name, value) VALUES ('access', 0), ('live', 0) "
    "ON CONFLICT (name) DO NOTHING",
    # One function for every stamp: the trigger names it as its argument.
    f"CREATE OR REPLACE FUNCTION {_S}.bump_stamp() RETURNS trigger LANGUAGE plpgsql AS $$ "
    f"BEGIN UPDATE {_S}.stamps SET value = value + 1 WHERE name = TG_ARGV[0]; "
    "RETURN NULL; END $$",
    # Chunk changes count only when they touch the live build. One check per statement, over
    # the statement's rows, so writing a whole new build costs one query.
    f"CREATE OR REPLACE FUNCTION {_S}.bump_live_if_active() RETURNS trigger LANGUAGE plpgsql "
    f"AS $$ BEGIN IF EXISTS (SELECT 1 FROM changed c JOIN {_S}.builds b "
    "ON b.build_id = c.build_id WHERE b.state = 'active') THEN "
    f"UPDATE {_S}.stamps SET value = value + 1 WHERE name = 'live'; END IF; "
    "RETURN NULL; END $$",
    f"DROP TRIGGER IF EXISTS documents_access_stamp ON {_S}.documents",
    f"CREATE TRIGGER documents_access_stamp AFTER INSERT OR DELETE OR UPDATE OF "
    f"{', '.join(ACCESS_COLUMNS)} ON {_S}.documents "
    f"FOR EACH STATEMENT EXECUTE FUNCTION {_S}.bump_stamp('access')",
    # A build entering or leaving "active" (activation, retirement), not "building" to
    # "complete". Builds are few, so per row is cheap.
    f"DROP TRIGGER IF EXISTS builds_live_stamp ON {_S}.builds",
    f"CREATE TRIGGER builds_live_stamp AFTER UPDATE OF state ON {_S}.builds FOR EACH ROW "
    "WHEN (OLD.state IS DISTINCT FROM NEW.state AND "
    "(OLD.state = 'active' OR NEW.state = 'active')) "
    f"EXECUTE FUNCTION {_S}.bump_stamp('live')",
    f"DROP TRIGGER IF EXISTS builds_live_stamp_del ON {_S}.builds",
    f"CREATE TRIGGER builds_live_stamp_del AFTER DELETE ON {_S}.builds FOR EACH ROW "
    f"WHEN (OLD.state = 'active') EXECUTE FUNCTION {_S}.bump_stamp('live')",
    f"DROP TRIGGER IF EXISTS chunks_live_stamp_ins ON {_S}.chunks",
    f"CREATE TRIGGER chunks_live_stamp_ins AFTER INSERT ON {_S}.chunks "
    f"REFERENCING NEW TABLE AS changed FOR EACH STATEMENT "
    f"EXECUTE FUNCTION {_S}.bump_live_if_active()",
    f"DROP TRIGGER IF EXISTS chunks_live_stamp_del ON {_S}.chunks",
    f"CREATE TRIGGER chunks_live_stamp_del AFTER DELETE ON {_S}.chunks "
    f"REFERENCING OLD TABLE AS changed FOR EACH STATEMENT "
    f"EXECUTE FUNCTION {_S}.bump_live_if_active()",
    f"DROP TRIGGER IF EXISTS chunks_live_stamp_upd ON {_S}.chunks",
    f"CREATE TRIGGER chunks_live_stamp_upd AFTER UPDATE ON {_S}.chunks "
    f"REFERENCING NEW TABLE AS changed FOR EACH STATEMENT "
    f"EXECUTE FUNCTION {_S}.bump_live_if_active()",
]


def _sqlite_bump(name: str) -> str:
    return f"UPDATE stamps SET value = value + 1 WHERE name = '{name}';"


_LIVE = "EXISTS (SELECT 1 FROM builds WHERE build_id = {row}.build_id AND state = 'active')"
_SQLITE_STAMPS = [
    "CREATE TABLE IF NOT EXISTS stamps (name VARCHAR(20) PRIMARY KEY, "
    "value INTEGER NOT NULL DEFAULT 0)",
    "INSERT OR IGNORE INTO stamps (name, value) VALUES ('access', 0), ('live', 0)",
    "CREATE TRIGGER IF NOT EXISTS documents_access_stamp_ins AFTER INSERT ON documents "
    f"BEGIN {_sqlite_bump('access')} END",
    "CREATE TRIGGER IF NOT EXISTS documents_access_stamp_del AFTER DELETE ON documents "
    f"BEGIN {_sqlite_bump('access')} END",
    "CREATE TRIGGER IF NOT EXISTS documents_access_stamp_upd AFTER UPDATE OF "
    f"{', '.join(ACCESS_COLUMNS)} ON documents BEGIN {_sqlite_bump('access')} END",
    "CREATE TRIGGER IF NOT EXISTS builds_live_stamp_upd AFTER UPDATE OF state ON builds "
    "WHEN OLD.state IS NOT NEW.state AND (OLD.state = 'active' OR NEW.state = 'active') "
    f"BEGIN {_sqlite_bump('live')} END",
    "CREATE TRIGGER IF NOT EXISTS builds_live_stamp_del AFTER DELETE ON builds "
    f"WHEN OLD.state = 'active' BEGIN {_sqlite_bump('live')} END",
    "CREATE TRIGGER IF NOT EXISTS chunks_live_stamp_ins AFTER INSERT ON chunks "
    f"WHEN {_LIVE.format(row='NEW')} BEGIN {_sqlite_bump('live')} END",
    "CREATE TRIGGER IF NOT EXISTS chunks_live_stamp_del AFTER DELETE ON chunks "
    f"WHEN {_LIVE.format(row='OLD')} BEGIN {_sqlite_bump('live')} END",
    "CREATE TRIGGER IF NOT EXISTS chunks_live_stamp_upd AFTER UPDATE ON chunks "
    f"WHEN {_LIVE.format(row='NEW')} BEGIN {_sqlite_bump('live')} END",
]


def upgrade_schema(engine: Engine) -> None:
    """Bring a database made by an earlier fihris up to this one, idempotently: the jobs'
    heartbeat column (0.8), and the stamps with their triggers (0.8)."""
    from sqlalchemy import inspect

    postgres = engine.dialect.name == "postgresql"
    with engine.begin() as conn:
        columns = {c["name"] for c in inspect(conn).get_columns(
            "jobs", schema=SCHEMA if postgres else None)}
        if "heartbeat_at" not in columns:
            conn.exec_driver_sql(
                f"ALTER TABLE {SCHEMA + '.' if postgres else ''}jobs ADD COLUMN heartbeat_at "
                + ("timestamp with time zone" if postgres else "DATETIME"))
        for statement in _PG_STAMPS if postgres else _SQLITE_STAMPS:
            conn.exec_driver_sql(statement)


def create_schema(engine: Engine) -> None:
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    Base.metadata.create_all(engine)
    upgrade_schema(engine)


def sessions(engine: Engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False)
