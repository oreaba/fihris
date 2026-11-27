"""The engine: the live index build in memory, retrieval, document changes and their jobs.

One ``Engine`` per process. It holds the active build's chunk set (texts, languages, BM25 and
the access mask) and a client for its vector collection, behind the live alias. Document
changes run as jobs on one worker thread, so the live build changes one document at a time.
A corpus re-index builds a new collection and switches the alias only when it is complete.

Other processes may share the database (``fihris access``, ``fihris ingest``, ``fihris
activate``):
- **Stamps:** triggers raise ``stamps.access`` whenever a document's access changes, and
  ``stamps.live`` whenever the live build's chunks change or another build becomes live. Each
  retrieval reads both: a moved ``live`` reloads the whole build, a moved ``access`` the mask.
- **Access changes** are written to whichever build is live in the database at that moment, not
  to the one this engine last loaded.
- **Activation** re-applies every document's current access to the target build before the
  alias moves, so a build made or last live earlier never serves stale groups.
- **Jobs** carry a heartbeat while their engine is alive. ``recover_interrupted`` fails only
  jobs whose heartbeat has gone stale, never another live process's.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import shutil
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from fihris.chunking import doc_title
from fihris.chunks import Access, ChunkSet, chunk_document, unit_language
from fihris.config import PINNED, Endpoints
from fihris.db import (
    Build,
    Chunk,
    Document,
    ExtractionItem,
    Job,
    Page,
    Stamp,
    new_id,
    now,
)
from fihris.ids import make_doc_id
from fihris.ingest.extract import ExtractionError, PageRow, extract_file
from fihris.library import NotFound, document_groups
from fihris.principal import Principal
from fihris.retrieve import Retrieval, Retriever, detect_language
from fihris.units import Unit, fingerprint
from fihris.vectors import Point, collection_name, open_index

EMBED_BATCH = 64
HEARTBEAT_SECONDS = 10.0
STALE_AFTER = dt.timedelta(seconds=60)
LIVE_JOB_STATES = ("queued", "running")
STEPS = {"ingest": ["extract", "quality_gate", "ocr", "chunk", "embed", "index"],
         "replace": ["extract", "quality_gate", "ocr", "chunk", "embed", "index"],
         "remove": ["remove"],
         "reindex_document": ["chunk", "embed", "index"],
         "rechunk_page": ["chunk", "embed", "index"],
         "reindex_corpus": ["chunk", "embed", "index", "switch_build"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_id_now() -> str:
    return "build-" + dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


class JobCancelled(RuntimeError):
    pass


class Engine:
    def __init__(self, endpoints: Endpoints, make_session: sessionmaker, index=None,
                 embedder=None, scorer=None, translator=None):
        self.endpoints, self.make_session = endpoints, make_session
        self._index, self._embedder, self._scorer, self._translator = (index, embedder, scorer,
                                                                       translator)
        self.chunks: ChunkSet | None = None
        self.build: Build | None = None
        self.lock = threading.RLock()
        self.jobs = ThreadPoolExecutor(1, thread_name_prefix="fihris-job")
        self.stamps: dict[str, int] = {}
        self._mine: set[str] = set()
        self._mine_lock = threading.Lock()
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None

    # --- clients, created on first use ---------------------------------------------------
    @property
    def index(self):
        if self._index is None:
            self._index = open_index(self.endpoints, self.make_session.kw["bind"])
        return self._index

    @property
    def embedder(self):
        if self._embedder is None:
            from fihris.clients import EmbeddingClient

            self._embedder = EmbeddingClient(self.endpoints)
        return self._embedder

    @property
    def scorer(self):
        if self._scorer is None:
            from fihris.clients import RerankClient

            self._scorer = RerankClient(self.endpoints)
        return self._scorer

    @property
    def translator(self):
        if self._translator is None:
            from fihris.clients import open_translator

            self._translator = open_translator(self.endpoints)
        return self._translator

    # --- the live build -------------------------------------------------------------------
    def active_build(self, session: Session) -> Build | None:
        return session.scalars(select(Build).where(Build.state == "active")).first()

    def access_map(self, session: Session) -> dict[str, Access]:
        return {d.doc_id: Access(tuple(document_groups(d)), d.searchable)
                for d in session.scalars(select(Document)).all()}

    @staticmethod
    def read_stamps(session: Session) -> dict[str, int]:
        return {s.name: s.value for s in session.scalars(select(Stamp)).all()}

    def load(self) -> None:
        """Read the active build's chunks into memory. No active build leaves search empty."""
        with self.lock, self.make_session() as session:
            self.stamps = self.read_stamps(session)
            build = self.active_build(session)
            if build is None:
                self.build, self.chunks = None, None
                return
            rows = session.scalars(select(Chunk).where(Chunk.build_id == build.build_id)
                                   .order_by(Chunk.ordinal)).all()
            units = [Unit(r.doc_id, r.page, "", r.text, r.chunk) for r in rows]
            self.chunks = ChunkSet(units, self.access_map(session))
            self.build = build

    def refresh_access(self, doc_ids: str | Iterable[str] | None = None) -> None:
        """After documents' groups, visibility or status change: the in-memory mask, and, for
        the documents named, their points in the live build. One id or a list of them."""
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        with self.lock:
            self._catch_up(force_access=True)
            with self.make_session() as session:
                # The build live now, in the database: another process may have switched
                # builds since this engine loaded one.
                live = self.active_build(session)
                access = self.chunks.access if self.chunks is not None else (
                    self.access_map(session))
            named = {d: (access[d].groups, access[d].searchable)
                     for d in (doc_ids or []) if d in access}
            if live is not None and named:
                self.index.set_access_many(live.collection, named)

    def sync(self) -> str | None:
        """Catch up with what other processes changed: the whole build when the live build
        changed (``"build"``), the access mask when only access changed (``"access"``), or
        nothing (``None``). One small read when nothing changed; the decision is re-made under
        the lock, so retrievals that all see one change reload once between them."""
        with self.make_session() as session:
            seen = self.read_stamps(session)
        if seen == self.stamps:
            return None
        with self.lock:
            return self._catch_up()

    def _catch_up(self, force_access: bool = False) -> str | None:
        """Under ``self.lock``. Each counter is recorded from a read made before the data it
        stands for, so a change committed in between is never marked as seen."""
        with self.make_session() as session:
            stamps = self.read_stamps(session)
            if stamps.get("live") != self.stamps.get("live"):
                session.close()
                self.load()                  # reads the stamps first, then the build
                return "build"
            if not force_access and stamps.get("access") == self.stamps.get("access"):
                return None
            access = self.access_map(session)
        if self.chunks is not None:
            self.chunks.set_access(access)
        self.stamps["access"] = stamps.get("access", 0)
        return "access"

    def retriever(self) -> Retriever:
        if self.chunks is None:
            raise NotFound("no active index build")
        return Retriever(self.chunks, self.index, self.embedder, self.scorer, self.translator)

    def retrieve(self, question: str, principal: Principal,
                 language: str | None = None) -> Retrieval:
        self.sync()
        with self.lock:
            retriever = self.retriever()
        return retriever.retrieve(question, principal, language)

    def chunk_text(self, doc_id: str, page: int, chunk: int) -> str | None:
        return self.chunks.texts.get((doc_id, page, chunk)) if self.chunks else None

    # --- vectors -------------------------------------------------------------------------
    def embed_chunks(self, texts: Sequence[str],
                     progress: Callable[[int], None] | None = None) -> np.ndarray:
        parts = []
        for start in range(0, len(texts), EMBED_BATCH):
            parts.append(self.embedder.embed(texts[start:start + EMBED_BATCH],
                                             PINNED.document_truncate_tokens))
            if progress:
                progress(min(start + EMBED_BATCH, len(texts)))
        return np.concatenate(parts) if parts else np.zeros((0, PINNED.vector_dim), np.float32)

    def points(self, rows: Sequence[dict], vectors: np.ndarray,
               access: dict[str, Access]) -> list[Point]:
        out = []
        for row, vector in zip(rows, vectors, strict=True):
            a = access.get(row["doc_id"], Access((), False))
            out.append(Point(row["doc_id"], row["page"], row["chunk"], vector,
                             unit_language(row["text"]), a.groups, a.searchable))
        return out

    # --- builds ----------------------------------------------------------------------------
    def create_build(self, rows: Sequence[dict], vectors: np.ndarray, created_by: str | None,
                     vectors_source: str, activate: bool, build_id: str | None = None) -> str:
        """A new build from chunk rows and their vectors: its chunk rows, its collection, and,
        when ``activate``, the live alias moved to it."""
        build_id = build_id or build_id_now()
        collection = collection_name(build_id)
        units = [Unit(r["doc_id"], r["page"], "", r["text"], r["chunk"]) for r in rows]
        with self.make_session() as session:
            access = self.access_map(session)
            session.add(Build(build_id=build_id, state="building", collection=collection,
                              config_sha256=PINNED.sha256, units_sha256=fingerprint(units),
                              document_count=len({r["doc_id"] for r in rows}),
                              chunk_count=len(rows), vectors_source=vectors_source,
                              created_by=created_by))
            for i, r in enumerate(rows):
                session.add(Chunk(build_id=build_id, doc_id=r["doc_id"], page=r["page"],
                                  chunk=r["chunk"], ordinal=i, text=r["text"],
                                  headings=r.get("headings") or [], lang=unit_language(r["text"])))
            session.commit()
        self.index.create(collection, self.points(rows, vectors, access))
        with self.make_session() as session:
            build = session.get(Build, build_id)
            build.state = "complete"
            session.commit()
        if activate:
            self.activate(build_id)
        return build_id

    def activate(self, build_id: str) -> None:
        """Make a complete or retired build live. Every document's current groups and
        searchable flag are written to its points first, so nothing removed, superseded or
        restricted since the build was written or last live can be returned."""
        with self.lock, self.make_session() as session:
            build = session.get(Build, build_id)
            if build is None or build.state not in ("complete", "retired"):
                raise ValueError(f"build {build_id} is not complete")
            access = self.access_map(session)
            self.index.set_access_many(build.collection, {
                d: (a.groups, a.searchable) for d, a in access.items()})
            self.index.activate(build.collection)
            for old in session.scalars(select(Build).where(Build.state == "active")).all():
                old.state = "retired"
            build.state, build.activated_at = "active", now()
            session.commit()
        self.load()

    def builds(self) -> list[Build]:
        with self.make_session() as session:
            return list(session.scalars(select(Build).order_by(Build.created_at.desc())).all())

    # --- bulk import of documents and pages ----------------------------------------------------
    def upsert_documents(self, rows: Iterable[dict]) -> int:
        """Document rows from any catalogue: each row holds ``Document`` column values and at
        least ``relative_path`` (under ``corpus_root``) or ``doc_id``. A missing ``doc_id`` is
        derived from the relative path; size and checksum are read from the file when it exists.
        Groups start empty and nothing is visible to all unless a row says so. Returns the number
        of documents stored."""
        columns = set(Document.__table__.columns.keys())
        with self.make_session() as session:
            for row in rows:
                relative = row.get("relative_path") or row.get("file_name") or ""
                values = {k: v for k, v in row.items() if k in columns}
                values.setdefault("doc_id", make_doc_id(relative))
                values.setdefault("file_name", Path(relative).name)
                values.setdefault("original_filename", values["file_name"])
                values.setdefault("title", doc_title(relative))
                values.setdefault("index_state", "pending")
                source = self.endpoints.corpus_root / relative if relative else None
                if source is not None and source.is_file():
                    values.setdefault("source_path", str(source))
                    values.setdefault("file_size_bytes", source.stat().st_size)
                    values.setdefault("checksum_sha256", sha256_file(source))
                    values.setdefault("file_format", source.suffix.lower().lstrip("."))
                session.merge(Document(**values))
            session.commit()
            return session.scalar(select(func.count()).select_from(Document))

    def import_pages(self, index_path: Path) -> int:
        """Page rows from a page index (JSON lines with ``doc_id``, ``page``, ``status``,
        ``text`` and optional ``md``), for documents extracted elsewhere."""
        n = 0
        with self.make_session() as session, index_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                session.merge(Page(doc_id=row["doc_id"], page=row["page"], status=row["status"],
                                   text=row["text"], md=row.get("md", ""),
                                   text_source="extracted"))
                n += 1
            session.commit()
        return n

    # --- jobs ------------------------------------------------------------------------------
    def submit(self, job_type: str, doc_id: str | None, created_by: str | None,
               **kwargs) -> str:
        if job_type not in STEPS:
            raise ValueError(f"unknown job type {job_type}")
        with self.make_session() as session:
            job = Job(job_id=new_id("job"), type=job_type, doc_id=doc_id, created_by=created_by,
                      heartbeat_at=now(),
                      steps=[{"step": s, "state": "queued", "done": 0, "total": 0,
                              "started_at": None, "finished_at": None} for s in STEPS[job_type]])
            session.add(job)
            if doc_id and (doc := session.get(Document, doc_id)):
                doc.last_job_id = job.job_id
            session.commit()
            job_id = job.job_id
        with self._mine_lock:
            self._mine.add(job_id)
        self._start_heartbeat()
        self.jobs.submit(self._run_job, job_id, kwargs)
        return job_id

    # --- heartbeats and recovery ------------------------------------------------------------
    def _start_heartbeat(self) -> None:
        if self._heartbeat is None or not self._heartbeat.is_alive():
            self._heartbeat = threading.Thread(target=self._beat_loop, daemon=True,
                                               name="fihris-heartbeat")
            self._heartbeat.start()

    def _beat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            # A missed beat only ages the heartbeat; the next one catches up.
            with contextlib.suppress(Exception):
                self.beat()

    def beat(self) -> None:
        """Refresh the heartbeat of every job this engine still owns."""
        with self._mine_lock:
            mine = list(self._mine)
        if not mine:
            return
        with self.make_session() as session:
            for job in session.scalars(select(Job).where(Job.job_id.in_(mine),
                                                         Job.state.in_(LIVE_JOB_STATES))):
                job.heartbeat_at = now()
            session.commit()

    def recover_interrupted(self, stale_after: dt.timedelta = STALE_AFTER) -> list[str]:
        """Jobs left queued or running by a process that died become failed, with the code
        ``interrupted``. A job is left alone while its heartbeat is fresher than
        ``stale_after``, so jobs that ``fihris ingest`` or ``fihris reindex`` are running in
        another live process are never touched. Returns the ids it failed."""
        cutoff = now() - stale_after
        with self._mine_lock:
            mine = set(self._mine)
        failed = []
        with self.make_session() as session:
            for job in session.scalars(select(Job).where(Job.state.in_(LIVE_JOB_STATES))).all():
                beat = job.heartbeat_at
                if beat is not None and beat.tzinfo is None:
                    beat = beat.replace(tzinfo=dt.UTC)
                if job.job_id in mine or (beat is not None and beat >= cutoff):
                    continue
                self._finish(session, job, "failed", [{
                    "code": "interrupted",
                    "detail": "the process running this job stopped before it finished"}])
                failed.append(job.job_id)
            session.commit()
        return failed

    def close(self) -> None:
        """Stop the heartbeat and wait for queued jobs."""
        self._stop.set()
        self.jobs.shutdown(wait=True)

    def cancel(self, job_id: str) -> bool:
        with self.make_session() as session:
            job = session.get(Job, job_id)
            if job is None or job.state not in ("queued", "running"):
                return False
            job.cancel_requested = True
            session.commit()
            return True

    def _step(self, job_id: str, step: str, state: str, done: int | None = None,
              total: int | None = None) -> None:
        with self.make_session() as session:
            job = session.get(Job, job_id)
            if job.cancel_requested and state == "running":
                raise JobCancelled(job_id)
            steps = [dict(s) for s in job.steps]
            for s in steps:
                if s["step"] == step:
                    s["state"] = state
                    if done is not None:
                        s["done"] = done
                    if total is not None:
                        s["total"] = total
                    stamp = now().isoformat()
                    if state == "running" and not s["started_at"]:
                        s["started_at"] = stamp
                    if state in ("succeeded", "failed", "cancelled"):
                        s["finished_at"] = stamp
            job.steps = steps
            finished = sum(s["state"] == "succeeded" for s in steps)
            job.progress = round(finished / len(steps), 3)
            if state == "running":
                job.state = "running"
            session.commit()

    def _run_job(self, job_id: str, kwargs: dict) -> None:
        with self.make_session() as session:
            job = session.get(Job, job_id)
            job_type, doc_id = job.type, job.doc_id
        try:
            if job_type in ("ingest", "replace"):
                self._ingest(job_id, doc_id, Path(kwargs["path"]))
            elif job_type == "remove":
                self._remove(job_id, doc_id)
            elif job_type in ("reindex_document", "rechunk_page"):
                self._reindex_document(job_id, doc_id)
            elif job_type == "reindex_corpus":
                self._reindex_corpus(job_id, kwargs.get("created_by"))
            state, errors = "succeeded", []
        except JobCancelled:
            state, errors = "cancelled", []
        except Exception as error:  # noqa: BLE001 - a job records its failure
            code = str(error).split(":", 1)[0] if isinstance(error, ExtractionError) else (
                "job_failed")
            state, errors = "failed", [{"code": code, "detail": f"{type(error).__name__}: "
                                                                f"{error}"[:500]}]
        try:
            with self.make_session() as session:
                self._finish(session, session.get(Job, job_id), state, errors)
                session.commit()
        finally:
            with self._mine_lock:
                self._mine.discard(job_id)

    @staticmethod
    def _finish(session: Session, job: Job, state: str, errors: list[dict]) -> None:
        job.state, job.finished_at = state, now()
        job.errors = list(job.errors or []) + errors
        job.steps = [dict(s, state=(state if s["state"] in LIVE_JOB_STATES else s["state"]))
                     for s in job.steps]
        if state == "succeeded":
            job.progress = 1.0
        if job.doc_id and (doc := session.get(Document, job.doc_id)) and state == "failed":
            doc.index_state = "failed"
            doc.flags = sorted(set(doc.flags or []) | {"ingestion_failed"})

    # --- one document -------------------------------------------------------------------------
    def store_pages(self, session: Session, doc: Document, rows: list[PageRow]) -> int:
        """Page rows and the extraction review queue. Review pages keep their repaired text
        indexed until an admin decides."""
        session.execute(delete(Page).where(Page.doc_id == doc.doc_id))
        session.execute(delete(ExtractionItem).where(ExtractionItem.doc_id == doc.doc_id,
                                                     ExtractionItem.status == "pending"))
        review = 0
        for r in rows:
            session.add(Page(doc_id=doc.doc_id, page=r.page, status=r.status, text=r.text,
                             md=r.md, text_source=r.text_source))
            if r.status == "review" and r.text_source != "ocr":
                review += 1
                session.add(ExtractionItem(doc_id=doc.doc_id, page=r.page,
                                           gate_reasons=r.reasons, ocr_passed_check=r.ocr_passed,
                                           repaired_text=r.text, repaired_md=r.md,
                                           ocr_text=r.ocr_text, ocr_md=r.ocr_md))
        flags = set(doc.flags or []) - {"extraction_review_pending", "ingestion_failed"}
        if review:
            flags.add("extraction_review_pending")
        doc.flags = sorted(flags)
        return review

    def document_chunks(self, session: Session, doc: Document) -> list[dict]:
        pages = [{"page": p.page, "text": p.text, "md": p.md}
                 for p in session.scalars(select(Page).where(Page.doc_id == doc.doc_id,
                                                             Page.excluded.is_(False))).all()]
        return list(chunk_document(pages, doc.title, doc.doc_id)) if pages else []

    def ensure_build(self) -> None:
        """An empty active build, on a fresh database, so the first document has one to join."""
        with self.make_session() as session:
            if self.active_build(session) is not None:
                return
        self.create_build([], np.zeros((0, PINNED.vector_dim), np.float32), None, "empty",
                          activate=True)

    def _replace_in_live(self, job_id: str, doc_id: str, rows: list[dict]) -> None:
        """Swap one document's chunks in the active build: its chunk rows and its points."""
        self.ensure_build()
        self._step(job_id, "embed", "running", 0, len(rows))
        vectors = self.embed_chunks([r["text"] for r in rows],
                                    lambda d: self._step(job_id, "embed", "running", d))
        self._step(job_id, "embed", "succeeded", len(rows), len(rows))
        self._step(job_id, "index", "running", 0, len(rows))
        with self.lock, self.make_session() as session:
            build = self.active_build(session)
            if build is None:
                raise NotFound("no active index build")
            session.execute(delete(Chunk).where(Chunk.build_id == build.build_id,
                                                Chunk.doc_id == doc_id))
            last = session.scalar(select(func.max(Chunk.ordinal))
                                  .where(Chunk.build_id == build.build_id)) or 0
            for i, r in enumerate(rows, last + 1):
                session.add(Chunk(build_id=build.build_id, doc_id=doc_id, page=r["page"],
                                  chunk=r["chunk"], ordinal=i, text=r["text"],
                                  headings=r.get("headings") or [], lang=unit_language(r["text"])))
            access = self.access_map(session)
            self.index.delete_document(build.collection, doc_id)
            self.index.upsert(build.collection, self.points(rows, vectors, access))
            build.chunk_count = session.scalar(select(func.count()).select_from(Chunk)
                                               .where(Chunk.build_id == build.build_id))
            doc = session.get(Document, doc_id)
            doc.index_state = "indexed" if rows else "not_indexed"
            session.commit()
        self.load()
        self._step(job_id, "index", "succeeded", len(rows), len(rows))

    def _ingest(self, job_id: str, doc_id: str, path: Path) -> None:
        work = self.endpoints.work_dir / "work" / doc_id
        self._step(job_id, "extract", "running")
        rows, warnings = extract_file(path, work, self.endpoints)
        self._step(job_id, "extract", "succeeded", len(rows), len(rows))
        self._step(job_id, "quality_gate", "succeeded",
                   sum(r.status == "ok" for r in rows), len(rows))
        self._step(job_id, "ocr", "succeeded", sum(r.text_source == "ocr" for r in rows),
                   sum(r.status == "review" for r in rows))
        with self.make_session() as session:
            doc = session.get(Document, doc_id)
            review = self.store_pages(session, doc, rows)
            doc.page_count = len(rows) if doc.paged else None
            text = " ".join(r.text for r in rows)
            if doc.language == "unknown" and text.strip():
                doc.language = detect_language(text)
            job = session.get(Job, job_id)
            job.pages_sent_to_review = review
            job.errors = [{"code": w.split(":", 1)[0], "detail": w} for w in warnings]
            session.commit()
            self._step(job_id, "chunk", "running")
            chunks = self.document_chunks(session, doc)
        self._step(job_id, "chunk", "succeeded", len(chunks), len(chunks))
        self._replace_in_live(job_id, doc_id, chunks)
        shutil.rmtree(work, ignore_errors=True)

    def _reindex_document(self, job_id: str, doc_id: str) -> None:
        self._step(job_id, "chunk", "running")
        with self.make_session() as session:
            chunks = self.document_chunks(session, session.get(Document, doc_id))
        self._step(job_id, "chunk", "succeeded", len(chunks), len(chunks))
        self._replace_in_live(job_id, doc_id, chunks)

    def _remove(self, job_id: str, doc_id: str) -> None:
        self._step(job_id, "remove", "running")
        with self.lock, self.make_session() as session:
            doc = session.get(Document, doc_id)
            doc.removed, doc.index_state = True, "not_indexed"
            build = self.active_build(session)
            if build is not None:
                session.execute(delete(Chunk).where(Chunk.build_id == build.build_id,
                                                    Chunk.doc_id == doc_id))
                self.index.delete_document(build.collection, doc_id)
            session.commit()
        self.load()
        self._step(job_id, "remove", "succeeded", 1, 1)

    def _reindex_corpus(self, job_id: str, created_by: str | None) -> None:
        """Every document re-chunked from its stored pages into a new build. Vectors of chunks
        whose text is unchanged are reused from the active build."""
        self._step(job_id, "chunk", "running")
        with self.make_session() as session:
            docs = session.scalars(select(Document).where(Document.removed.is_(False))
                                   .order_by(Document.doc_id)).all()
            rows = [r for d in docs for r in self.document_chunks(session, d)]
            build = self.active_build(session)
            reuse = {}
            if build is not None:
                reuse = {(c.doc_id, c.page, c.chunk, c.text): None for c in session.scalars(
                    select(Chunk).where(Chunk.build_id == build.build_id)).all()}
        self._step(job_id, "chunk", "succeeded", len(rows), len(rows))
        self._step(job_id, "embed", "running", 0, len(rows))
        vectors = np.zeros((len(rows), PINNED.vector_dim), dtype=np.float32)
        todo = list(range(len(rows)))
        if build is not None:
            def key(r: dict) -> tuple[str, int, int]:
                return (r["doc_id"], r["page"], r["chunk"])

            same = [i for i, r in enumerate(rows) if (*key(r), r["text"]) in reuse]
            found = self.index.vectors(build.collection, [key(rows[i]) for i in same])
            reused = set()
            for i in same:
                if key(rows[i]) in found:
                    vectors[i] = found[key(rows[i])]
                    reused.add(i)
            todo = [i for i in range(len(rows)) if i not in reused]
        if todo:
            fresh = self.embed_chunks([rows[i]["text"] for i in todo],
                                      lambda d: self._step(job_id, "embed", "running", d))
            for i, vector in zip(todo, fresh, strict=True):
                vectors[i] = vector
        self._step(job_id, "embed", "succeeded", len(rows), len(rows))
        self._step(job_id, "index", "running", 0, len(rows))
        build_id = self.create_build(rows, vectors, created_by,
                                     f"reused {len(rows) - len(todo)}, embedded {len(todo)}",
                                     activate=False)
        with self.make_session() as session:
            session.get(Job, job_id).target_build = build_id
            session.commit()
        self._step(job_id, "index", "succeeded", len(rows), len(rows))
        self._step(job_id, "switch_build", "running")
        self.activate(build_id)
        self._step(job_id, "switch_build", "succeeded", 1, 1)

    # --- the extraction review queue ----------------------------------------------------------
    def decide_extraction(self, item_id: str, decision: str, user: str | None,
                          note: str = "") -> str:
        """Apply an admin's decision to the page, then re-chunk its document in the live build.
        Returns the job id."""
        if decision not in ("accept_repaired", "accept_ocr", "exclude"):
            raise ValueError(f"unknown decision {decision}")
        with self.make_session() as session:
            item = session.get(ExtractionItem, item_id)
            if item is None:
                raise NotFound(item_id)
            if decision == "accept_ocr" and not item.ocr_text.strip():
                raise ValueError("no OCR text for this page")
            page = session.get(Page, (item.doc_id, item.page))
            if decision == "accept_repaired":
                page.text, page.md, page.text_source = (item.repaired_text, item.repaired_md,
                                                        "repaired")
                page.excluded = False
            elif decision == "accept_ocr":
                page.text, page.md, page.text_source = item.ocr_text, item.ocr_md, "ocr"
                page.excluded = False
            else:
                page.excluded = True
            item.status, item.decision, item.note = "decided", decision, note
            item.decided_by, item.decided_at = user, now()
            doc = session.get(Document, item.doc_id)
            pending = session.scalar(select(func.count()).select_from(ExtractionItem).where(
                ExtractionItem.doc_id == item.doc_id, ExtractionItem.status == "pending"))
            if not pending:
                doc.flags = sorted(set(doc.flags or []) - {"extraction_review_pending"})
            session.commit()
            doc_id = item.doc_id
        return self.submit("rechunk_page", doc_id, user)

    # --- uploads --------------------------------------------------------------------------
    def store_upload(self, data: bytes, filename: str) -> tuple[str, Path, str]:
        """Keep an uploaded file under the work folder; returns its doc_id, path and SHA-256."""
        checksum = hashlib.sha256(data).hexdigest()
        safe = Path(filename).name.replace("..", "_") or "upload"
        relative = f"uploads/{checksum[:12]}/{safe}"
        path = self.endpoints.work_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return make_doc_id(relative), path, checksum

    def new_document(self, session: Session, doc_id: str, path: Path, checksum: str,
                     filename: str, metadata: dict, user: str | None) -> Document:
        suffix = path.suffix.lower().lstrip(".")
        twins = session.scalars(select(Document).where(Document.checksum_sha256 == checksum,
                                                       Document.doc_id != doc_id)).all()
        flags = ["duplicate_checksum"] if twins else []
        for twin in twins:
            twin.flags = sorted(set(twin.flags or []) | {"duplicate_checksum"})
        doc = session.get(Document, doc_id) or Document(doc_id=doc_id)
        doc.title = metadata.get("title") or doc_title(filename)
        doc.file_name = doc.original_filename = Path(filename).name
        doc.relative_path = str(path.relative_to(self.endpoints.work_dir).as_posix())
        doc.file_format = suffix
        doc.paged = suffix == "pdf"
        doc.source_path, doc.checksum_sha256 = str(path), checksum
        doc.file_size_bytes = path.stat().st_size
        doc.flags, doc.index_state, doc.removed = flags, "pending", False
        doc.updated_by, doc.updated_at = user, now()
        apply_metadata(doc, metadata)
        if twins:
            doc.status, doc.duplicate_of = "duplicate", twins[0].doc_id
        if doc.language is None:
            doc.language = "unknown"
        session.add(doc)
        return doc


EDITABLE = ("title", "code", "doc_type", "language", "sector", "department", "section",
            "version", "issue_date", "access_level", "owner_department", "status",
            "superseded_by", "duplicate_of", "visible_to_all", "edition_of")


def apply_metadata(doc: Document, metadata: dict) -> None:
    for name in EDITABLE:
        if name in metadata:
            setattr(doc, name, metadata[name])
    if "group_ids" in metadata:
        doc.groups = sorted(set(metadata["group_ids"] or []))
    if doc.status != "superseded":
        doc.superseded_by = None
    if doc.status != "duplicate":
        doc.duplicate_of = None
