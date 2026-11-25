"""The ``fihris`` command line: set up, ingest files, search, manage builds and access. Output is
content-free unless ``--text`` is asked for.

    fihris init-db
    fihris ingest documents/ --visible-to-all
    fihris search "How many days of annual leave?" --text
    fihris status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sqlalchemy import func, select

from fihris.config import PINNED, Endpoints
from fihris.db import Chunk, Document, Job, Page, create_schema, engine_for, sessions
from fihris.principal import Principal


def _engine(args, **clients):
    from fihris.engine import Engine

    endpoints = Endpoints.from_env()
    db = engine_for(args.database or endpoints.database_url)
    create_schema(db)
    engine = Engine(endpoints, sessions(db), **clients)
    engine.load()
    return engine


def _print(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=1, default=str))


def cmd_init_db(args) -> int:
    from fihris.vectors import open_index

    endpoints = Endpoints.from_env()
    db = engine_for(args.database or endpoints.database_url)
    create_schema(db)
    if endpoints.vector_store == "pgvector":
        open_index(endpoints, db)
    _print({"database": db.url.render_as_string(hide_password=True),
            "vector_store": endpoints.vector_store, "created": True})
    return 0


def cmd_import_pages(args) -> int:
    """Documents and their page text extracted elsewhere, chunked and embedded as one new build:
    documents from a JSON-lines catalogue (``Document`` columns per row, at least
    ``relative_path`` or ``doc_id``), pages from a page index."""
    from fihris.chunking import load_pages
    from fihris.chunks import chunk_corpus

    engine = _engine(args)
    with args.catalogue.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report: dict = {"documents": engine.upsert_documents(rows),
                    "pages": engine.import_pages(args.pages)}
    with engine.make_session() as session:
        titles = {d.doc_id: d.title for d in session.scalars(select(Document)).all()}
    chunks = chunk_corpus(load_pages(args.pages), titles)
    report["chunks"] = len(chunks)
    vectors = engine.embed_chunks([r["text"] for r in chunks])
    report["build"] = engine.create_build(chunks, vectors, None, "embedded", args.activate,
                                          args.build_id)
    with engine.make_session() as session:
        indexed = {r["doc_id"] for r in chunks}
        for doc in session.scalars(select(Document)).all():
            doc.index_state = "indexed" if doc.doc_id in indexed else "not_indexed"
        session.commit()
    engine.load()
    _print(report)
    return 0


def cmd_status(args) -> int:
    engine = _engine(args)
    with engine.make_session() as session:
        def count(model, *where):
            return session.scalar(select(func.count()).select_from(model).where(*where))

        build = engine.active_build(session)
        _print({"active_build": build.build_id if build else None,
                "build_chunks": count(Chunk, Chunk.build_id == build.build_id) if build else 0,
                "documents": count(Document), "removed": count(Document, Document.removed),
                "visible_to_all": count(Document, Document.visible_to_all),
                "pages": count(Page), "jobs_running": count(Job, Job.state == "running"),
                "config_sha256": PINNED.sha256})
    return 0


def cmd_builds(args) -> int:
    engine = _engine(args)
    _print({"builds": [{"build_id": b.build_id, "state": b.state, "chunks": b.chunk_count,
                        "documents": b.document_count, "units_sha256": b.units_sha256,
                        "vectors": b.vectors_source, "created_at": b.created_at}
                       for b in engine.builds()]})
    return 0


def cmd_activate(args) -> int:
    engine = _engine(args)
    engine.activate(args.build_id)
    _print({"active_build": args.build_id})
    return 0


def cmd_search(args) -> int:
    engine = _engine(args)
    principal = (Principal.for_groups(args.groups, args.pilot) if args.groups or args.pilot
                 else Principal.unrestricted())
    result = engine.retrieve(args.question, principal)
    _print({"question_language": result.question_language, "timings": result.timings,
            "sources": [{"n": s.n, "doc_id": s.doc_id, "page": s.page, "chunk": s.chunk,
                         "score": s.score, "language": s.language,
                         **({"text": s.text} if args.text else {})} for s in result.sources]})
    return 0


def cmd_access(args) -> int:
    """Set documents' groups or visibility, by id or by access-level label."""
    engine = _engine(args)
    with engine.make_session() as session:
        query = select(Document)
        if args.doc:
            query = query.where(Document.doc_id.in_(args.doc))
        if args.access_level:
            query = query.where(Document.access_level.in_(args.access_level))
        docs = session.scalars(query).all()
        for doc in docs:
            if args.visible_to_all is not None:
                doc.visible_to_all = args.visible_to_all == "yes"
            if args.groups is not None:
                doc.groups = sorted(set(args.groups))
        session.commit()
        ids = [d.doc_id for d in docs]
    engine.refresh_access()
    if engine.build is not None:
        for doc_id in ids:
            engine.refresh_access(doc_id)
    _print({"documents_changed": len(ids)})
    return 0


SUFFIXES = (".pdf", ".docx", ".doc", ".xlsx", ".pptx", ".html", ".htm", ".md", ".markdown",
            ".csv", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp")


def _files(paths: list[Path]) -> list[Path]:
    out = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*")
                              if p.is_file() and p.suffix.lower() in SUFFIXES))
        else:
            out.append(path)
    return out


def cmd_ingest(args) -> int:
    """Add files, or every supported file in folders, one job each, and wait for them."""
    engine = _engine(args)
    files = _files(args.paths)
    if args.title and len(files) != 1:
        print("--title needs exactly one file", file=sys.stderr)
        return 2
    metadata = {"visible_to_all": True} if args.visible_to_all else {}
    if args.groups:
        metadata["group_ids"] = args.groups
    jobs = []
    for file in files:
        with engine.make_session() as session:
            doc_id, path, checksum = engine.store_upload(file.read_bytes(), file.name)
            engine.new_document(session, doc_id, path, checksum, file.name,
                                {**metadata, **({"title": args.title} if args.title else {})},
                                None)
            session.commit()
        jobs.append((engine.submit("ingest", doc_id, None, path=str(path)), doc_id, file))
    results = []
    for job_id, doc_id, file in jobs:
        while True:
            with engine.make_session() as session:
                job = session.get(Job, job_id)
                if job.state in ("succeeded", "failed", "cancelled"):
                    results.append({"file": file.name, "doc_id": doc_id, "state": job.state,
                                    "pages_sent_to_review": job.pages_sent_to_review,
                                    "errors": job.errors})
                    break
            time.sleep(1)
    _print({"files": len(files), "results": results})
    return 0 if all(r["state"] == "succeeded" for r in results) else 1


def cmd_reindex(args) -> int:
    engine = _engine(args)
    job_id = engine.submit("reindex_corpus", None, None)
    while True:
        with engine.make_session() as session:
            job = session.get(Job, job_id)
            if job.state in ("succeeded", "failed", "cancelled"):
                _print({"job_id": job_id, "state": job.state, "target_build": job.target_build,
                        "errors": job.errors})
                return 0 if job.state == "succeeded" else 1
        time.sleep(2)


def _percentiles(ms: list[float]) -> dict:
    import numpy as np

    return {"p50": round(float(np.percentile(ms, 50)), 1),
            "p95": round(float(np.percentile(ms, 95)), 1)} if ms else {}


def cmd_compare_stores(args) -> int:
    """The live build's dense search in Qdrant and in pgvector, question by question: lists
    compared under EXP-024's matching rule, and the time of each search. ``--copy`` first copies
    the live build's vectors from Qdrant into pgvector, unchanged, under the same collection."""
    import numpy as np

    from fihris.vectors import open_index, same_ranking

    engine = _engine(args)
    if engine.build is None:
        print("no active build", file=sys.stderr)
        return 2
    db, build = engine.make_session.kw["bind"], engine.build
    collection = build.collection
    qdrant = open_index(engine.endpoints, db, "qdrant")
    pg = open_index(engine.endpoints, engine_for(args.pgvector_url) if args.pgvector_url else db,
                    "pgvector")
    with engine.make_session() as session:
        rows = [{"doc_id": c.doc_id, "page": c.page, "chunk": c.chunk, "text": c.text}
                for c in session.scalars(select(Chunk).where(Chunk.build_id == build.build_id)
                                         .order_by(Chunk.ordinal)).all()]
        access = engine.access_map(session)
    keys = [(r["doc_id"], r["page"], r["chunk"]) for r in rows]
    stored = qdrant.vectors(collection, keys)
    if len(stored) != len(keys):
        print(f"Qdrant holds {len(stored)} of the build's {len(keys)} vectors", file=sys.stderr)
        return 2
    report: dict = {"build": build.build_id, "chunks": len(keys)}
    if args.copy:
        pg.drop(collection)
        pg.create(collection, engine.points(rows, np.stack([stored[k] for k in keys]), access))
        pg.activate(collection)
        report["copied"] = len(keys)
    report["counts"] = {"qdrant": qdrant.count(), "pgvector": pg.count()}
    if report["counts"]["pgvector"] != len(keys):
        _print(report)
        print("pgvector does not hold the live build; run with --copy", file=sys.stderr)
        return 2

    with args.questions.open(encoding="utf-8") as handle:
        questions = [line.strip() for line in handle if line.strip()]
    texts = list(questions)
    if not args.no_translate:
        from fihris.retrieve import detect_language

        texts += [engine.translator.translate(q, detect_language(q)) for q in questions]
    queries = engine.embedder.embed([PINNED.query_instruction + t for t in texts])
    principal = (Principal.for_groups(args.groups, False) if args.groups
                 else Principal.unrestricted())
    qdrant.search(queries[:1], principal)
    pg.search(queries[:1], principal)
    verdicts = {"identical": 0, "near_ties_swapped": 0, "different": 0}
    times: dict[str, list[float]] = {"qdrant": [], "pgvector": []}
    different = []
    for i, query in enumerate(queries):
        lists = {}
        for name, store in (("qdrant", qdrant), ("pgvector", pg)):
            start = time.perf_counter()
            lists[name] = store.search(query[None, :], principal)[0]
            times[name].append(1000 * (time.perf_counter() - start))
        a, b = lists["qdrant"], lists["pgvector"]
        if [k for k, _ in a] == [k for k, _ in b]:
            verdicts["identical"] += 1
        elif same_ranking(a, b, lambda key, q=query: float(stored[key] @ q)):
            verdicts["near_ties_swapped"] += 1
        else:
            verdicts["different"] += 1
            different.append(i)
    report.update({"queries": len(queries), "principal_groups": sorted(args.groups) or "all",
                   **verdicts, "different_queries": different,
                   "search_ms": {name: _percentiles(ms) for name, ms in times.items()}})
    _print(report)
    return 0 if verdicts["different"] == 0 else 1


def cmd_download_translators(args) -> int:
    """A ready-made translator set into the Hugging Face cache, at its pinned revisions."""
    from fihris.clients import download_translators

    name = args.models or Endpoints.from_env().translator_models
    _print({"models": name, "downloaded": download_translators(name)})
    return 0


def cmd_convert_translators(args) -> int:
    """Both opus-mt models from the Hugging Face cache into CTranslate2's format, for
    ``FIHRIS_TRANSLATOR_DIR``. Needs torch, once."""
    from fihris.clients import convert_translators

    _print({"translators": convert_translators(Endpoints.from_env(), args.out, args.force)})
    return 0


def _translator(endpoints: Endpoints, name: str):
    """``torch``, ``converted`` (FIHRIS_TRANSLATOR_DIR, default ``translators``) or the name of a
    ready-made set."""
    from dataclasses import replace

    from fihris.clients import CT2Translator, OpusTranslator

    if name == "torch":
        return OpusTranslator(endpoints)
    if name == "converted":
        folder = endpoints.translator_dir or Path("translators")
        return CT2Translator(replace(endpoints, translator_dir=folder))
    return CT2Translator(replace(endpoints, translator_models=name, translator_dir=None))


def cmd_compare_translators(args) -> int:
    """Two translators on the same questions: identical translations, and, for those that differ,
    whether retrieval's sources change. Needs the whole pipeline unless ``--translations-only``,
    and torch only when one side is ``torch``."""
    from fihris.retrieve import Retriever, detect_language

    endpoints = Endpoints.from_env()
    with args.questions.open(encoding="utf-8") as handle:
        questions = [line.strip() for line in handle if line.strip()]
    a, b = args.a, args.b
    if a == b:
        print("--a and --b name the same translator", file=sys.stderr)
        return 2
    translators = {a: _translator(endpoints, a), b: _translator(endpoints, b)}
    times: dict[str, list[float]] = {name: [] for name in translators}
    differing = []
    for i, question in enumerate(questions):
        lang = detect_language(question)
        out = {}
        for name, translator in translators.items():
            start = time.perf_counter()
            out[name] = translator.translate(question, lang)
            times[name].append(1000 * (time.perf_counter() - start))
        if out[a] != out[b]:
            differing.append(i)
    report: dict = {"questions": len(questions),
                    "translations_identical": len(questions) - len(differing),
                    "differing_questions": differing,
                    "translate_ms": {name: _percentiles(ms) for name, ms in times.items()}}
    if differing and not args.translations_only:
        engine = _engine(args)
        if engine.chunks is None:
            print("no active build", file=sys.stderr)
            return 2
        sources = {"identical": 0, "same_sources_other_order": 0, "different": 0}
        changed = []
        for i in differing:
            lists = {}
            for name, translator in translators.items():
                retriever = Retriever(engine.chunks, engine.index, engine.embedder,
                                      engine.scorer, translator)
                result = retriever.retrieve(questions[i], Principal.unrestricted())
                lists[name] = [(s.doc_id, s.page, s.chunk) for s in result.sources]
            if lists[a] == lists[b]:
                sources["identical"] += 1
            elif sorted(lists[a]) == sorted(lists[b]):
                sources["same_sources_other_order"] += 1
            else:
                sources["different"] += 1
                changed.append(i)
        report["sources_of_differing_questions"] = sources
        report["questions_with_different_sources"] = changed
    _print(report)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fihris", description="Fihris: ingestion, storage and "
                                "retrieval of Arabic and English documents.")
    p.add_argument("--database", help="SQLAlchemy URL (default FIHRIS_DATABASE_URL)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db").set_defaults(run=cmd_init_db)
    s = sub.add_parser("import-pages", help="documents and page text extracted elsewhere, as a "
                       "new build")
    s.add_argument("--catalogue", type=Path, required=True,
                   help="JSON lines, one document per line")
    s.add_argument("--pages", type=Path, required=True, help="page index, JSON lines")
    s.add_argument("--build-id")
    s.add_argument("--activate", action="store_true")
    s.set_defaults(run=cmd_import_pages)
    sub.add_parser("status").set_defaults(run=cmd_status)
    sub.add_parser("builds").set_defaults(run=cmd_builds)
    s = sub.add_parser("activate")
    s.add_argument("build_id")
    s.set_defaults(run=cmd_activate)
    s = sub.add_parser("search")
    s.add_argument("question")
    s.add_argument("--groups", nargs="*", default=[])
    s.add_argument("--pilot", action="store_true")
    s.add_argument("--text", action="store_true", help="print the source texts")
    s.set_defaults(run=cmd_search)
    s = sub.add_parser("access", help="set groups or visibility")
    s.add_argument("--doc", nargs="*")
    s.add_argument("--access-level", nargs="*")
    s.add_argument("--visible-to-all", choices=["yes", "no"])
    s.add_argument("--groups", nargs="*")
    s.set_defaults(run=cmd_access)
    s = sub.add_parser("ingest", help="add files or folders and wait for their jobs")
    s.add_argument("paths", type=Path, nargs="+")
    s.add_argument("--title", help="for a single file")
    s.add_argument("--visible-to-all", action="store_true")
    s.add_argument("--groups", nargs="*", help="group ids that may read the documents")
    s.set_defaults(run=cmd_ingest)
    sub.add_parser("reindex", help="re-chunk every document into a new build").set_defaults(
        run=cmd_reindex)
    s = sub.add_parser("compare-stores", help="the live build's dense search in Qdrant and in "
                       "pgvector: same lists, and the time of each")
    s.add_argument("questions", type=Path, help="one question per line")
    s.add_argument("--copy", action="store_true",
                   help="first copy the live build's vectors from Qdrant into pgvector")
    s.add_argument("--groups", nargs="*", default=[],
                   help="search as a principal with these groups (default: unrestricted)")
    s.add_argument("--no-translate", action="store_true",
                   help="search the questions only, not their translations too")
    s.add_argument("--pgvector-url", help="PostgreSQL URL for the pgvector side (default: the "
                   "fihris database, which must then be PostgreSQL)")
    s.set_defaults(run=cmd_compare_stores)
    from fihris.config import TRANSLATOR_SETS

    s = sub.add_parser("download-translators", help="a ready-made translator set into the "
                       "Hugging Face cache (the only step that uses the network)")
    s.add_argument("--models", choices=sorted(TRANSLATOR_SETS),
                   help="the set (default FIHRIS_TRANSLATOR_MODELS)")
    s.set_defaults(run=cmd_download_translators)
    s = sub.add_parser("convert-translators", help="the opus-mt models into CTranslate2's "
                       "format, for FIHRIS_TRANSLATOR_DIR (needs torch, once)")
    s.add_argument("--out", type=Path, help="folder for the converted models (default "
                   "FIHRIS_TRANSLATOR_DIR)")
    s.add_argument("--force", action="store_true", help="convert again over an existing copy")
    s.set_defaults(run=cmd_convert_translators)
    s = sub.add_parser("compare-translators", help="two translators on the same questions, and "
                       "the sources retrieval returns with each")
    s.add_argument("questions", type=Path, help="one question per line")
    names = ["torch", "converted", *sorted(TRANSLATOR_SETS)]
    s.add_argument("--a", choices=names, default="opus-mt", help="default opus-mt")
    s.add_argument("--b", choices=names, default="tc-big", help="default tc-big")
    s.add_argument("--translations-only", action="store_true",
                   help="compare the translations only, without retrieval")
    s.set_defaults(run=cmd_compare_translators)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
