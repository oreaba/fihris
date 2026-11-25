"""Fihris: retrieval, access filtering, builds, the library, ingestion and the extraction queue.
Model servers are replaced by deterministic fakes. The engine tests run on SQLite with the NumPy
store, and again on PostgreSQL with pgvector when ``FIHRIS_TEST_PG_URL`` names a database the
tests may wipe (its ``fihris`` schema is dropped before each test)."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from fihris.chunks import Access, ChunkSet, unit_language
from fihris.config import EVERYONE, PINNED, Endpoints
from fihris.db import Document, ExtractionItem, Page, create_schema, engine_for, sessions
from fihris.engine import Engine
from fihris.ids import make_doc_id
from fihris.library import NotFound, Query, detail, locate, page_view, search, utf16
from fihris.principal import Principal
from fihris.retrieve import Retriever, fit
from fihris.units import Unit
from fihris.vectors import (
    MemoryIndex,
    PgVectorIndex,
    Point,
    collection_name,
    open_index,
    same_ranking,
)

DIM = PINNED.vector_dim


def vector(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    def embed(self, texts, truncate=None):
        return np.stack([vector(t.removeprefix(PINNED.query_instruction)) for t in texts])


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score(self, pairs):
        self.calls.append(list(pairs))
        return [int(hashlib.sha256((q + t).encode()).hexdigest()[:6], 16) / 0xFFFFFF
                for q, t in pairs]


class FakeTranslator:
    def translate(self, text, source_lang):
        return ("ترجمة " if source_lang == "en" else "translated ") + text


TEXTS = [
    ("d1", 1, 0, "Document: Leave Policy\nSection: Annual leave\nStaff get annual leave of 30 "
                 "working days approved by the line manager."),
    ("d1", 1, 1, "Document: Leave Policy\nSection: Sick leave\nSick leave needs a medical "
                 "certificate after two days."),
    ("d2", 1, 0, "Document: سياسة الإجازات\nالإجازة السنوية للموظف ثلاثون يوم عمل بموافقة "
                 "المدير المباشر."),
    ("d2", 2, 0, "Document: سياسة الإجازات\nالإجازة المرضية تتطلب شهادة طبية بعد يومين."),
    ("d3", 1, 0, "Document: Data Request Procedure\nA data request is approved by the section "
                 "head within five working days."),
    ("d4", 1, 0, "Document: Secret Annex\nThe annual leave of directors is approved by the "
                 "director general."),
]


def units():
    return [Unit(d, p, "", t, c) for d, p, c, t in TEXTS]


def memory_index(access):
    index = MemoryIndex()
    index.create("c", [Point(u.doc_id, u.page, u.chunk, vector(u.text), unit_language(u.text),
                             access[u.doc_id].groups, access[u.doc_id].searchable)
                       for u in units()])
    index.activate("c")
    return index


ACCESS = {"d1": Access((EVERYONE,)), "d2": Access((EVERYONE,)), "d3": Access(("hr",)),
          "d4": Access(("board",))}


@pytest.mark.parametrize(("question", "lang"), [
    ("How many days of annual leave do staff get?", "en"),
    ("كم مدة الإجازة السنوية للموظف؟", "ar")])
def test_retrieval_finds_both_editions_across_languages(question, lang):
    access = {d: Access((EVERYONE,)) for d in ACCESS}
    chunks = ChunkSet(units(), access)
    scorer = FakeScorer()
    result = Retriever(chunks, memory_index(access), FakeEmbedder(), scorer,
                       FakeTranslator()).retrieve(question, Principal.unrestricted())
    assert result.question_language == lang
    assert [s.n for s in result.sources] == list(range(1, len(result.sources) + 1))
    assert {"d1", "d2"} <= {h["doc_id"] for h in result.ranked}
    translation = FakeTranslator().translate(question, lang)
    for q, text in scorer.calls[0]:
        unit_lang = "ar" if unit_language(text) == "ar" else "en"
        assert q == (question if unit_lang == lang else translation)


@pytest.mark.parametrize(("groups", "pilot", "allowed"), [
    ((), True, {"d1", "d2"}),
    (("hr",), True, {"d1", "d2"}),
    (("hr",), False, {"d1", "d2", "d3"}),
    (("hr", "board"), False, {"d1", "d2", "d3", "d4"}),
])
def test_no_leak_through_keyword_dense_or_rerank(groups, pilot, allowed):
    chunks = ChunkSet(units(), ACCESS)
    scorer = FakeScorer()
    retriever = Retriever(chunks, memory_index(ACCESS), FakeEmbedder(), scorer, FakeTranslator())
    principal = Principal.for_groups(groups, pilot)
    result = retriever.retrieve("annual leave approved", principal)
    assert {h["doc_id"] for h in result.ranked} <= allowed
    assert {s.doc_id for s in result.sources} <= allowed
    scored_texts = {t for call in scorer.calls for _q, t in call}
    for d, _p, _c, t in TEXTS:
        if d not in allowed:
            assert t not in scored_texts


def test_unsearchable_documents_are_never_returned():
    access = dict(ACCESS, d1=Access((EVERYONE,), searchable=False))
    retriever = Retriever(ChunkSet(units(), access), memory_index(access), FakeEmbedder(),
                          FakeScorer(), FakeTranslator())
    result = retriever.retrieve("annual leave", Principal.for_groups(["hr", "board"], False))
    assert "d1" not in {h["doc_id"] for h in result.ranked}


def test_keyword_top_list_is_filled_after_filtering():
    chunks = ChunkSet(units(), ACCESS)
    mask = chunks.mask(Principal.for_groups([], True))
    hits = chunks.keyword("leave", mask, depth=3)
    assert len(hits) == 3 and {h["doc_id"] for h in hits} <= {"d1", "d2"}


def test_fit_keeps_whole_chunks_within_budget():
    hits = [{"text": "a" * 5000}, {"text": "b" * 3000}, {"text": "c" * 10}]
    assert len(fit(hits, 8000)) == 2


def test_principal_cannot_widen_itself():
    assert Principal.for_groups(["board"], pilot_mode=True).allowed == frozenset({EVERYONE})
    assert EVERYONE in Principal.for_groups([], pilot_mode=False).allowed
    assert not Principal.for_groups(["hr"], False).may_read(["board"])


# --- the engine and the library ---------------------------------------------------------------

def fresh_postgres():
    url = os.environ.get("FIHRIS_TEST_PG_URL")
    if not url:
        pytest.skip("FIHRIS_TEST_PG_URL is not set")
    db = engine_for(url)
    with db.begin() as conn:
        conn.exec_driver_sql("DROP SCHEMA IF EXISTS fihris CASCADE")
    return db


@pytest.fixture(params=["memory", "pgvector"])
def engine(request, tmp_path):
    if request.param == "pgvector":
        db = fresh_postgres()
        create_schema(db)
        index = PgVectorIndex(db)
    else:
        db = engine_for(f"sqlite:///{tmp_path / 'f.db'}")
        create_schema(db)
        index = MemoryIndex()
    make = sessions(db)
    with make() as session:
        for doc_id, title, lang, groups, everyone in [
                ("d1", "Leave Policy", "en", [], True), ("d2", "سياسة الإجازات", "ar", [], True),
                ("d3", "Data Request Procedure", "en", ["hr"], False),
                ("d4", "Secret Annex", "en", ["board"], False)]:
            session.add(Document(doc_id=doc_id, title=title, language=lang, groups=groups,
                                 visible_to_all=everyone, code="POL-1" if doc_id in ("d1", "d2")
                                 else f"X-{doc_id}", file_format="pdf", paged=True,
                                 page_count=2))
        for d, p, _c, t in TEXTS:
            if session.get(Page, (d, p)) is None:
                session.add(Page(doc_id=d, page=p, status="ok", text=t.split("\n")[-1], md=""))
        session.commit()
    eng = Engine(Endpoints(work_dir=tmp_path / "work"), make, index=index,
                 embedder=FakeEmbedder(), scorer=FakeScorer(), translator=FakeTranslator())
    rows = [{"doc_id": d, "page": p, "chunk": c, "text": t, "headings": []}
            for d, p, c, t in TEXTS]
    eng.create_build(rows, FakeEmbedder().embed([r["text"] for r in rows]), None, "test",
                     activate=True, build_id="build-test")
    yield eng
    eng.jobs.shutdown(wait=True)
    db.dispose()


def test_text_with_nul_round_trips_and_keeps_fingerprint(tmp_path):
    from fihris.db import Chunk

    db = engine_for(f"sqlite:///{tmp_path / 'n.db'}")
    create_schema(db)
    make = sessions(db)
    text = "Document: X\na\x00b \ue000 \ue000\ue002 c\x00"
    with make() as session:
        session.add(Chunk(build_id="b", doc_id="d", page=1, chunk=0, ordinal=0, text=text,
                          headings=[], lang="en"))
        session.commit()
    with make() as session:
        assert session.get(Chunk, ("b", "d", 1, 0)).text == text
        raw = session.connection().exec_driver_sql("select text from chunks").scalar()
        assert "\x00" not in raw


def test_build_is_active_and_searchable(engine):
    assert engine.build.build_id == "build-test"
    assert engine.index.count() == engine.index.count(collection_name("build-test")) == len(TEXTS)
    result = engine.retrieve("annual leave", Principal.for_groups(["hr"], False))
    assert result.sources and {s.doc_id for s in result.sources} <= {"d1", "d2", "d3"}


def test_library_hides_other_groups_as_missing(engine):
    with engine.make_session() as session:
        user = Principal.for_groups(["hr"], False)
        listed = {e["doc_id"] for e in search(session, user, Query())["items"]}
        assert listed == {"d1", "d2", "d3"}
        with pytest.raises(NotFound):
            detail(session, user, "d4", downloads=True)
        with pytest.raises(NotFound):
            page_view(session, user, "d4", 1)
        pilot = {e["doc_id"] for e in search(session, Principal.for_groups(["hr"], True),
                                             Query())["items"]}
        assert pilot == {"d1", "d2"}
        d1 = detail(session, user, "d1", downloads=True)
        assert [e["doc_id"] for e in d1["editions"]] == ["d2"]


def test_access_change_reaches_mask_and_store(engine):
    with engine.make_session() as session:
        session.get(Document, "d3").visible_to_all = True
        session.commit()
    engine.refresh_access("d3")
    result = engine.retrieve("data request approved", Principal.for_groups([], True))
    assert "d3" in {h["doc_id"] for h in result.ranked}


def test_page_view_highlights_in_utf16(engine):
    with engine.make_session() as session:
        view = page_view(session, Principal.unrestricted(), "d2", 1,
                         (0, TEXTS[2][3]))
    assert view["highlights"]
    h = view["highlights"][0]
    assert view["text"].encode("utf-16-le")[2 * h["start"]:2 * h["end"]].decode("utf-16-le")
    assert utf16("a😀b", 2) == 3


def test_locate_returns_none_when_absent():
    assert locate("some page text", "Document: X\nnothing like it here at all") is None


def test_extraction_decision_rechunks_live_build(engine):
    with engine.make_session() as session:
        session.add(ExtractionItem(item_id="ext_1", doc_id="d3", page=1, repaired_text="garbled",
                                   ocr_text="A data request needs the section head approval.",
                                   ocr_passed_check=True))
        session.commit()
    job_id = engine.decide_extraction("ext_1", "accept_ocr", "u1")
    engine.jobs.shutdown(wait=True)
    with engine.make_session() as session:
        from fihris.db import Job

        assert session.get(Job, job_id).state == "succeeded"
        assert session.get(Page, ("d3", 1)).text_source == "ocr"
    assert any("section head approval" in t for t in engine.chunks.texts.values())


def test_remove_drops_document_everywhere(engine):
    job_id = engine.submit("remove", "d1", "u1")
    engine.jobs.shutdown(wait=True)
    with engine.make_session() as session:
        from fihris.db import Job

        assert session.get(Job, job_id).state == "succeeded"
        assert session.get(Document, "d1").removed
    assert "d1" not in {u.doc_id for u in engine.chunks.units}
    result = engine.retrieve("annual leave", Principal.unrestricted())
    assert "d1" not in {h["doc_id"] for h in result.ranked}


def test_ingest_office_file_end_to_end(engine, tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Travel Procedure", level=1)
    d.add_paragraph("Business travel requests are approved by the department director at least "
                    "ten working days before departure. " * 4)
    path = tmp_path / "TRV-SOP-01 Travel Procedure.docx"
    d.save(path)
    with engine.make_session() as session:
        doc_id, stored, checksum = engine.store_upload(path.read_bytes(), path.name)
        engine.new_document(session, doc_id, stored, checksum, path.name,
                            {"visible_to_all": True, "code": "TRV-SOP-01"}, "u1")
        session.commit()
    job_id = engine.submit("ingest", doc_id, "u1", path=str(stored))
    engine.jobs.shutdown(wait=True)
    with engine.make_session() as session:
        from fihris.db import Job

        job = session.get(Job, job_id)
        assert job.state == "succeeded", job.errors
        doc = session.get(Document, doc_id)
        assert doc.index_state == "indexed" and doc.language == "en" and not doc.paged
        assert session.get(Page, (doc_id, 1)).status == "ok"
    result = engine.retrieve("Who approves business travel requests?",
                             Principal.for_groups([], True))
    assert doc_id in {s.doc_id for s in result.sources}


def test_duplicate_upload_is_flagged(engine, tmp_path):
    data = b"same bytes"
    with engine.make_session() as session:
        a, pa, ca = engine.store_upload(data, "a.docx")
        engine.new_document(session, a, pa, ca, "a.docx", {}, None)
        session.commit()
        b, pb, cb = engine.store_upload(data, "b.docx")
        doc = engine.new_document(session, b, pb, cb, "b.docx", {}, None)
        session.commit()
        assert "duplicate_checksum" in doc.flags
        assert "duplicate_checksum" in session.get(Document, a).flags


def test_reindex_corpus_reuses_unchanged_vectors(engine):
    job_id = engine.submit("reindex_corpus", None, "u1")
    engine.jobs.shutdown(wait=True)
    with engine.make_session() as session:
        from fihris.db import Build, Job

        job = session.get(Job, job_id)
        assert job.state == "succeeded", job.errors
        assert session.get(Build, job.target_build).state == "active"
    assert engine.build.build_id == job.target_build


def test_first_ingest_on_a_fresh_database_creates_the_build(tmp_path):
    docx = pytest.importorskip("docx")
    db = engine_for(f"sqlite:///{tmp_path / 'fresh.db'}")
    create_schema(db)
    eng = Engine(Endpoints(work_dir=tmp_path / "work"), sessions(db), index=MemoryIndex(),
                 embedder=FakeEmbedder(), scorer=FakeScorer(), translator=FakeTranslator())
    eng.load()
    assert eng.build is None
    d = docx.Document()
    d.add_heading("Expense Claims", level=1)
    d.add_paragraph("Expense claims are submitted within thirty days with receipts. " * 4)
    path = tmp_path / "expenses.docx"
    d.save(path)
    with eng.make_session() as session:
        doc_id, stored, checksum = eng.store_upload(path.read_bytes(), path.name)
        eng.new_document(session, doc_id, stored, checksum, path.name,
                         {"visible_to_all": True}, None)
        session.commit()
    job_id = eng.submit("ingest", doc_id, None, path=str(stored))
    eng.jobs.shutdown(wait=True)
    with eng.make_session() as session:
        from fihris.db import Job

        assert session.get(Job, job_id).state == "succeeded"
    assert eng.build is not None
    result = eng.retrieve("When are expense claims submitted?", Principal.for_groups([], True))
    assert doc_id in {s.doc_id for s in result.sources}


def test_upsert_documents_derives_ids_and_reads_files(tmp_path):
    root = tmp_path / "documents"
    (root / "hr").mkdir(parents=True)
    (root / "hr" / "Leave_Policy.pdf").write_bytes(b"%PDF-1.7 test")
    db = engine_for(f"sqlite:///{tmp_path / 'u.db'}")
    create_schema(db)
    eng = Engine(Endpoints(work_dir=tmp_path / "work", corpus_root=root), sessions(db),
                 index=MemoryIndex(), embedder=FakeEmbedder(), scorer=FakeScorer(),
                 translator=FakeTranslator())
    count = eng.upsert_documents([
        {"relative_path": "hr/Leave_Policy.pdf", "language": "en", "department": "HR",
         "unknown_field": "ignored"},
        {"doc_id": "doc_given", "relative_path": "missing.docx", "title": "Given"}])
    assert count == 2
    with eng.make_session() as session:
        doc = session.get(Document, make_doc_id("hr/Leave_Policy.pdf"))
        assert doc.title == "Leave Policy" and doc.department == "HR"
        assert doc.file_format == "pdf" and doc.file_size_bytes == 13
        assert len(doc.checksum_sha256) == 64 and doc.index_state == "pending"
        given = session.get(Document, "doc_given")
        assert given.title == "Given" and given.checksum_sha256 in ("", None)


def test_doc_ids_are_stable_across_separators_and_unicode_forms():
    import unicodedata

    name = "سياسة الإجازات.pdf"
    assert make_doc_id("hr\\" + name) == make_doc_id("hr/" + unicodedata.normalize("NFD", name))


# --- the vector stores --------------------------------------------------------------------------

def test_same_ranking_allows_only_near_tie_swaps():
    scores = {"a": 0.9, "b": 0.8, "c": 0.8 + 5e-6, "d": 0.7}
    ref = [(k, scores[k]) for k in "abcd"]
    assert same_ranking(ref, [(k, 0) for k in "acbd"], scores.get)
    assert not same_ranking(ref, [(k, 0) for k in "abdc"], scores.get)
    assert not same_ranking(ref, [(k, 0) for k in "abbd"], scores.get)
    assert not same_ranking(ref, [(k, 0) for k in "abc"], scores.get)


def test_translator_is_ct2_by_default_and_names_what_is_missing(tmp_path, monkeypatch):
    from fihris import clients
    from fihris.config import TRANSLATOR_SETS

    assert (Endpoints().translator, Endpoints().translator_models) == ("ct2", "opus-mt")
    assert all(set(s) == {"en", "ar"} and s["en"]["token"] == ">>ara<<"
               for s in TRANSLATOR_SETS.values())
    with pytest.raises(FileNotFoundError, match="convert-translators"):
        clients.CT2Translator(Endpoints(translator_dir=tmp_path / "none"))

    def never_downloaded(repo, revision=None, allow_patterns=None):
        raise OSError(repo)

    monkeypatch.setattr(clients, "snapshot", never_downloaded)
    with pytest.raises(FileNotFoundError, match="download-translators"):
        clients.CT2Translator(Endpoints(translator_models="tc-big"))
    with pytest.raises(ValueError, match="FIHRIS_TRANSLATOR_MODELS"):
        clients.CT2Translator(Endpoints(translator_models="nllb"))
    with pytest.raises(ValueError, match="ct2, torch or server"):
        clients.open_translator(Endpoints(translator="marian"))


def test_server_translator_sends_the_cards_prompt_greedy(monkeypatch):
    from fihris import clients

    sent = []

    def post(url, body, timeout):
        sent.append((url, body))
        return {"choices": [{"message": {"content": "  كم يوما؟ \n"}}]}

    monkeypatch.setattr(clients, "_post", post)
    translator = clients.open_translator(Endpoints(translator="server",
                                                   translate_url="http://t:1"))
    assert translator.translate("How many days?", "en") == "كم يوما؟"
    url, body = sent[0]
    assert url == "http://t:1/v1/chat/completions" and body["model"] == "hy-mt3-1.8b"
    assert body["temperature"] == 0 and body["max_tokens"] == 256
    prompt = body["messages"][0]["content"]
    assert prompt.startswith("Translate the following text into Arabic.")
    assert prompt.endswith("\n\nHow many days?")
    translator.translate("كم يوما؟", "ar")
    assert "into English." in sent[1][1]["messages"][0]["content"]


def test_pgvector_needs_postgresql_and_a_known_store(tmp_path):
    db = engine_for(f"sqlite:///{tmp_path / 's.db'}")
    with pytest.raises(ValueError, match="PostgreSQL"):
        open_index(Endpoints(vector_store="pgvector"), db)
    with pytest.raises(ValueError, match="qdrant or pgvector"):
        open_index(Endpoints(vector_store="faiss"), db)


def test_pgvector_matches_the_numpy_reference_through_updates():
    db = fresh_postgres()
    rng = np.random.default_rng(7)
    groups = [("all",), ("hr",), ("board",), ("hr", "board")]
    points = []
    for d in range(30):
        g = groups[d % len(groups)]
        for c in range(12):
            v = rng.standard_normal(DIM).astype(np.float32)
            points.append(Point(f"doc{d:02d}", 1 + c // 4, c % 4, v / np.linalg.norm(v),
                                "en" if d % 2 else "ar", g, searchable=d % 7 != 3))
    queries = rng.standard_normal((6, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    principals = [Principal.unrestricted(), Principal.for_groups(["hr"], False),
                  Principal.for_groups(["hr"], True), Principal.for_groups(["nobody"], False)]
    pg, ref = PgVectorIndex(db), MemoryIndex()

    def same(k=100):
        for principal in principals:
            for a, b in zip(ref.search(queries, principal, k), pg.search(queries, principal, k),
                            strict=True):
                assert [key for key, _ in a] == [key for key, _ in b]
                assert np.allclose([s for _, s in a], [s for _, s in b], atol=1e-5)

    for store in (ref, pg):
        store.create("fihris_build_a", points)
        store.activate("fihris_build_a")
        with pytest.raises(ValueError, match="already exists"):
            store.create("fihris_build_a", [])
    same()
    same(k=5)
    keys = [p.key for p in points[:40]]
    assert all(np.array_equal(pg.vectors("fihris_build_a", keys)[p.key], p.vector)
               for p in points[:40])

    replacement = [Point(p.doc_id, p.page, p.chunk, -p.vector, p.lang, p.groups)
                   for p in points if p.doc_id == "doc05"]
    for store in (ref, pg):
        store.delete_document("fihris_build_a", "doc04")
        store.set_access("fihris_build_a", "doc01", ("all",), True)
        store.set_access("fihris_build_a", "doc02", ("hr",), False)
        store.upsert("fihris_build_a", replacement)
    same()
    assert pg.count() == ref.count() == len(points) - 12

    for store in (ref, pg):
        store.create("fihris_build_b", points[:24])
        store.activate("fihris_build_b")
    same()
    assert pg.count() == 24 and pg.count("fihris_build_a") == len(points) - 12
    pg.drop("fihris_build_a")
    assert pg.count("fihris_build_a") == 0 and pg.healthy()
    db.dispose()
