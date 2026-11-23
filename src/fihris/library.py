"""The library: catalogue search and facets, document detail, the page viewer and page images,
always for a principal. A document outside the principal's groups is reported as missing, the
same as one that does not exist, so its existence is not revealed."""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from fihris.config import EVERYONE
from fihris.db import Document, Page
from fihris.principal import Principal

SORTS = ("title", "code", "doc_type", "language", "department", "issue_date", "version")
FACETS = ("doc_type", "language", "sector", "department", "section", "access_level", "version")
IMAGE_SCALE = 1.5


class NotFound(LookupError):
    pass


def document_groups(doc: Document) -> list[str]:
    return sorted(set(doc.groups or []) | ({EVERYONE} if doc.visible_to_all else set()))


def visible(doc: Document, principal: Principal) -> bool:
    return doc.searchable and principal.may_read(document_groups(doc))


def visible_documents(session: Session, principal: Principal) -> list[Document]:
    rows = session.scalars(select(Document).where(Document.removed.is_(False))).all()
    return [d for d in rows if visible(d, principal)]


def get_visible(session: Session, principal: Principal, doc_id: str) -> Document:
    doc = session.get(Document, doc_id)
    if doc is None or not visible(doc, principal):
        raise NotFound(doc_id)
    return doc


def editions(session: Session, principal: Principal, doc: Document) -> list[Document]:
    if not doc.code or doc.language not in ("en", "ar"):
        return []
    rows = session.scalars(select(Document).where(Document.code == doc.code,
                                                  Document.doc_id != doc.doc_id)).all()
    return [o for o in rows if o.language in ("en", "ar") and o.language != doc.language
            and visible(o, principal)]


def entry(doc: Document, has_edition: bool) -> dict:
    return {"doc_id": doc.doc_id, "title": doc.title, "code": doc.code or None,
            "doc_type": doc.doc_type, "language": doc.language, "sector": doc.sector,
            "department": doc.department, "section": doc.section,
            "access_level": doc.access_level, "version": doc.version or None,
            "issue_date": doc.issue_date, "file_format": doc.file_format,
            "page_count": doc.page_count, "paged": doc.paged,
            "has_other_language_edition": has_edition}


@dataclass
class Query:
    q: str | None = None
    title: str | None = None
    code: str | None = None
    doc_type: tuple[str, ...] = ()
    language: tuple[str, ...] = ()
    sector: tuple[str, ...] = ()
    department: tuple[str, ...] = ()
    section: tuple[str, ...] = ()
    access_level: tuple[str, ...] = ()
    version: str | None = None
    issued_from: str | None = None
    issued_to: str | None = None
    sort: str = "title"
    order: str = "asc"
    page: int = 1
    page_size: int = 25


def _matches(doc: Document, q: Query) -> bool:
    def has(text: str | None, needle: str | None) -> bool:
        return not needle or needle.casefold() in (text or "").casefold()

    if q.q and not (has(doc.title, q.q) or has(doc.code, q.q)):
        return False
    if not (has(doc.title, q.title) and has(doc.code, q.code)):
        return False
    for name in ("doc_type", "language", "sector", "department", "section", "access_level"):
        wanted = getattr(q, name)
        if wanted and (getattr(doc, name) or "") not in wanted:
            return False
    if q.version and (doc.version or "") != q.version:
        return False
    if q.issued_from and (not doc.issue_date or doc.issue_date < q.issued_from):
        return False
    return not (q.issued_to and (not doc.issue_date or doc.issue_date > q.issued_to))


def search(session: Session, principal: Principal, q: Query) -> dict:
    docs = [d for d in visible_documents(session, principal) if _matches(d, q)]
    if q.sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}")
    present = [d for d in docs if getattr(d, q.sort)]
    missing = [d for d in docs if not getattr(d, q.sort)]
    present.sort(key=lambda d: (str(getattr(d, q.sort)).casefold(), d.title.casefold()),
                 reverse=q.order == "desc")
    ordered = present + sorted(missing, key=lambda d: d.title.casefold())
    codes = Counter((d.code, d.language) for d in docs if d.code)
    start = (q.page - 1) * q.page_size
    items = []
    for d in ordered[start:start + q.page_size]:
        other = "ar" if d.language == "en" else "en" if d.language == "ar" else None
        items.append(entry(d, bool(other and codes.get((d.code, other)))))
    return {"page": q.page, "page_size": q.page_size, "total": len(ordered), "items": items}


def facets(session: Session, principal: Principal) -> dict:
    docs = visible_documents(session, principal)
    out: dict = {}
    for name in FACETS:
        counts = Counter(getattr(d, name) for d in docs if getattr(d, name))
        out[name] = [{"value": v, "count": n} for v, n in sorted(counts.items())]
    dates = sorted(d.issue_date for d in docs if d.issue_date)
    out["issue_date_range"] = {"min": dates[0], "max": dates[-1]} if dates else {}
    return out


def detail(session: Session, principal: Principal, doc_id: str, downloads: bool) -> dict:
    doc = get_visible(session, principal, doc_id)
    others = editions(session, principal, doc)
    return {**entry(doc, bool(others)),
            "editions": [{"doc_id": o.doc_id, "title": o.title, "language": o.language}
                         for o in others],
            "downloadable": downloads and bool(doc.source_path),
            "owner_department": doc.owner_department}


def utf16(text: str, index: int) -> int:
    return len(text[:index].encode("utf-16-le")) // 2


def locate(page_text: str, chunk_text: str) -> tuple[int, int] | None:
    """Where a chunk's body sits in its page's text, in code points: the whole body, or else the
    span from its first line to its last. ``None`` when neither is found."""
    from fihris.text import body

    text = body(chunk_text).strip()
    if not text:
        return None
    at = page_text.find(text)
    if at >= 0:
        return at, at + len(text)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 8]
    if not lines:
        return None
    first = page_text.find(lines[0])
    if first < 0:
        return None
    last = page_text.find(lines[-1], first)
    end = last + len(lines[-1]) if last >= 0 else first + len(lines[0])
    return first, end


def page_view(session: Session, principal: Principal, doc_id: str, number: int,
              chunk: tuple[int, str] | None = None) -> dict:
    """``chunk``: the cited chunk's number on this page and its indexed text."""
    doc = get_visible(session, principal, doc_id)
    row = session.get(Page, (doc_id, number))
    if row is None:
        raise NotFound(f"{doc_id} page {number}")
    from fihris.retrieve import detect_language

    text = "" if row.excluded else row.text
    highlights = []
    if chunk and not row.excluded:
        span = locate(text, chunk[1])
        if span:
            highlights.append({"chunk_id": f"{doc_id}:p{number}:c{chunk[0]}",
                               "start": utf16(text, span[0]), "end": utf16(text, span[1])})
    view = {"doc_id": doc_id, "title": doc.title, "code": doc.code or None, "page": number,
            "page_count": doc.page_count, "paged": doc.paged, "file_format": doc.file_format,
            "image_url": (f"/api/documents/{doc_id}/pages/{number}/image"
                          if doc.paged and doc.file_format == "pdf" and doc.source_path else ""),
            "text": text, "text_language": detect_language(text) if text else "en",
            "text_source": row.text_source if row.text_source in ("extracted", "repaired", "ocr")
            else "extracted",
            "highlights": highlights, "excluded": row.excluded}
    return view


def render_page(doc_id: str, source: Path, number: int, cache: Path) -> tuple[bytes, int, int]:
    """The page as PNG, rendered with pypdfium2 and cached on disk."""
    target = cache / doc_id / f"page-{number:04d}.png"
    if target.exists():
        data = target.read_bytes()
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            return data, image.width, image.height
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(source))
    try:
        if not 1 <= number <= len(pdf):
            raise NotFound(f"page {number}")
        image = pdf[number - 1].render(scale=IMAGE_SCALE).to_pil()
    finally:
        pdf.close()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(buffer.getvalue())
    return buffer.getvalue(), image.width, image.height
