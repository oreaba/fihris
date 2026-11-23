"""Page text for one file:

- PDFs: pdf-ar's ``fonts`` engine (the font rebuild, gated), one row per page with the
  gate's status, the page text and the page's Markdown;
- Office files: textkeel's own readers and its Office text check, one row for the whole file;
- OCR fallback: pages the gate sends to review are OCRed and replaced when OCR passes its own
  check; accepted pages with images and fewer than 50 words get OCR lines they lack appended.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from fihris.config import PINNED, Endpoints
from fihris.text import normalize

_PAGE_MARK = re.compile(r"<!-- page (\d+) -->")
OFFICE = (".docx", ".xlsx", ".pptx", ".doc")


@dataclass
class PageRow:
    page: int
    status: str
    text: str
    md: str
    has_images: bool = False
    words: int = 0
    reasons: list[str] = field(default_factory=list)
    text_source: str = "extracted"
    ocr_text: str = ""
    ocr_md: str = ""
    ocr_passed: bool = False


class ExtractionError(RuntimeError):
    pass


def _read_pdf_output(folder: Path) -> list[PageRow]:
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    md_path = next((p for p in folder.glob("*.md")), None)
    parts = _PAGE_MARK.split(md_path.read_text(encoding="utf-8")) if md_path else []
    md = {int(parts[i]): parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    rows = []
    for item in report.get("pages", []):
        number = item["page"]
        text_path = folder / "pages" / f"page-{number:03d}.txt"
        rows.append(PageRow(number, item.get("status", "missing"),
                            text_path.read_text(encoding="utf-8") if text_path.exists() else "",
                            md.get(number, ""), bool(item.get("has_images")),
                            int(item.get("words") or 0), list(item.get("reasons") or [])))
    return rows


def _run_pdf(path: Path, out: Path, endpoints: Endpoints, engine: str,
             pages: set[int] | None = None) -> Path:
    from pdf_ar.extract import MissingFontsError, Options, run
    from pdf_ar.ocr import OcrOptions

    ocr = OcrOptions(endpoints.tesseract_cmd, None, "ara+eng", 300, 3, PINNED.ocr_min_confidence,
                     "fast")
    options = Options(out_dir=out, fonts_dir=endpoints.fonts_dir, engine=engine, pages=pages,
                      ocr=ocr, formats=("txt", "md", "json"))
    try:
        run(path, options)
    except MissingFontsError as error:
        raise ExtractionError(f"missing_fonts: {len(error.rows)} font(s) missing or a different "
                              f"version; see {out / 'fonts_report.csv'}") from error
    reports = sorted(out.rglob("report.json"))
    if not reports:
        raise ExtractionError("extraction_failed: no report written")
    return reports[0].parent


def extract_pdf(path: Path, work: Path, endpoints: Endpoints) -> list[PageRow]:
    return _read_pdf_output(_run_pdf(path, work / "pdf", endpoints, "fonts"))


def extract_office(path: Path, work: Path) -> list[PageRow]:
    from textkeel.extract import extract, write

    document = extract(path, doc_cache=work / ".doc-cache")
    folder = work / "office"
    write(document, folder)
    stem = Path(document.source).stem
    text = (folder / f"{stem}.txt").read_text(encoding="utf-8")
    md = (folder / f"{stem}.md").read_text(encoding="utf-8")
    return [PageRow(1, document.status, text, md, reasons=list(document.reasons))]


def _appended(base: str, extra: str) -> str:
    """``base`` plus the lines of ``extra`` it does not already contain."""
    have = normalize(base, loose=True)
    lines = [ln for ln in extra.splitlines()
             if ln.strip() and normalize(ln, loose=True) not in have]
    return base.rstrip() + ("\n\n" + "\n".join(lines) if lines else "")


def needs_ocr(row: PageRow) -> str | None:
    if row.status == "review":
        return "review"
    if row.status == "ok" and row.has_images and row.words < PINNED.ocr_sparse_words:
        return "sparse"
    return None


def apply_o1(rows: list[PageRow], ocr: dict[int, PageRow]) -> list[PageRow]:
    """The OCR fallback rule, page by page. OCR text is kept on review pages either way, for the
    extraction review queue."""
    for row in rows:
        read = ocr.get(row.page)
        if read is None or read.status == "missing":
            continue
        kind = needs_ocr(row)
        row.ocr_text, row.ocr_md, row.ocr_passed = read.text, read.md, read.status == "ok"
        if not row.ocr_passed:
            continue
        if kind == "review":
            row.text, row.md, row.text_source = read.text, read.md, "ocr"
        elif kind == "sparse":
            row.text, row.md = _appended(row.text, read.text), _appended(row.md, read.md)
            row.text_source = "ocr"
    return rows


def ocr(path: Path, work: Path, endpoints: Endpoints, pages: set[int]) -> dict[int, PageRow]:
    if not pages:
        return {}
    folder = _run_pdf(path, work / "ocr", endpoints, "ocr", pages)
    return {r.page: r for r in _read_pdf_output(folder) if r.page in pages}


def extract_file(path: Path, work: Path, endpoints: Endpoints,
                 use_ocr: bool = True) -> tuple[list[PageRow], list[str]]:
    """O1 page rows for one file, and any warnings (OCR unavailable)."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    suffix = path.suffix.lower()
    warnings: list[str] = []
    if suffix == ".pdf":
        rows = extract_pdf(path, work, endpoints)
        wanted = {r.page for r in rows if needs_ocr(r)}
        if wanted and use_ocr:
            try:
                rows = apply_o1(rows, ocr(path, work, endpoints, wanted))
            except Exception as error:  # noqa: BLE001 - OCR is a fallback; record and go on
                warnings.append(f"ocr_failed: {type(error).__name__}: {error}")
        elif wanted:
            warnings.append(f"ocr_skipped: {len(wanted)} page(s) wanted OCR")
    elif suffix in OFFICE:
        rows = extract_office(path, work)
    else:
        raise ExtractionError(f"unsupported_format: {suffix}")
    return rows, warnings
