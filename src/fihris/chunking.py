"""Structure-aware chunks of a document's pages.

Chunks never cross a page, so every chunk belongs to exactly one page and citations stay exact.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from fihris.text import normalize

MAX_CHARS = 1000
MIN_CHARS = 200

_COLON_HEADING = re.compile(r"^[^|]{1,60}[:：]$")
_NUMBERED_HEADING = re.compile(r"^([0-9٠-٩]+(?:\.[0-9٠-٩]+)+)[.)]?\s+\S")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}")


def windows(text: str, size: int, overlap: int = 0) -> list[str]:
    """Fixed-size windows of about ``size`` characters, cut at spaces, overlapping by about
    ``overlap`` characters."""
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > start else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = text.find(" ", max(start + 1, end - overlap))
        start = next_start + 1 if 0 <= next_start < end else end
    return [w for w in out if w]


def _blocks(md: str) -> list[tuple[str, str]]:
    """(kind, text) blocks of one page's Markdown: ``table`` rows grouped, ``text`` paragraphs
    split at blank lines."""
    out: list[tuple[str, str]] = []
    paragraph: list[str] = []
    table: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append(("text", " ".join(paragraph)))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            out.append(("table", "\n".join(table)))
            table.clear()

    for raw in md.splitlines():
        line = raw.strip()
        if line.startswith("<!--"):
            continue
        if line.startswith("|"):
            flush_paragraph()
            table.append(line)
        elif not line:
            flush_paragraph()
            flush_table()
        else:
            flush_table()
            paragraph.append(line)
    flush_paragraph()
    flush_table()
    return out


def table_rows(table: str) -> list[str]:
    """Each row of a Markdown table as its non-empty cells joined by `` | ``."""
    rows = []
    for line in table.splitlines():
        if _TABLE_RULE.match(line.strip("| ")) or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(dict.fromkeys(cells)))
    return rows


def heading_level(text: str) -> int:
    """0 if not a heading; else 1 for a label ending in a colon, the number of parts for a
    numbered section such as 2.1, or the number of # for a Markdown heading."""
    text = text.strip()
    if match := _MD_HEADING.match(text):
        return len(match.group(1))
    if len(text) > 80:
        return 0
    words = len(text.split())
    if _COLON_HEADING.match(text) and words <= 8:
        return 1
    if (match := _NUMBERED_HEADING.match(text)) and words <= 10 and not text.endswith("."):
        return match.group(1).count(".") + 1
    return 0


def _furniture(pages: list[list[str]]) -> set[str]:
    """Normalized lines that repeat on at least half of a document's pages (and on two or
    more): page headers, footers and document-control tables."""
    if len(pages) < 2:
        return set()
    counts = Counter(key for lines in pages for key in {normalize(x, loose=True) for x in lines})
    need = max(2, math.ceil(len(pages) / 2))
    return {key for key, n in counts.items() if n >= need and key}


def structured(doc_pages: list[dict], max_chars: int = MAX_CHARS,
               min_chars: int = MIN_CHARS) -> Iterator[dict]:
    """Structure-aware chunks of one document: repeated page furniture kept only on its first
    page, a new chunk at each heading, table rows kept whole, at most ``max_chars``, and no
    chunk crossing a page. Each chunk carries its heading path."""
    parsed = []
    for page in doc_pages:
        lines: list[tuple[str, str]] = []
        for kind, text in _blocks(page.get("md") or page.get("text", "")):
            if kind == "table":
                lines.extend(("row", row) for row in table_rows(text))
            else:
                lines.append(("text", text))
        parsed.append((page["page"], lines))
    furniture = _furniture([[t for _k, t in lines] for _p, lines in parsed])
    seen: set[str] = set()
    path: list[tuple[int, str]] = []
    for number, lines in parsed:
        yield from _page_chunks(number, lines, path, furniture, seen, max_chars, min_chars)


def _page_chunks(number: int, lines: list[tuple[str, str]], path: list[tuple[int, str]],
                 furniture: set[str], seen: set[str], max_chars: int,
                 min_chars: int) -> Iterator[dict]:
    """One page's chunks. ``path`` (the heading path) and ``seen`` (furniture already kept)
    carry over from page to page. Text shorter than ``min_chars`` before a heading joins the
    next chunk instead of standing alone."""
    buffer: list[str] = []

    def flush(force: bool = False) -> Iterator[dict]:
        text = "\n".join(buffer).strip()
        if text and (force or len(text) >= min_chars):
            yield {"page": number, "headings": [h for _lvl, h in path], "text": text}
            buffer.clear()

    for kind, text in lines:
        key = normalize(text, loose=True)
        if key in furniture:
            if key in seen:
                continue
            seen.add(key)
        level = heading_level(text) if kind == "text" else 0
        if level:
            yield from flush()
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, _MD_HEADING.sub(r"\2", text).strip()))
            buffer.append(path[-1][1])
            continue
        pieces = [text] if len(text) <= max_chars else windows(text, max_chars, 0)
        for piece in pieces:
            if buffer and sum(len(b) + 1 for b in buffer) + len(piece) > max_chars:
                yield from flush(force=True)
            buffer.append(piece)
    yield from flush(force=True)


def contextual(chunk: dict, title: str) -> str:
    """The chunk text under a header naming its document and section."""
    lines = [f"Document: {title}"]
    if chunk.get("headings"):
        lines.append("Section: " + " > ".join(chunk["headings"]))
    return "\n".join(lines) + "\n" + chunk["text"]


def doc_title(relative_path: str) -> str:
    """A title from a file name: its stem, with underscores as spaces."""
    return " ".join(Path(relative_path).stem.replace("_", " ").split())


def load_pages(index: Path) -> dict[str, list[dict]]:
    """A page index (JSON lines with ``doc_id``, ``page``, ``text`` and optional ``md``),
    grouped by document in page order."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    with index.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_doc[row["doc_id"]].append(row)
    for pages in by_doc.values():
        pages.sort(key=lambda r: r["page"])
    return by_doc
