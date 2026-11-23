"""Retrieval units: one chunk of one page, and the run-file hits that point to them."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Unit:
    """One PDF page, one whole Office file (page 1), or one chunk of a page."""

    doc_id: str
    page: int
    status: str
    text: str
    chunk: int | None = None

    @property
    def key(self) -> tuple:
        return (self.doc_id, self.page) if self.chunk is None else (
            self.doc_id, self.page, self.chunk)

    def hit(self, score: float | None) -> dict:
        """A run-file hit. A chunk hit carries its text, as indexed."""
        if self.chunk is None:
            return {"doc_id": self.doc_id, "page": self.page, "score": score}
        return {"doc_id": self.doc_id, "page": self.page, "chunk": self.chunk, "score": score,
                "text": self.text}


def hit_key(hit: dict) -> tuple:
    """The unit a hit points to: its page, or its chunk of the page."""
    return (hit["doc_id"], hit["page"]) if "chunk" not in hit else (
        hit["doc_id"], hit["page"], hit["chunk"])


def load_chunks(path: Path) -> list[Unit]:
    """Every chunk of a chunk file (JSON lines), in file order, with its text exactly as given."""
    units = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                units.append(Unit(row["doc_id"], row["page"], "", row["text"], row["chunk"]))
    return units


def fingerprint(units: list[Unit]) -> str:
    """SHA-256 over unit keys and text, to tie caches and runs to one unit set without storing
    any text."""
    digest = hashlib.sha256()
    for unit in units:
        digest.update(("\x1f".join([*map(str, unit.key), unit.text]) + "\x1e").encode())
    return digest.hexdigest()
