"""The chunk set of one index build, held in memory: unit texts and languages, the BM25 index,
and each document's groups and searchability, for the access mask."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fihris import bm25
from fihris.chunking import contextual, structured
from fihris.config import PINNED
from fihris.principal import Principal
from fihris.text import body, is_arabic
from fihris.units import Unit, fingerprint, load_chunks


def unit_language(text: str) -> str:
    return "ar" if is_arabic(body(text)) else "en"


def chunk_document(pages: list[dict], title: str, doc_id: str) -> Iterator[dict]:
    """Structure-aware chunks of one document's pages, each under its document and section
    header."""
    for n, chunk in enumerate(structured(sorted(pages, key=lambda r: r["page"]),
                                         PINNED.max_chars, PINNED.min_chars)):
        yield {"doc_id": doc_id, "page": chunk["page"], "chunk": n,
               "headings": chunk["headings"], "text": contextual(chunk, title)}


def chunk_corpus(by_doc: dict[str, list[dict]], titles: dict[str, str]) -> list[dict]:
    """Every document's chunks, in ``doc_id`` order."""
    out = []
    for doc_id in sorted(by_doc):
        out.extend(chunk_document(by_doc[doc_id], titles.get(doc_id, doc_id), doc_id))
    return out


def write_chunks(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Access:
    """A document's groups (``all`` when visible to everyone) and whether search may return it:
    superseded, duplicate and removed documents may not."""

    groups: tuple[str, ...]
    searchable: bool = True


@dataclass
class ChunkSet:
    units: list[Unit]
    access: dict[str, Access] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_key = {u.key: u for u in self.units}
        self.texts = {u.key: u.text for u in self.units}
        self.langs = {u.key: unit_language(u.text) for u in self.units}
        self.doc_ids = [u.doc_id for u in self.units]
        self.bm25 = bm25.BM25([bm25.tokenize(u.text) for u in self.units])
        self._masks: dict[frozenset[str] | None, np.ndarray] = {}

    @classmethod
    def from_file(cls, path: Path, access: dict[str, Access] | None = None) -> ChunkSet:
        return cls(load_chunks(path), access or {})

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.units)

    def set_access(self, access: dict[str, Access]) -> None:
        self.access = access
        self._masks.clear()

    def mask(self, principal: Principal) -> np.ndarray:
        """Which units the principal may retrieve."""
        key = principal.allowed
        if key not in self._masks:
            allowed_docs = {doc for doc, a in self.access.items()
                            if a.searchable and principal.may_read(a.groups)}
            if not self.access:
                allowed_docs = set(self.doc_ids) if key is None else set()
            self._masks[key] = np.fromiter((d in allowed_docs for d in self.doc_ids), dtype=bool,
                                           count=len(self.doc_ids))
        return self._masks[key]

    def keyword(self, text: str, mask: np.ndarray, depth: int = PINNED.bm25_depth) -> list[dict]:
        """The top BM25 hits over the allowed units only: the stable order of all scores,
        restricted before cutting, so a restricted principal still gets a full top list."""
        scores = self.bm25.scores(bm25.tokenize(text))
        order = np.argsort(-scores, kind="stable")
        if not mask.all():
            order = order[mask[order]]
        return [self.units[i].hit(float(scores[i])) for i in order[:depth]]
