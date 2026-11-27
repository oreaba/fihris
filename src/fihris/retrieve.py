"""Retrieval, with the principal's access applied at every step.

1. Translate the question into the other language (opus-mt).
2. BM25 on the question and on its translation, top 100 each.
3. Qwen3-Embedding 4B, with its query instruction, on both, top 100 each, in the store.
4. Reciprocal rank fusion of the four lists, k 60, depth 100.
5. ``bge-reranker-v2-m3`` on the top 30, each chunk against the question in its own language.
6. The chunks that fit whole in 8,000 characters, numbered from 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fihris import ranking
from fihris.chunks import ChunkSet
from fihris.clients import Embedder, Scorer, Translate
from fihris.config import PINNED
from fihris.principal import Principal
from fihris.text import is_arabic


def detect_language(text: str) -> str:
    return "ar" if is_arabic(text) else "en"


def fit(hits: list[dict], budget: int = PINNED.context_budget_chars) -> list[dict]:
    """The leading hits whose texts fit, whole, in ``budget`` characters."""
    used, kept = 0, []
    for hit in hits:
        used += len(hit.get("text") or "")
        if used > budget:
            break
        kept.append(hit)
    return kept


@dataclass
class Source:
    n: int
    doc_id: str
    page: int
    chunk: int
    text: str
    score: float | None
    language: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}:p{self.page}:c{self.chunk}"


@dataclass
class Retrieval:
    question: str
    question_language: str
    translation: str
    sources: list[Source]
    ranked: list[dict] = field(repr=False, default_factory=list)
    timings: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, chunks: ChunkSet, index, embedder: Embedder, scorer: Scorer,
                 translator: Translate):
        self.chunks, self.index = chunks, index
        self.embedder, self.scorer, self.translator = embedder, scorer, translator

    def retrieve(self, question: str, principal: Principal,
                 language: str | None = None) -> Retrieval:
        start = time.perf_counter()
        lang = language or detect_language(question)
        translation = self.translator.translate(question, lang)
        translated = time.perf_counter()
        texts = [question, translation]
        mask = self.chunks.mask(principal)
        keyword = [self.chunks.keyword(x, mask) for x in texts]
        queries = self.embedder.embed([PINNED.query_instruction + x for x in texts])
        # The store filters by access too, but its copy can lag the database, or stay stale when
        # its write failed after the database committed (the store down during a change). So a
        # dense hit also needs the in-memory mask's leave, the same rule keyword search uses. A
        # hit for a chunk this engine does not hold yet (another process is writing the live
        # build; the engine reloads at its next retrieval) is skipped rather than fail the
        # question. In steady state neither drops anything.
        allowed = self.chunks.allowed_docs(principal)
        semantic = [[self.chunks.by_key[key].hit(score) for key, score in hits
                     if key in self.chunks.by_key and key[0] in allowed]
                    for hits in self.index.search(queries, principal, PINNED.dense_depth)]
        fused = ranking.rrf([keyword[0], keyword[1], semantic[0], semantic[1]], k=PINNED.rrf_k,
                           depth=PINNED.fusion_depth)
        head = fused[:PINNED.rerank_depth]
        pairs = [(question if self.chunks.langs[(h["doc_id"], h["page"], h["chunk"])] == lang
                  else translation, h["text"]) for h in head]
        probs = self.scorer.score(pairs)
        ranked = ranking.rerank(fused, lambda _head: probs, top=PINNED.rerank_depth)
        sources = [Source(i, h["doc_id"], h["page"], h["chunk"], h["text"], h["score"],
                          self.chunks.langs[(h["doc_id"], h["page"], h["chunk"])])
                   for i, h in enumerate(fit(ranked), 1)]
        done = time.perf_counter()
        return Retrieval(question, lang, translation, sources, ranked,
                         {"translate_ms": round(1000 * (translated - start)),
                          "retrieval_ms": round(1000 * (done - translated))})
