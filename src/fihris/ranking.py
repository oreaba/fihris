"""Rank-list operations: reciprocal rank fusion and reranking of a list's head."""

from __future__ import annotations

from collections.abc import Callable

from fihris.units import hit_key

K = 60


def rrf(lists: list[list[dict]], k: int = K, depth: int = 100) -> list[dict]:
    """The top ``depth`` units by the sum of ``1 / (k + rank)`` over the lists, each cut to its
    top ``depth``. Equal scores keep the order of first appearance when the lists are read
    rank by rank, first list first. A fused hit keeps the fields of the unit's first hit."""
    lists = [hits[:depth] for hits in lists]
    scores: dict[tuple, float] = {}
    first: dict[tuple, dict] = {}
    for rank in range(max((len(hits) for hits in lists), default=0)):
        for hits in lists:
            if rank < len(hits):
                key = hit_key(hits[rank])
                scores[key] = scores.get(key, 0.0) + 1 / (k + rank + 1)
                first.setdefault(key, hits[rank])
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    return [{**first[key], "score": score} for key, score in ranked[:depth]]


def rerank(hits: list[dict], score: Callable[[list[dict]], list[float]],
           top: int = 30) -> list[dict]:
    """The first ``top`` hits reordered by ``score`` (equal scores keep their order), then the
    rest unchanged and without a score."""
    head = hits[:top]
    values = score(head)
    order = sorted(range(len(head)), key=lambda i: -values[i])
    return ([{**head[i], "score": float(values[i])} for i in order]
            + [{**hit, "score": None} for hit in hits[top:]])
