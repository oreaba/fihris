"""Dense search over one index build, with the access filter applied in the store.

``QdrantIndex``: one collection per build, exact search by inner product over unit vectors; the
live build is reached through an alias, switched only when a build is complete. ``MemoryIndex``
is the same contract in NumPy, for tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from fihris.config import PINNED, Endpoints
from fihris.principal import Principal


def point_id(key: tuple) -> int:
    """A stable 63-bit integer for a chunk key, for stores that need integer IDs."""
    digest = hashlib.sha256("\x1f".join(map(str, key)).encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1

Key = tuple[str, int, int]
LIVE_ALIAS = "fihris_live"
BATCH = 256


@dataclass(frozen=True)
class Point:
    doc_id: str
    page: int
    chunk: int
    vector: np.ndarray
    lang: str
    groups: tuple[str, ...]
    searchable: bool = True

    @property
    def key(self) -> Key:
        return (self.doc_id, self.page, self.chunk)


def collection_name(build_id: str) -> str:
    return "fihris_" + build_id.replace("-", "_")


class QdrantIndex:
    def __init__(self, endpoints: Endpoints, collection: str = LIVE_ALIAS):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.collection = collection
        self.client = QdrantClient(host=endpoints.qdrant_host, port=endpoints.qdrant_port,
                                   grpc_port=endpoints.qdrant_grpc_port, prefer_grpc=True,
                                   timeout=120)

    def _payload(self, p: Point) -> dict:
        return {"doc_id": p.doc_id, "page": p.page, "chunk": p.chunk, "lang": p.lang,
                "groups": list(p.groups), "searchable": p.searchable}

    def create(self, collection: str, points: Sequence[Point]) -> None:
        m = self.models
        if self.client.collection_exists(collection):
            raise ValueError(f"collection {collection} already exists")
        self.client.create_collection(
            collection, vectors_config=m.VectorParams(size=PINNED.vector_dim,
                                                      distance=m.Distance.DOT),
            hnsw_config=m.HnswConfigDiff(m=0))
        for name in ("doc_id", "groups"):
            self.client.create_payload_index(collection, name,
                                             field_schema=m.PayloadSchemaType.KEYWORD, wait=True)
        self.client.create_payload_index(collection, "page",
                                         field_schema=m.PayloadSchemaType.INTEGER, wait=True)
        self.client.create_payload_index(collection, "searchable",
                                         field_schema=m.PayloadSchemaType.BOOL, wait=True)
        self.upsert(collection, points)

    def upsert(self, collection: str, points: Sequence[Point]) -> None:
        m = self.models
        for start in range(0, len(points), BATCH):
            batch = points[start:start + BATCH]
            self.client.upsert(collection, [m.PointStruct(
                id=point_id(p.key), vector=p.vector.astype(np.float32).tolist(),
                payload=self._payload(p)) for p in batch], wait=True)

    def delete_document(self, collection: str, doc_id: str) -> None:
        m = self.models
        self.client.delete(collection, points_selector=m.FilterSelector(filter=m.Filter(
            must=[m.FieldCondition(key="doc_id", match=m.MatchValue(value=doc_id))])), wait=True)

    def set_access(self, collection: str, doc_id: str, groups: Iterable[str],
                   searchable: bool) -> None:
        m = self.models
        self.client.set_payload(collection, payload={"groups": list(groups),
                                                     "searchable": searchable},
                                points=m.Filter(must=[m.FieldCondition(
                                    key="doc_id", match=m.MatchValue(value=doc_id))]), wait=True)

    def activate(self, collection: str) -> None:
        """Point the live alias at ``collection``, in one operation."""
        m = self.models
        ops = []
        if any(a.alias_name == LIVE_ALIAS for a in self.client.get_aliases().aliases):
            ops.append(m.DeleteAliasOperation(delete_alias=m.DeleteAlias(alias_name=LIVE_ALIAS)))
        ops.append(m.CreateAliasOperation(create_alias=m.CreateAlias(
            collection_name=collection, alias_name=LIVE_ALIAS)))
        self.client.update_collection_aliases(change_aliases_operations=ops)

    def drop(self, collection: str) -> None:
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)

    def vectors(self, collection: str, keys: Sequence[Key]) -> dict[Key, np.ndarray]:
        """Stored vectors of the given chunks, for reuse when their text is unchanged."""
        out: dict[Key, np.ndarray] = {}
        ids = {point_id(k): k for k in keys}
        names = list(ids)
        for start in range(0, len(names), BATCH):
            for p in self.client.retrieve(collection, ids=names[start:start + BATCH],
                                          with_vectors=True, with_payload=False):
                out[ids[int(p.id)]] = np.asarray(p.vector, dtype=np.float32)
        return out

    def count(self, collection: str | None = None) -> int:
        return self.client.count(collection or self.collection, exact=True).count

    def search(self, vectors: np.ndarray, principal: Principal,
               k: int = PINNED.dense_depth) -> list[list[tuple[Key, float]]]:
        m = self.models
        must = [m.FieldCondition(key="searchable", match=m.MatchValue(value=True))]
        if principal.allowed is not None:
            must.append(m.FieldCondition(key="groups",
                                         match=m.MatchAny(any=sorted(principal.allowed))))
        out = []
        for vector in np.asarray(vectors, dtype=np.float32):
            points = self.client.query_points(
                self.collection, query=vector.tolist(), limit=k,
                query_filter=m.Filter(must=must), search_params=m.SearchParams(exact=True),
                with_payload=m.PayloadSelectorInclude(include=["doc_id", "page", "chunk"])).points
            out.append([((p.payload["doc_id"], int(p.payload["page"]), int(p.payload["chunk"])),
                         float(p.score)) for p in points])
        return out

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - any failure means down
            return False

    def close(self) -> None:
        self.client.close()


class MemoryIndex:
    """Exact inner-product search in NumPy with the same collections, alias and filter rules."""

    def __init__(self):
        self.collections: dict[str, dict[Key, Point]] = {}
        self.live: str | None = None

    def create(self, collection: str, points: Sequence[Point]) -> None:
        if collection in self.collections:
            raise ValueError(f"collection {collection} already exists")
        self.collections[collection] = {}
        self.upsert(collection, points)

    def upsert(self, collection: str, points: Sequence[Point]) -> None:
        self.collections[collection].update({p.key: p for p in points})

    def delete_document(self, collection: str, doc_id: str) -> None:
        self.collections[collection] = {k: p for k, p in self.collections[collection].items()
                                        if p.doc_id != doc_id}

    def set_access(self, collection: str, doc_id: str, groups: Iterable[str],
                   searchable: bool) -> None:
        groups = tuple(groups)
        self.collections[collection] = {
            k: (Point(p.doc_id, p.page, p.chunk, p.vector, p.lang, groups, searchable)
                if p.doc_id == doc_id else p) for k, p in self.collections[collection].items()}

    def activate(self, collection: str) -> None:
        self.live = collection

    def drop(self, collection: str) -> None:
        self.collections.pop(collection, None)

    def vectors(self, collection: str, keys: Sequence[Key]) -> dict[Key, np.ndarray]:
        points = self.collections.get(collection, {})
        return {k: points[k].vector for k in keys if k in points}

    def count(self, collection: str | None = None) -> int:
        return len(self.collections.get(collection or self.live or "", {}))

    def search(self, vectors: np.ndarray, principal: Principal,
               k: int = PINNED.dense_depth) -> list[list[tuple[Key, float]]]:
        points = self.collections.get(self.live or "", {})
        allowed = [p for p in points.values()
                   if p.searchable and principal.may_read(p.groups)]
        if not allowed:
            return [[] for _ in vectors]
        matrix = np.stack([p.vector for p in allowed]).astype(np.float32)
        out = []
        for vector in np.asarray(vectors, dtype=np.float32):
            scores = matrix @ vector
            order = np.argsort(-scores, kind="stable")[:k]
            out.append([(allowed[i].key, float(scores[i])) for i in order])
        return out

    def healthy(self) -> bool:
        return True
