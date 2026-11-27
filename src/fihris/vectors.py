"""Dense search over one index build, with the access filter applied in the store.

``QdrantIndex``: one collection per build, exact search by inner product over unit vectors; the
live build is reached through an alias, switched only when a build is complete. ``PgVectorIndex``
is the same contract in PostgreSQL with pgvector, in fihris's own database: one table per build,
a one-row alias table, and a sequential scan (no vector index), so search stays exact.
``MemoryIndex`` is the same contract in NumPy, for tests.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
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


def same_ranking(reference: Sequence[tuple[Key, float]], candidate: Sequence[tuple[Key, float]],
                 score_of, tolerance: float = 1e-5) -> bool:
    """EXP-024's matching rule: at every rank, the reference score of the candidate's chunk is
    within ``tolerance`` of the reference score at that rank, and no chunk repeats. Near-ties
    may swap, even across the last rank; nothing else may differ."""
    keys = [k for k, _ in candidate]
    if len(keys) != len(reference) or len(set(keys)) != len(keys):
        return False
    return all(abs(score_of(r) - score_of(c)) <= tolerance
               for (r, _), c in zip(reference, keys, strict=True))


def open_index(endpoints: Endpoints, db, store: str | None = None, collection: str = LIVE_ALIAS):
    """The vector store ``FIHRIS_VECTOR_STORE`` names: ``pgvector`` (the default), which keeps
    the vectors in the database ``db`` that holds everything else, or ``qdrant``."""
    store = store or endpoints.vector_store
    if store == "qdrant":
        return QdrantIndex(endpoints, collection)
    if store == "pgvector":
        return PgVectorIndex(db, collection)
    raise ValueError(f"FIHRIS_VECTOR_STORE must be qdrant or pgvector, not {store!r}")


class QdrantIndex:
    def __init__(self, endpoints: Endpoints, collection: str = LIVE_ALIAS):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as error:
            raise ImportError('the Qdrant store needs the qdrant extra: '
                              'pip install "fihris[qdrant]"') from error

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

    def set_access_many(self, collection: str,
                        access: dict[str, tuple[tuple[str, ...], bool]]) -> None:
        """Every listed document's groups and searchable flag, in batches of payload updates."""
        m = self.models
        items = sorted(access.items())
        for start in range(0, len(items), BATCH):
            self.client.batch_update_points(collection, update_operations=[
                m.SetPayloadOperation(set_payload=m.SetPayload(
                    payload={"groups": list(groups), "searchable": searchable},
                    filter=m.Filter(must=[m.FieldCondition(
                        key="doc_id", match=m.MatchValue(value=doc_id))])))
                for doc_id, (groups, searchable) in items[start:start + BATCH]], wait=True)

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


_TABLE_NAME = re.compile(r"^[a-z0-9_]{1,63}$")


def _literal(vector: np.ndarray) -> str:
    """pgvector's text form. ``repr`` of each float32 widened to a double parses back to the
    same float32, so stored vectors equal the embedded ones bit for bit."""
    return "[" + ",".join(map(repr, np.asarray(vector, dtype=np.float32).tolist())) + "]"


def _parse(text: str) -> np.ndarray:
    return np.array(text[1:-1].split(","), dtype=np.float32)


class PgVectorIndex:
    """One table per build in schema ``fihris``, the live alias in ``fihris.vector_alias``. No
    vector index is created: every search scans the build, as the exact Qdrant search does."""

    def __init__(self, db, collection: str = LIVE_ALIAS):
        from sqlalchemy import text

        if db.dialect.name != "postgresql":
            raise ValueError("FIHRIS_VECTOR_STORE=pgvector needs a PostgreSQL database "
                             f"(FIHRIS_DATABASE_URL), not {db.dialect.name}")
        self.db, self.text, self.collection = db, text, collection
        with db.begin() as conn:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
            conn.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS fihris")
            conn.exec_driver_sql("CREATE TABLE IF NOT EXISTS fihris.vector_alias "
                                 "(name text PRIMARY KEY, collection text NOT NULL)")

    @staticmethod
    def _table(collection: str) -> str:
        if not _TABLE_NAME.match(collection):
            raise ValueError(f"not a collection name: {collection!r}")
        return f"fihris.{collection}"

    def _resolve(self, conn, collection: str | None) -> str | None:
        name = collection or self.collection
        if name != LIVE_ALIAS:
            return name
        return conn.execute(self.text("SELECT collection FROM fihris.vector_alias "
                                      "WHERE name = :n"), {"n": LIVE_ALIAS}).scalar()

    def _exists(self, conn, collection: str) -> bool:
        return conn.execute(self.text("SELECT to_regclass(:t) IS NOT NULL"),
                            {"t": self._table(collection)}).scalar()

    def create(self, collection: str, points: Sequence[Point]) -> None:
        table = self._table(collection)
        with self.db.begin() as conn:
            if self._exists(conn, collection):
                raise ValueError(f"collection {collection} already exists")
            conn.exec_driver_sql(
                f"CREATE TABLE {table} (doc_id varchar(40) NOT NULL, page integer NOT NULL, "
                "chunk integer NOT NULL, lang varchar(2) NOT NULL, groups text[] NOT NULL, "
                f"searchable boolean NOT NULL, embedding vector({PINNED.vector_dim}) NOT NULL, "
                "PRIMARY KEY (doc_id, page, chunk))")
        self.upsert(collection, points)

    def upsert(self, collection: str, points: Sequence[Point]) -> None:
        sql = self.text(
            f"INSERT INTO {self._table(collection)} AS t "
            "(doc_id, page, chunk, lang, groups, searchable, embedding) "
            "VALUES (:doc_id, :page, :chunk, :lang, CAST(:groups AS text[]), :searchable, "
            "CAST(:embedding AS vector)) ON CONFLICT (doc_id, page, chunk) DO UPDATE SET "
            "lang = EXCLUDED.lang, groups = EXCLUDED.groups, searchable = EXCLUDED.searchable, "
            "embedding = EXCLUDED.embedding")
        for start in range(0, len(points), BATCH):
            rows = [{"doc_id": p.doc_id, "page": p.page, "chunk": p.chunk, "lang": p.lang,
                     "groups": list(p.groups), "searchable": p.searchable,
                     "embedding": _literal(p.vector)} for p in points[start:start + BATCH]]
            with self.db.begin() as conn:
                conn.execute(sql, rows)

    def delete_document(self, collection: str, doc_id: str) -> None:
        with self.db.begin() as conn:
            conn.execute(self.text(f"DELETE FROM {self._table(collection)} WHERE doc_id = :d"),
                         {"d": doc_id})

    def set_access(self, collection: str, doc_id: str, groups: Iterable[str],
                   searchable: bool) -> None:
        with self.db.begin() as conn:
            conn.execute(self.text(f"UPDATE {self._table(collection)} SET "
                                   "groups = CAST(:g AS text[]), searchable = :s "
                                   "WHERE doc_id = :d"),
                         {"g": list(groups), "s": searchable, "d": doc_id})

    def set_access_many(self, collection: str,
                        access: dict[str, tuple[tuple[str, ...], bool]]) -> None:
        """Every listed document's groups and searchable flag, in one transaction."""
        rows = [{"g": list(groups), "s": searchable, "d": doc_id}
                for doc_id, (groups, searchable) in sorted(access.items())]
        if not rows:
            return
        with self.db.begin() as conn:
            conn.execute(self.text(f"UPDATE {self._table(collection)} SET "
                                   "groups = CAST(:g AS text[]), searchable = :s "
                                   "WHERE doc_id = :d"), rows)

    def activate(self, collection: str) -> None:
        """Point the live alias at ``collection``, in one statement."""
        self._table(collection)
        with self.db.begin() as conn:
            conn.execute(self.text("INSERT INTO fihris.vector_alias (name, collection) "
                                   "VALUES (:n, :c) ON CONFLICT (name) DO UPDATE SET "
                                   "collection = EXCLUDED.collection"),
                         {"n": LIVE_ALIAS, "c": collection})

    def drop(self, collection: str) -> None:
        with self.db.begin() as conn:
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {self._table(collection)}")

    def vectors(self, collection: str, keys: Sequence[Key]) -> dict[Key, np.ndarray]:
        """Stored vectors of the given chunks, for reuse when their text is unchanged."""
        wanted = set(keys)
        out: dict[Key, np.ndarray] = {}
        docs = sorted({k[0] for k in wanted})
        sql = self.text(f"SELECT doc_id, page, chunk, CAST(embedding AS text) "
                        f"FROM {self._table(collection)} "
                        "WHERE doc_id = ANY(CAST(:docs AS text[]))")
        with self.db.connect() as conn:
            for start in range(0, len(docs), BATCH):
                for doc_id, page, chunk, embedding in conn.execute(
                        sql, {"docs": docs[start:start + BATCH]}):
                    if (doc_id, page, chunk) in wanted:
                        out[(doc_id, page, chunk)] = _parse(embedding)
        return out

    def count(self, collection: str | None = None) -> int:
        with self.db.connect() as conn:
            name = self._resolve(conn, collection)
            if name is None or not self._exists(conn, name):
                return 0
            return conn.execute(self.text(f"SELECT count(*) FROM {self._table(name)}")).scalar()

    def search(self, vectors: np.ndarray, principal: Principal,
               k: int = PINNED.dense_depth) -> list[list[tuple[Key, float]]]:
        vectors = np.asarray(vectors, dtype=np.float32)
        with self.db.connect() as conn:
            name = self._resolve(conn, None)
        if name is None:
            return [[] for _ in vectors]
        where = "searchable"
        params: dict = {"k": k}
        if principal.allowed is not None:
            where += " AND groups && CAST(:groups AS text[])"
            params["groups"] = sorted(principal.allowed)
        sql = self.text(
            "SELECT doc_id, page, chunk, -(embedding <#> CAST(:q AS vector)) AS score "
            f"FROM {self._table(name)} WHERE {where} "
            "ORDER BY embedding <#> CAST(:q AS vector), doc_id, page, chunk LIMIT :k")

        def one(vector: np.ndarray) -> list[tuple[Key, float]]:
            with self.db.connect() as conn:
                return [((d, int(p), int(c)), float(s))
                        for d, p, c, s in conn.execute(sql, {**params, "q": _literal(vector)})]

        if len(vectors) < 2:
            return [one(v) for v in vectors]
        with ThreadPoolExecutor(len(vectors)) as pool:
            return list(pool.map(one, vectors))

    def healthy(self) -> bool:
        try:
            with self.db.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 - any failure means down
            return False

    def close(self) -> None:
        pass


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

    def set_access_many(self, collection: str,
                        access: dict[str, tuple[tuple[str, ...], bool]]) -> None:
        for doc_id, (groups, searchable) in access.items():
            self.set_access(collection, doc_id, groups, searchable)

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
