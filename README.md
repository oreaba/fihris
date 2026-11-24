# fihris

Fihris (فهرس, "index") is a retrieval engine for Arabic and English documents. It does both halves of retrieval-augmented generation up to the answer:

- **Ingestion:** it extracts files, checks and stores their page text, chunks it, embeds the chunks and indexes them ([Ingestion](#ingestion)).
- **Retrieval:** it answers a question with numbered, access-filtered sources that fit a context budget ([Retrieval](#retrieval)).

It stops there: generating the answer from the sources is the caller's job.

- **Formats:** PDF, Word (`.docx`, and `.doc` through a local Word), Excel (`.xlsx`) and PowerPoint (`.pptx`), read by [textkeel](https://github.com/oreaba/textkeel), with Arabic PDFs rebuilt from their fonts by [pdf-ar](https://github.com/oreaba/pdf-ar). Other file types fail with `unsupported_format`.
- **Quality:** PDF pages the quality gate rejects fall back to OCR when OCR passes its own check. Otherwise they go to a review queue, and their rebuilt text stays searchable until an admin decides.
- **Chunks:** structure-aware, never crossing a page, each under a `Document:` and `Section:` header. Repeated page furniture is kept once.
- **Search across languages:** the question is translated into the other language (opus-mt), and BM25 and dense search run on both, so an English question finds Arabic-only evidence and the reverse.
- **Ranking:** reciprocal rank fusion of the four lists, then a cross-encoder reranker on the top 30, each chunk scored against the question in its own language.
- **Access:** every document carries groups. A principal sees only documents sharing a group, at every step: keyword, dense, reranking and the library. A document outside a principal's groups is reported as missing.
- **Builds:** each index build is its own vector collection behind a live alias. Single-document changes update the live build; a corpus re-index builds a new collection, reusing unchanged vectors, and switches only when complete.

## Ingestion

`fihris ingest` (or `Engine.submit("ingest", ...)`) runs each file as a job of six steps, each recording its progress:

1. **extract:** PDFs through pdf-ar's `fonts` engine, so Arabic PDFs need their exact fonts in the fonts folder (a missing font fails the job with `missing_fonts`); Word, Excel and PowerPoint through textkeel and its Office text check.
2. **quality_gate:** each PDF page is `ok` or `review`; an Office file is one page.
3. **ocr:** PDF pages in review are OCRed, and the OCR text replaces them when it passes its own check. Accepted pages with images and fewer than 50 words get the OCR lines they lack appended. Pages still in review go to the extraction review queue.
4. **chunk:** structure-aware chunks, never crossing a page, each under a `Document:` and `Section:` header.
5. **embed:** each chunk through the embedding server (Qwen3-Embedding 4B).
6. **index:** the chunks into the database, where keyword search reads them, and their vectors into Qdrant, in the live build.

Other jobs reuse these steps: `replace` (a new version of a file: all six), `remove`, `reindex_document` and `rechunk_page` (chunk, embed, index) and `reindex_corpus` (chunk, embed and index into a new build, then switch the live alias). `fihris import-pages` (see [Use](#use)) loads page text extracted elsewhere and skips extraction.

## Retrieval

1. Translate the question into the other language (opus-mt, local).
2. BM25 on the question and on its translation, top 100 each, over the allowed chunks.
3. Dense search with Qwen3-Embedding 4B (with its query instruction) on both, top 100 each, filtered in the vector store.
4. Reciprocal rank fusion of the four lists (k 60, depth 100).
5. `bge-reranker-v2-m3` on the top 30.
6. The chunks that fit whole in 8,000 characters, numbered from 1.

These settings are pinned in [`config.py`](https://github.com/oreaba/fihris/blob/main/src/fihris/config.py) (`Pinned`) and fingerprinted, and every build records the fingerprint of the configuration that made it.

## What it needs

| Part | Default | Setting |
|---|---|---|
| Embedding server (OpenAI-compatible `/v1/embeddings`, such as vLLM serving `Qwen/Qwen3-Embedding-4B`) | `http://127.0.0.1:8100` | `FIHRIS_EMBED_URL` |
| Reranker server (vLLM `/classify` serving `BAAI/bge-reranker-v2-m3`) | `http://127.0.0.1:8101` | `FIHRIS_RERANK_URL` |
| Qdrant | `127.0.0.1:6333` (gRPC 6334) | `FIHRIS_QDRANT_HOST`, `FIHRIS_QDRANT_PORT`, `FIHRIS_QDRANT_GRPC_PORT` |
| Database (SQLAlchemy URL; SQLite or PostgreSQL) | `sqlite:///artifacts/fihris/fihris.db` | `FIHRIS_DATABASE_URL` |
| Translator models (`Helsinki-NLP/opus-mt-en-ar`, `opus-mt-ar-en`) in the Hugging Face cache | GPU when available | `FIHRIS_TRANSLATOR_DEVICE` |
| Fonts folder for Arabic PDFs (see [pdf-ar](https://github.com/oreaba/pdf-ar#fonts)) | `fonts` | `FIHRIS_FONTS_DIR` |
| Documents folder for `import-pages` | `documents` | `FIHRIS_CORPUS_ROOT` |
| Work folder for uploads and temporary files | `artifacts/fihris` | `FIHRIS_WORK_DIR` |

Model files are read from the local Hugging Face cache only (`local_files_only`), so the engine runs offline once they are downloaded.

## Install

Python 3.11 or later:

```
pip install "fihris[translate]"                  # with SQLite
pip install "fihris[postgres,translate,ocr]"     # PostgreSQL and the OCR fallback too
```

Or from a clone of the [repository](https://github.com/oreaba/fihris): `pip install -e ".[translate,ocr,dev]"`.

Search translates every question, so it needs the `translate` extra (torch). Without it, ingestion, the library and builds still work. Install the torch build that matches your hardware first if you want CUDA.

## Use

```
fihris init-db
fihris ingest documents/ --visible-to-all          # files or folders, one job per file
fihris search "How many days of annual leave do staff get?" --text
fihris status
fihris builds
fihris access --doc doc_0123456789abcdef --groups hr
fihris reindex                                    # re-chunk everything into a new build
```

`fihris import-pages --catalogue docs.jsonl --pages pages.jsonl --activate` loads documents extracted elsewhere: a catalogue with one document per line (any `Document` column, and at least `relative_path` or `doc_id`) and a page index (`doc_id`, `page`, `status`, `text` and optional `md` per line).

Output is content-free (counts, ids, scores) unless `--text` is asked for.

From Python:

```python
from fihris.config import Endpoints
from fihris.db import create_schema, engine_for, sessions
from fihris.engine import Engine
from fihris.principal import Principal

endpoints = Endpoints.from_env()
db = engine_for(endpoints.database_url)
create_schema(db)
engine = Engine(endpoints, sessions(db))
engine.load()
result = engine.retrieve("كم مدة الإجازة السنوية؟", Principal.for_groups(["hr"], pilot_mode=False))
for source in result.sources:
    print(source.n, source.doc_id, source.page, source.score)
```

`Engine.upsert_documents` adds documents from any catalogue, and `Engine.submit("ingest", ...)` runs a file through extraction, chunking, embedding and indexing as a job with step-by-step progress.

## Access model

- Every document visible to everyone carries the group `all`, and every principal holds it.
- `Principal.for_groups(groups, pilot_mode)`: in pilot mode a principal is narrowed to `all` alone.
- `Principal.unrestricted()` exists for the command line and offline checks; nothing a caller sends can produce it.
- Superseded, duplicate and removed documents are never returned by search.

## Tests

```
pip install -e ".[dev]"
pytest
```

Model servers are replaced by deterministic fakes, and the vector store by `MemoryIndex`, the same contract in NumPy.

## License

[AGPL-3.0](https://github.com/oreaba/fihris/blob/main/LICENSE). Fihris requires textkeel, pdf-ar and PyMuPDF, which are AGPL-3.0 (PyMuPDF also sells a commercial license).
