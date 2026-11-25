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
6. **index:** the chunks into the database, where keyword search reads them, and their vectors into the vector store (pgvector in the same database, or Qdrant: see [Vector store](#vector-store)), in the live build.

Other jobs reuse these steps: `replace` (a new version of a file: all six), `remove`, `reindex_document` and `rechunk_page` (chunk, embed, index) and `reindex_corpus` (chunk, embed and index into a new build, then switch the live alias). `fihris import-pages` (see [Use](#use)) loads page text extracted elsewhere and skips extraction.

## Retrieval

1. Translate the question into the other language (opus-mt through CTranslate2, on the CPU; see [Translator](#translator)).
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
| PostgreSQL with the pgvector extension: documents, pages, chunks, builds, jobs and the vectors | `postgresql+psycopg://fihris@127.0.0.1:5432/fihris` | `FIHRIS_DATABASE_URL` |
| Vector store: `pgvector`, or `qdrant` (see [Vector store](#vector-store)) | `pgvector` | `FIHRIS_VECTOR_STORE` |
| Qdrant, only when it is the vector store | `127.0.0.1:6333` (gRPC 6334) | `FIHRIS_QDRANT_HOST`, `FIHRIS_QDRANT_PORT`, `FIHRIS_QDRANT_GRPC_PORT` |
| Translator: `ct2`, or `torch` (see [Translator](#translator)) | `ct2` | `FIHRIS_TRANSLATOR` |
| Translator models: a ready-made set in the Hugging Face cache (`fihris download-translators`) | `opus-mt` | `FIHRIS_TRANSLATOR_MODELS` |
| Or your own converted copies (`fihris convert-translators`) | unset | `FIHRIS_TRANSLATOR_DIR` |
| Translator device | `cpu` for `ct2`, `cuda` for `torch` | `FIHRIS_TRANSLATOR_DEVICE` |
| Reranker tokenizer (`BAAI/bge-reranker-v2-m3`) in the Hugging Face cache: fihris tokenizes the pairs it sends | | `HF_HOME` |
| Fonts folder for Arabic PDFs (see [pdf-ar](https://github.com/oreaba/pdf-ar#fonts)) | `fonts` | `FIHRIS_FONTS_DIR` |
| Documents folder for `import-pages` | `documents` | `FIHRIS_CORPUS_ROOT` |
| Work folder for uploads and temporary files | `artifacts/fihris` | `FIHRIS_WORK_DIR` |

Model files are read from the local Hugging Face cache only (`local_files_only`), so the engine runs offline once they are downloaded.

## Install

Python 3.11 or later:

```
pip install fihris                  # PostgreSQL, pgvector and the CTranslate2 translator
pip install "fihris[ocr]"           # with the OCR fallback too
pip install "fihris[qdrant]"        # Qdrant as the vector store instead
pip install "fihris[torch]"         # the torch translator, or your own translator conversion
```

Or from a clone of the [repository](https://github.com/oreaba/fihris): `pip install -e ".[ocr,dev]"`.

PostgreSQL itself runs outside pip, with the pgvector extension available, for example:

```
docker run -d --name fihris-postgres -p 127.0.0.1:5432:5432 -e POSTGRES_USER=fihris \
  -e POSTGRES_PASSWORD=change-me -v fihris-pg:/var/lib/postgresql pgvector/pgvector:pg18
export FIHRIS_DATABASE_URL=postgresql+psycopg://fihris:change-me@127.0.0.1:5432/fihris
```

`fihris init-db` then creates the `fihris` schema, its tables and the `vector` extension (its database role must be allowed to create it).

Search translates every question: run `fihris download-translators` once to fetch the ready-made translator models (see [Translator](#translator)).

## Use

```
fihris download-translators                       # once: the translator models
fihris init-db
fihris ingest documents/ --visible-to-all          # files or folders, one job per file
fihris search "How many days of annual leave do staff get?" --text
fihris status
fihris builds
fihris access --doc doc_0123456789abcdef --groups hr
fihris reindex                                    # re-chunk everything into a new build
```

`fihris compare-stores questions.txt --copy` copies the live build's vectors from Qdrant into pgvector and checks that both return the same dense lists (see [Vector store](#vector-store); needs the `qdrant` extra).

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

## Translator

The question is translated into the other language with beam search of 4, one question at a time, by CTranslate2 (`FIHRIS_TRANSLATOR=ct2`, the default) on the CPU. No torch: the `ctranslate2` package is 2 MiB on ARM64 Linux and about 25 MiB on x86-64, against about 500 MiB for torch alone. The ARM64 wheels on PyPI have no GPU support.

The models come ready-made from the Hugging Face Hub, pinned to the revisions tested; nothing is converted. `fihris download-translators` fetches the set `FIHRIS_TRANSLATOR_MODELS` names into the Hugging Face cache, and is the only step that uses the network:

| Set | English to Arabic | Arabic to English | Size | FLORES-200 devtest chrF++, en→ar / ar→en |
|---|---|---|---|---|
| `opus-mt` (default) | `ooeoeo/opus-mt-en-ar-ct2-float16` | `ooeoeo/opus-mt-ar-en-ct2-float16` | 303 MiB | 51.5 / 59.1 |
| `tc-big` | `ooeoeo/opus-mt-tc-big-en-ar-ct2-float16` | `ai-path/opus-mt-tc-big-ar-en-ct2`, with the tokenizer files of `Helsinki-NLP/opus-mt-tc-big-ar-en` | 927 MiB | 57.8 / 65.3 |

- **`opus-mt`** holds the same models as earlier versions of fihris (`Helsinki-NLP/opus-mt-en-ar` and `-ar-en`), converted to float16. On 100 test questions it gave the same translation as a float32 conversion for 98, and the same FLORES scores.
- **`tc-big`** holds the larger Tatoeba-Challenge models. They score about 6 chrF++ higher in each direction, and take about 2.5 times the CPU time (a median of 0.74 s per question on 4 laptop threads, against 0.36 s). Whether they also retrieve better is being evaluated, so `opus-mt` stays the default.

Other ways to run the translator:

- **Your own conversion:** `fihris convert-translators` converts `Helsinki-NLP/opus-mt-en-ar` and `-ar-en` from the Hugging Face cache into CTranslate2's format in float32. It needs torch once (`fihris[torch]`, or an environment that has it, such as a vLLM container). Set `FIHRIS_TRANSLATOR_DIR` to its output folder to use it.
- **torch:** `FIHRIS_TRANSLATOR=torch` with `pip install "fihris[torch]"` runs `Helsinki-NLP/opus-mt-en-ar` and `-ar-en` through transformers, on the GPU by default, as fihris 0.3 and earlier did. On 100 test questions, a float32 CTranslate2 conversion gave the same translation as torch for 93; the other 7 differed by 1 to 6 words.

`fihris compare-translators questions.txt --a opus-mt --b tc-big` runs two translators on the same questions and reports how many translations are identical. For those that differ, it runs retrieval with each and reports whether the sources change. Each side is `torch`, `converted` or a set's name; it needs the whole pipeline unless `--translations-only`.

transformers notes on start that PyTorch was not found: fihris only uses its tokenizers.

## Vector store

Both stores search exactly, comparing the question with every vector and using no approximate index, so they return the same lists. They differ in speed and in what you run:

- **pgvector (default):** the vectors live in fihris's PostgreSQL database (schema `fihris`, one table per build, and the live alias in `fihris.vector_alias`), so there is one database to run and back up. On a DGX Spark at about 12,000 chunks, a search took 80 ms at the 95th percentile with 4 in flight, growing with the corpus (632 ms at ten times the chunks). A question's two searches run in parallel.
- **Qdrant:** its own container, with `pip install "fihris[qdrant]"` and `FIHRIS_VECTOR_STORE=qdrant`. The same measurement gave 8.7 ms (56 ms at ten times the chunks).

With Qdrant, the database may also be SQLite (`FIHRIS_DATABASE_URL=sqlite:///artifacts/fihris/fihris.db`), for a laptop or a test; pgvector needs PostgreSQL.

To move a Qdrant build to pgvector without re-embedding, keep Qdrant reachable and run `fihris compare-stores questions.txt --copy`. It copies the live build's vectors unchanged, then searches every question and its translation in both stores and compares their top 100: chunks whose scores tie within 1e-5 may swap, and anything else counts as different (exit code 1). It also reports each store's search time. `--groups` searches as a principal with those groups, and `--pgvector-url` points the pgvector side at a PostgreSQL database other than fihris's own.

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

Model servers are replaced by deterministic fakes, the database by SQLite and the vector store by `MemoryIndex`, the same contract in NumPy.

With `FIHRIS_TEST_PG_URL` set to a PostgreSQL database with pgvector that the tests may wipe (they drop its `fihris` schema), the engine tests run again on PostgreSQL and pgvector, and pgvector is checked against `MemoryIndex` through updates and build switches:

```
docker run -d -p 127.0.0.1:55432:5432 -e POSTGRES_USER=fihris -e POSTGRES_PASSWORD=fihris pgvector/pgvector:pg18
FIHRIS_TEST_PG_URL=postgresql+psycopg://fihris:fihris@127.0.0.1:55432/fihris pytest
```

## License

[AGPL-3.0](https://github.com/oreaba/fihris/blob/main/LICENSE). Fihris requires textkeel, pdf-ar and PyMuPDF, which are AGPL-3.0 (PyMuPDF also sells a commercial license). The translator models are not part of the package: they are OPUS-MT models by the University of Helsinki, under CC BY 4.0, in conversions published by `ooeoeo` and `ai-path`.
