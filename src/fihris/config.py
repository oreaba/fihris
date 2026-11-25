"""The pinned configuration: the pipeline's settings, frozen and fingerprinted, so a build
records exactly what produced it. Nothing here is read from the environment. Only where the
model servers and stores live is configurable (``Endpoints``)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

QUERY_INSTRUCTION = ("Instruct: Given a web search query, retrieve relevant passages that "
                     "answer the query\nQuery:")
EVERYONE = "all"

# Ready-made CTranslate2 translators on the Hugging Face Hub, pinned to the revisions tested.
# ``opus-mt``: the models Stage 6 measured, converted to float16. ``tc-big``: the larger
# Tatoeba-Challenge models; ai-path's repo has no tokenizer files, so they come from upstream.
TRANSLATOR_SETS = {
    "opus-mt": {
        "en": {"model": "ooeoeo/opus-mt-en-ar-ct2-float16",
               "revision": "908c1eb592272e7e9144145e6785f9af4b70d2db", "token": ">>ara<<"},
        "ar": {"model": "ooeoeo/opus-mt-ar-en-ct2-float16",
               "revision": "4f50aae25375af44f24f87becac4349059d92486", "token": ""},
    },
    "tc-big": {
        "en": {"model": "ooeoeo/opus-mt-tc-big-en-ar-ct2-float16",
               "revision": "28f8107e24e43464db6509ed772a75c4a9c418ba", "token": ">>ara<<"},
        "ar": {"model": "ai-path/opus-mt-tc-big-ar-en-ct2",
               "revision": "787b4410734b7e9b21ef27b9011e37b09c0b00c6", "token": "",
               "tokenizer": "Helsinki-NLP/opus-mt-tc-big-ar-en",
               "tokenizer_revision": "bcb4acd39ee8e3552e171653a8e31a10729b4330"},
    },
}
TOKENIZER_PATTERNS = ["*.spm", "vocab.json", "tokenizer_config.json", "special_tokens_map.json"]

# The reranker runs on a server, but Fihris tokenizes the pairs it sends, so it needs the
# reranker's tokenizer files locally: the revision Stage 6 used, without the model weights.
RERANKER_TOKENIZER = {"revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
                      "patterns": ["*.json", "*.model"]}

# A translation model served behind an OpenAI-compatible chat API (FIHRIS_TRANSLATOR=server):
# Hy-MT3's own prompt (its model card), greedy, with the card's repetition penalty.
SERVER_TRANSLATION = {
    "prompt": ("Translate the following text into {target}. Note that you should only output "
               "the translated result without any additional explanation:\n\n{text}"),
    "target": {"en": "Arabic", "ar": "English"},
    "sampling": {"temperature": 0, "repetition_penalty": 1.05, "max_tokens": 256},
}


@dataclass(frozen=True)
class Pinned:
    chunking: str = "C3"
    page_text: str = "O1"
    max_chars: int = 1000
    min_chars: int = 200
    embedder: str = "Qwen/Qwen3-Embedding-4B"
    embedder_served: str = "qwen3-embedding-4b"
    query_instruction: str = QUERY_INSTRUCTION
    document_truncate_tokens: int = 512
    vector_dim: int = 2560
    reranker: str = "BAAI/bge-reranker-v2-m3"
    reranker_served: str = "bge-reranker-v2-m3"
    reranker_max_tokens: int = 512
    translators: dict = field(default_factory=lambda: {
        "en": ["Helsinki-NLP/opus-mt-en-ar", ">>ara<<"], "ar": ["Helsinki-NLP/opus-mt-ar-en", ""]})
    translation_beams: int = 4
    translation_max_tokens: int = 256
    bm25_depth: int = 100
    dense_depth: int = 100
    rrf_k: int = 60
    fusion_depth: int = 100
    rerank_depth: int = 30
    context_budget_chars: int = 8000
    ocr_sparse_words: int = 50
    ocr_min_confidence: float = 0.75

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


PINNED = Pinned()


@dataclass(frozen=True)
class Endpoints:
    """Where things live. Defaults are loopback ports; each can be set with a ``FIHRIS_*``
    environment variable."""

    embed_url: str = "http://127.0.0.1:8100"
    rerank_url: str = "http://127.0.0.1:8101"
    answer_url: str = "http://127.0.0.1:8000"
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    database_url: str = "postgresql+psycopg://fihris@127.0.0.1:5432/fihris"
    vector_store: str = "pgvector"
    corpus_root: Path = Path("documents")
    work_dir: Path = Path("artifacts/fihris")
    fonts_dir: Path = Path("fonts")
    tesseract_cmd: str | None = None
    translator: str = "ct2"
    translator_models: str = "opus-mt"
    translator_dir: Path | None = None
    translator_device: str = ""
    translate_url: str = "http://127.0.0.1:8102"
    translate_model: str = "hy-mt3-1.8b"
    min_free_gpu_gib: float = 2.0
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls) -> Endpoints:
        env = os.environ.get
        base = cls()
        return cls(
            embed_url=env("FIHRIS_EMBED_URL", base.embed_url),
            rerank_url=env("FIHRIS_RERANK_URL", base.rerank_url),
            answer_url=env("FIHRIS_ANSWER_URL", base.answer_url),
            qdrant_host=env("FIHRIS_QDRANT_HOST", base.qdrant_host),
            qdrant_port=int(env("FIHRIS_QDRANT_PORT", base.qdrant_port)),
            qdrant_grpc_port=int(env("FIHRIS_QDRANT_GRPC_PORT", base.qdrant_grpc_port)),
            database_url=env("FIHRIS_DATABASE_URL", base.database_url),
            vector_store=env("FIHRIS_VECTOR_STORE", base.vector_store),
            corpus_root=Path(env("FIHRIS_CORPUS_ROOT", str(base.corpus_root))),
            work_dir=Path(env("FIHRIS_WORK_DIR", str(base.work_dir))),
            fonts_dir=Path(env("FIHRIS_FONTS_DIR", str(base.fonts_dir))),
            tesseract_cmd=env("FIHRIS_TESSERACT_CMD") or None,
            translator=env("FIHRIS_TRANSLATOR", base.translator),
            translator_models=env("FIHRIS_TRANSLATOR_MODELS", base.translator_models),
            translator_dir=Path(env("FIHRIS_TRANSLATOR_DIR")) if env("FIHRIS_TRANSLATOR_DIR")
            else base.translator_dir,
            translator_device=env("FIHRIS_TRANSLATOR_DEVICE", base.translator_device),
            translate_url=env("FIHRIS_TRANSLATE_URL", base.translate_url),
            translate_model=env("FIHRIS_TRANSLATE_MODEL", base.translate_model),
            min_free_gpu_gib=float(env("FIHRIS_MIN_FREE_GPU_GIB", base.min_free_gpu_gib)),
            timeout_s=float(env("FIHRIS_TIMEOUT_S", base.timeout_s)))
