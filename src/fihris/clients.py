"""Clients for the model servers (OpenAI-compatible, such as vLLM) and the opus-mt translators.

They are only clients: nothing here starts, stops or reconfigures a server.

- embeddings: ``/v1/embeddings``, vectors normalised to unit length;
- reranking: token IDs from the reranker's own tokenizer, sent to ``/classify``, scored by the
  first class probability;
- translation: opus-mt with beam search over one question at a time, either through CTranslate2
  (``CT2Translator``, the default: ready-made models, no torch, on the CPU) or through
  transformers and torch (``OpusTranslator``, on the GPU by default); or a translation model
  behind ``/v1/chat/completions`` (``ServerTranslator``).
"""

from __future__ import annotations

import json
import threading
import urllib.request
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from fihris.config import (
    PINNED,
    RERANKER_TOKENIZER,
    SERVER_TRANSLATION,
    TOKENIZER_PATTERNS,
    TRANSLATOR_SETS,
    Endpoints,
)

# The measured translators tokenize without sacremoses: installing it, as this warning suggests,
# adds Moses punctuation normalisation and changes translations.
warnings.filterwarnings("ignore", message="Recommended: pip install sacremoses")


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], truncate: int | None = None) -> np.ndarray: ...


class Scorer(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class Translate(Protocol):
    def translate(self, text: str, source_lang: str) -> str: ...


def _post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def snapshot(repo: str, revision: str | None = None,
             allow_patterns: list[str] | None = None) -> Path:
    """A model already in the local Hugging Face cache. Never downloads (air-gapped). A repo
    fetched in part is found only with the patterns it was fetched with."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=allow_patterns,
                                  local_files_only=True))


class EmbeddingClient:
    def __init__(self, endpoints: Endpoints, model: str = PINNED.embedder_served):
        self.url, self.model, self.timeout = endpoints.embed_url, model, endpoints.timeout_s

    def embed(self, texts: Sequence[str], truncate: int | None = None) -> np.ndarray:
        body: dict = {"model": self.model, "input": list(texts)}
        if truncate:
            body["truncate_prompt_tokens"] = truncate
        data = sorted(_post(f"{self.url}/v1/embeddings", body, self.timeout)["data"],
                      key=lambda r: r["index"])
        vectors = np.array([row["embedding"] for row in data], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def reranker_tokenizer() -> Path:
    """The reranker's tokenizer files in the Hugging Face cache: the pinned revision, as
    ``fihris download-models`` fetches it, or else whatever revision ``main`` points to."""
    spec = RERANKER_TOKENIZER
    try:
        return snapshot(PINNED.reranker, spec["revision"], spec["patterns"])
    except Exception:  # noqa: BLE001 - fall back to an older, unpinned download
        pass
    try:
        return snapshot(PINNED.reranker)
    except Exception as error:  # noqa: BLE001 - any miss means it was never downloaded
        raise FileNotFoundError(f"{PINNED.reranker}'s tokenizer is not in the Hugging Face "
                                "cache: run `fihris download-models`") from error


class RerankClient:
    def __init__(self, endpoints: Endpoints, tokenizer_path: Path | None = None,
                 model: str = PINNED.reranker_served):
        from transformers import AutoTokenizer

        self.url, self.model, self.timeout = endpoints.rerank_url, model, endpoints.timeout_s
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or reranker_tokenizer())

    def inputs(self, pairs: Sequence[tuple[str, str]]) -> list[list[int]]:
        return self.tokenizer([q for q, _ in pairs], [t for _, t in pairs],
                              max_length=PINNED.reranker_max_tokens, truncation=True)["input_ids"]

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        data = _post(f"{self.url}/classify", {"model": self.model, "input": self.inputs(pairs)},
                     self.timeout)["data"]
        return [row["probs"][0] for row in sorted(data, key=lambda row: row["index"])]


class GpuMemoryError(RuntimeError):
    pass


def check_gpu(min_free_gib: float) -> dict:
    """Free and total GPU memory. Raises rather than load next to servers that need the room."""
    import torch

    if not torch.cuda.is_available():
        raise GpuMemoryError("no CUDA device; set FIHRIS_TRANSLATOR_DEVICE=cpu to run on the CPU "
                             "(latency is then not the measured configuration)")
    free, total = torch.cuda.mem_get_info()
    info = {"free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2)}
    if free / 2**30 < min_free_gib:
        raise GpuMemoryError(f"only {info['free_gib']} GiB of GPU memory free, need "
                             f"{min_free_gib}; not loading the translators")
    return info


class OpusTranslator:
    """Both opus-mt models, loaded once. One lock per model: beam search runs one question at a
    time, so padding cannot change a translation."""

    def __init__(self, endpoints: Endpoints, paths: dict[str, Path] | None = None):
        try:
            import torch
        except ImportError as error:
            raise ImportError('FIHRIS_TRANSLATOR=torch needs torch: pip install "fihris[torch]"'
                              ) from error
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = endpoints.translator_device or "cuda"
        if self.device != "cpu":
            self.gpu = check_gpu(endpoints.min_free_gpu_gib)
        self.models = {}
        for lang, (repo, token) in PINNED.translators.items():
            path = (paths or {}).get(lang) or snapshot(repo)
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=torch.float32).eval()
            if token and token not in tokenizer.supported_language_codes:
                raise ValueError(f"{token} is not a target language of {repo}")
            self.models[lang] = (tokenizer, model.to(self.device), token, threading.Lock())

    def translate(self, text: str, source_lang: str) -> str:
        import torch

        tokenizer, model, token, lock = self.models[source_lang]
        source = f"{token} {text}" if token else text
        inputs = tokenizer([source], max_length=PINNED.translation_max_tokens, truncation=True,
                           return_tensors="pt").to(self.device)
        with lock, torch.inference_mode():
            ids = model.generate(**inputs, num_beams=PINNED.translation_beams,
                                 max_new_tokens=PINNED.translation_max_tokens)
        return tokenizer.decode(ids[0], skip_special_tokens=True)


TOKENIZER_FILES = ("vocab.json", "source.spm", "target.spm", "tokenizer_config.json",
                   "special_tokens_map.json")


def converted_path(endpoints: Endpoints, repo: str) -> Path:
    """Where ``fihris convert-translators`` puts ``repo`` in CTranslate2's format."""
    return (endpoints.translator_dir or Path("translators")) / repo.split("/")[-1]


def translator_set(name: str) -> dict:
    if name not in TRANSLATOR_SETS:
        raise ValueError(f"FIHRIS_TRANSLATOR_MODELS must be one of {sorted(TRANSLATOR_SETS)}, "
                         f"not {name!r}")
    return TRANSLATOR_SETS[name]


def _cached(repo: str, revision: str, allow_patterns: list[str] | None = None) -> Path:
    try:
        return snapshot(repo, revision, allow_patterns)
    except Exception as error:  # noqa: BLE001 - any miss means it was never downloaded
        raise FileNotFoundError(f"{repo} at {revision[:12]} is not in the Hugging Face cache: "
                                "run `fihris download-models`") from error


def translator_dirs(endpoints: Endpoints) -> dict[str, tuple[Path, Path, str, str]]:
    """Per source language: the model folder, the tokenizer folder, the target-language token and
    the model's name. Converted copies in ``FIHRIS_TRANSLATOR_DIR`` when it is set; otherwise
    the ready-made set ``FIHRIS_TRANSLATOR_MODELS`` from the Hugging Face cache."""
    out = {}
    if endpoints.translator_dir:
        for lang, (repo, token) in PINNED.translators.items():
            path = converted_path(endpoints, repo)
            if not (path / "model.bin").is_file():
                raise FileNotFoundError(
                    f"no CTranslate2 copy of {repo} in {path}: run `fihris convert-translators` "
                    "once where torch is installed, or unset FIHRIS_TRANSLATOR_DIR to use the "
                    "ready-made models")
            out[lang] = (path, path, token, repo)
        return out
    for lang, spec in translator_set(endpoints.translator_models).items():
        model = _cached(spec["model"], spec["revision"])
        tokenizer = (_cached(spec["tokenizer"], spec["tokenizer_revision"], TOKENIZER_PATTERNS)
                     if "tokenizer" in spec else model)
        out[lang] = (model, tokenizer, spec["token"], spec["model"])
    return out


def download_models(translators: str) -> list[dict]:
    """Everything Fihris reads from the Hugging Face cache, at pinned revisions: the ready-made
    translator set ``translators`` and the reranker's tokenizer files. The only step that uses
    the network; nothing else does."""
    from huggingface_hub import snapshot_download

    wanted = []
    for spec in translator_set(translators).values():
        wanted.append((spec["model"], spec["revision"], None))
        if "tokenizer" in spec:
            wanted.append((spec["tokenizer"], spec["tokenizer_revision"], TOKENIZER_PATTERNS))
    wanted.append((PINNED.reranker, RERANKER_TOKENIZER["revision"],
                   RERANKER_TOKENIZER["patterns"]))
    report = []
    for repo, revision, patterns in wanted:
        path = Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))
        report.append({"repo": repo, "revision": revision, "path": str(path),
                       "mib": round(sum(f.stat().st_size for f in path.rglob("*")
                                        if f.is_file()) / 2**20)})
    return report


class CT2Translator:
    """Both directions in CTranslate2's format, float32 compute, with the same beam search as
    ``OpusTranslator`` and no torch: a ready-made set from the Hugging Face cache, or copies
    converted by ``fihris convert-translators``. CPU by default: the ARM64 wheels on PyPI have no
    GPU support. One lock per model, as in ``OpusTranslator``."""

    def __init__(self, endpoints: Endpoints):
        import ctranslate2
        from transformers import MarianTokenizer

        self.device = endpoints.translator_device or "cpu"
        self.models = {}
        for lang, (model, tokenizer_dir, token, name) in translator_dirs(endpoints).items():
            tokenizer = MarianTokenizer.from_pretrained(tokenizer_dir)
            if token and token not in tokenizer.supported_language_codes:
                raise ValueError(f"{token} is not a target language of {name}")
            translator = ctranslate2.Translator(str(model), device=self.device,
                                                compute_type="float32")
            self.models[lang] = (tokenizer, translator, token, threading.Lock())

    def translate(self, text: str, source_lang: str) -> str:
        tokenizer, translator, token, lock = self.models[source_lang]
        source = f"{token} {text}" if token else text
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(source, max_length=PINNED.translation_max_tokens, truncation=True))
        with lock:
            result = translator.translate_batch(
                [tokens], beam_size=PINNED.translation_beams,
                max_decoding_length=PINNED.translation_max_tokens)[0]
        return tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0]),
                                skip_special_tokens=True)


def convert_translators(endpoints: Endpoints, out: Path | None = None,
                        force: bool = False) -> list[dict]:
    """Both opus-mt models from the local Hugging Face cache into CTranslate2's format, float32,
    with their tokenizer files, for ``FIHRIS_TRANSLATOR_DIR``. Needs torch, once: ctranslate2's
    converter loads the model. The ready-made sets need none of this."""
    import ctranslate2

    report = []
    for _lang, (repo, _token) in PINNED.translators.items():
        source = snapshot(repo)
        target = (out / repo.split("/")[-1]) if out else converted_path(endpoints, repo)
        if (target / "model.bin").is_file() and not force:
            report.append({"model": repo, "path": str(target), "converted": False})
            continue
        try:
            converter = ctranslate2.converters.TransformersConverter(
                str(source), copy_files=[f for f in TOKENIZER_FILES if (source / f).is_file()])
            converter.convert(str(target), quantization="float32", force=True)
        except ImportError as error:
            raise ImportError("converting needs torch once: pip install torch, or run it where "
                              "torch is installed (such as the vLLM container)") from error
        (target / "fihris.json").write_text(json.dumps(
            {"model": repo, "revision": source.name, "ctranslate2": ctranslate2.__version__,
             "quantization": "float32"}, indent=1), encoding="utf-8")
        report.append({"model": repo, "revision": source.name, "path": str(target),
                       "converted": True})
    return report


class ServerTranslator:
    """A translation model behind an OpenAI-compatible ``/v1/chat/completions``, such as vLLM
    serving ``tencent/Hy-MT3-1.8B``: one request per question, greedy. The prompt and settings
    are ``SERVER_TRANSLATION``'s."""

    def __init__(self, endpoints: Endpoints):
        self.url, self.model = endpoints.translate_url, endpoints.translate_model
        self.timeout = endpoints.timeout_s

    def translate(self, text: str, source_lang: str) -> str:
        prompt = SERVER_TRANSLATION["prompt"].format(
            target=SERVER_TRANSLATION["target"][source_lang], text=text)
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                **SERVER_TRANSLATION["sampling"]}
        reply = _post(f"{self.url}/v1/chat/completions", body, self.timeout)
        return reply["choices"][0]["message"]["content"].strip()


def open_translator(endpoints: Endpoints) -> Translate:
    """The translator ``FIHRIS_TRANSLATOR`` names: ``ct2`` (the default), ``torch`` or
    ``server``."""
    if endpoints.translator == "ct2":
        return CT2Translator(endpoints)
    if endpoints.translator == "torch":
        return OpusTranslator(endpoints)
    if endpoints.translator == "server":
        return ServerTranslator(endpoints)
    raise ValueError(f"FIHRIS_TRANSLATOR must be ct2, torch or server, not "
                     f"{endpoints.translator!r}")


def health(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=timeout) as response:
            return response.status == 200
    except OSError:
        return False
