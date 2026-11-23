"""Deterministic document identifiers.

A document ID derives only from its relative path, so it is stable across runs, machines and
content edits. Moving or renaming a file yields a new ID; content changes are tracked separately
through the SHA-256.

The path is normalised before hashing: separators become ``/`` and every component is Unicode
NFC, so the same Arabic file name produces the same ID whether the filesystem stores it composed
(Windows, Linux) or decomposed (macOS). Case is preserved.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable

DOC_ID_PREFIX = "doc_"
DOC_ID_HEX_LENGTH = 16


def normalize_relative_path(path_or_parts: str | Iterable[str]) -> str:
    """The canonical POSIX, NFC form of a relative path."""
    if isinstance(path_or_parts, str):
        parts = path_or_parts.replace("\\", "/").split("/")
    else:
        parts = list(path_or_parts)
    return "/".join(unicodedata.normalize("NFC", part) for part in parts if part)


def _digest(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()


def make_doc_id(relative_path: str | Iterable[str]) -> str:
    canonical = normalize_relative_path(relative_path)
    if not canonical:
        raise ValueError("relative path is empty")
    return DOC_ID_PREFIX + _digest("doc", canonical)[:DOC_ID_HEX_LENGTH]
