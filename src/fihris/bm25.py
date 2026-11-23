"""Okapi BM25 over loose-normalized words, without the Arabic article or proclitic+article."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from fihris.text import normalize

K1 = 1.2
B = 0.75
PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
MIN_STEM = 3
_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Words of the loose-normalized text, each stripped of one leading article or
    proclitic+article when at least three letters remain."""
    tokens = []
    for word in _WORD.findall(normalize(text, loose=True)):
        prefix = next((p for p in PREFIXES if word.startswith(p)), "")
        tokens.append(word[len(prefix):] if len(word) - len(prefix) >= MIN_STEM else word)
    return tokens


class BM25:
    """Inverted index scored with Lucene's non-negative IDF,
    ``ln(1 + (N - df + 0.5) / (df + 0.5))``."""

    def __init__(self, docs: list[list[str]], k1: float = K1, b: float = B):
        self.k1, self.b, self.n = k1, b, len(docs)
        lengths = np.array([len(doc) for doc in docs], dtype=np.float64)
        self.avgdl = float(lengths.mean()) if lengths.any() else 1.0
        self._norm = k1 * (1 - b + b * lengths / self.avgdl)
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                postings[term][i] = tf
        self.postings = {term: (np.fromiter(found, dtype=np.int64),
                                np.fromiter(found.values(), dtype=np.float64))
                         for term, found in postings.items()}
        self.idf = {term: math.log(1 + (self.n - len(found) + 0.5) / (len(found) + 0.5))
                    for term, found in postings.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        """One score per document. Every query word counts, so a repeated word weighs more."""
        out = np.zeros(self.n)
        for term, count in Counter(query).items():
            if term in self.postings:
                docs, tf = self.postings[term]
                out[docs] += (count * self.idf[term] * tf * (self.k1 + 1)
                              / (tf + self._norm[docs]))
        return out
