"""Text normalisation and language detection for Arabic and English."""

from __future__ import annotations

import re
import unicodedata

_MARKS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_PUNCT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                        "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u0640": None,
                        "\u200b": None, "\u200e": None, "\u200f": None, "|": " "})
_LOOSE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه",
                        "ؤ": "و", "ئ": "ي", "٪": "%", "،": ",", "؛": ";", "؟": "?",
                        **{chr(0x660 + i): str(i) for i in range(10)},
                        **{chr(0x6F0 + i): str(i) for i in range(10)}})
_ARABIC = re.compile("[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")
_LATIN = re.compile("[A-Za-z]")
HEADER_PREFIXES = ("Document: ", "Section: ")


def normalize(text: str, loose: bool = False) -> str:
    """Strict: NFKC, no Arabic marks or tatweel, plain quotes, dashes and table pipes, folded
    case and spaces. Loose also folds hamza seats, alef maksura, teh marbuta and digits."""
    text = _MARKS.sub("", unicodedata.normalize("NFKC", text)).translate(_PUNCT)
    if loose:
        text = text.translate(_LOOSE)
    return " ".join(text.casefold().split())


def arabic_share(text: str) -> float:
    """Arabic letters as a share of Arabic and Latin letters."""
    arabic, latin = len(_ARABIC.findall(text)), len(_LATIN.findall(text))
    return arabic / (arabic + latin) if arabic + latin else 0.0


def is_arabic(text: str) -> bool:
    """Arabic letters make up at least half of the letters."""
    return arabic_share(text) >= 0.5


def body(text: str) -> str:
    """Chunk text without its leading ``Document:`` and ``Section:`` header lines."""
    lines = text.split("\n")
    start = 0
    for prefix in HEADER_PREFIXES:
        if start < len(lines) and lines[start].startswith(prefix):
            start += 1
    return "\n".join(lines[start:])
