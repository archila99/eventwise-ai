"""Keyword tokenisation for BM25.

Keeps identifiers and numbers searchable:
- ``CW-AGR-2026-003`` -> the whole identifier, its contiguous sub-identifiers
  (``agr-2026-003``, ...) and its parts, so a partial ID still matches;
- ``£176,400`` / ``176,400.00`` -> ``176400``; ``16.8%`` -> ``16.8``.

Words are lower-cased, accent-folded and lightly singularised. The same
function is used for documents and queries.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"      # 1,234 or 1,234.56
    r"|[^\W_]+(?:[-/.'\u2019][^\W_]+)*"  # words, decimals and compound identifiers
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_COMPOUND_SPLIT = re.compile(r"[-/.]")
MAX_COMPOUND_PARTS = 6

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves
""".split())


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _canonical_number(token: str) -> str:
    token = token.replace(",", "")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    if token.isdigit():
        return token.lstrip("0") or "0"
    return token


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _word(token: str) -> str | None:
    if token in STOPWORDS or (len(token) == 1 and token.isalpha()):
        return None
    return _stem(token) if token.isalpha() else token


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(_fold(text)):
        token = match.group(0)
        if "'" in token or "\u2019" in token:  # possessives and contractions: keep the stem
            token = re.split(r"['\u2019]", token)[0]
        if _NUMBER.fullmatch(token.replace(",", "")):
            tokens.append(_canonical_number(token))
            continue
        parts = [p for p in _COMPOUND_SPLIT.split(token) if p]
        if len(parts) == 1:
            word = _word(token)
            if word:
                tokens.append(word)
            continue
        tokens.append(token)
        seps = _COMPOUND_SPLIT.findall(token)
        if len(parts) <= MAX_COMPOUND_PARTS:
            for size in range(2, len(parts)):
                for start in range(len(parts) - size + 1):
                    piece = parts[start]
                    for i in range(start + 1, start + size):
                        piece += seps[i - 1] + parts[i]
                    tokens.append(piece)
        for part in parts:
            word = _word(part)
            if word:
                tokens.append(_canonical_number(word) if word.isdigit() else word)
    return tokens
