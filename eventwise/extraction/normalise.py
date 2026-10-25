"""Text clean-up and number parsing shared by all parsers."""

from __future__ import annotations

import re
import unicodedata

from ..models import NormalisedCell

_WHITESPACE = re.compile(r"\s+")

_NUMBER = re.compile(
    r"""^
    (?P<open>\()?
    (?P<sign>[-\u2212\u2013])?
    (?P<currency>[£$€])?
    (?P<sign2>[-\u2212\u2013])?
    (?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    (?P<multiplier>k|m|bn)?
    (?P<percent>%)?
    (?P<close>\))?
    $""",
    re.VERBOSE | re.IGNORECASE,
)
_CURRENCY = {"£": "GBP", "$": "USD", "€": "EUR"}
_MULTIPLIER = {"k": 1e3, "m": 1e6, "bn": 1e9}


def clean_text(text: str | None) -> str:
    """NFKC-normalise (expands ligatures) and collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    return _WHITESPACE.sub(" ", text).strip()


def parse_number(text: str | None) -> tuple[float, str | None] | None:
    """Parse a cell that consists of a single number, e.g. "(£40,400)" -> (-40400.0, "GBP").

    Returns None when the text is not purely numeric, so mixed cells such as
    "9,420 (Sat 18 July)" keep only their text.
    """
    if not text:
        return None
    compact = clean_text(text).replace(" ", "")
    match = _NUMBER.match(compact)
    if not match:
        return None
    if bool(match["open"]) != bool(match["close"]):
        return None
    value = float(match["number"].replace(",", ""))
    if match["multiplier"]:
        if not match["currency"]:  # "10 m" is far more likely metres than millions
            return None
        value *= _MULTIPLIER[match["multiplier"].lower()]
    if match["open"] or match["sign"] or match["sign2"]:
        value = -value
    unit = "%" if match["percent"] else _CURRENCY.get(match["currency"] or "")
    return value, unit


def normalise_cell(raw: str | None) -> NormalisedCell | None:
    if raw is None:
        return None
    parsed = parse_number(raw)
    if parsed is None:
        return NormalisedCell(text=clean_text(raw))
    return NormalisedCell(text=clean_text(raw), value=parsed[0], unit=parsed[1])
