"""Text utilities: sentence splitting and normalisation (English + Arabic)."""

from __future__ import annotations

import re
import unicodedata

# Arabic diacritics (tashkeel), superscript alef, and tatweel
_AR_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي"})
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """For matching/search: NFKC, lowercase, Arabic letter normalisation, no punctuation."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _AR_DIACRITICS.sub("", text).translate(_AR_MAP)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "inc", "ltd", "co", "corp", "no", "vs",
    "etc", "e.g", "i.e", "u.s", "u.k", "u.n", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "approx", "ca", "c", "est", "mt", "ft", "gen", "gov",
}
# candidate boundary: end punctuation (+ closing quotes/brackets) followed by whitespace
_BOUNDARY = re.compile(r"[.!?؟]+[\"'”’)\]]*\s+")
_HEADER = re.compile(r"^\s*=+.*=+\s*$")


def _starts_sentence(ch: str) -> bool:
    return ch.isupper() or ch.isdigit() or ch in "\"“'(['" or "؀" <= ch <= "ۿ"


def _split_paragraph(par: str) -> list[str]:
    out, start = [], 0
    for m in _BOUNDARY.finditer(par):
        end = m.end()
        if end >= len(par) or not _starts_sentence(par[end]):
            continue
        before = par[start:m.start() + 1].rstrip(".!?؟").split()
        last = before[-1].lower().strip("(\"'") if before else ""
        if last in _ABBREV or (len(last) == 1 and last.isalpha()):  # "Dr." / initials "J. R."
            continue
        out.append(par[start:end].strip())
        start = end
    if par[start:].strip():
        out.append(par[start:].strip())
    return out


def split_sentences(text: str) -> list[str]:
    """Plain-text article -> sentences. Drops wiki section headers and fragments."""
    sentences: list[str] = []
    for par in text.splitlines():
        if not par.strip() or _HEADER.match(par):
            continue
        sentences.extend(s for s in _split_paragraph(par.strip()) if len(s) > 3)
    return sentences


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("؀" <= c <= "ۿ" for c in letters) / len(letters) > 0.5
