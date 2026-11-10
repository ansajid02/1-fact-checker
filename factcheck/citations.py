"""Hallucination control: every citation must point to a real sentence the agent actually read."""

from __future__ import annotations

from difflib import SequenceMatcher

from .schemas import CheckedCitation, Citation, Document
from .text import normalize

MATCH_THRESHOLD = 0.85


class EvidenceStore:
    """Every sentence shown to the agent during one check, keyed by (source_id, sent_idx)."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[int, str]] = {}
        self.titles: dict[str, str] = {}

    def add(self, doc: Document) -> None:
        self.docs.setdefault(doc.source_id, {}).update({s.idx: s.text for s in doc.sentences})
        self.titles[doc.source_id] = doc.title

    def sources(self) -> list[str]:
        return list(self.docs)


def _matches(quote: str, sentence: str) -> bool:
    q, s = normalize(quote), normalize(sentence)
    if not q:
        return False
    return q in s or SequenceMatcher(None, q, s).ratio() >= MATCH_THRESHOLD


def verify(citations: list[Citation], store: EvidenceStore) -> list[CheckedCitation]:
    checked = []
    for c in citations:
        sentences = store.docs.get(c.source_id)
        if sentences is None:
            valid, reason = False, "source was never read"
        elif c.sent_idx not in sentences:
            valid, reason = False, "sentence index was never read"
        elif _matches(c.quote, sentences[c.sent_idx]):
            valid, reason = True, ""
        elif any(_matches(c.quote, s) for s in sentences.values()):
            valid, reason = False, "quote exists but under a different sentence index"
        else:
            valid, reason = False, "quote does not match the sentence"
        checked.append(CheckedCitation(**c.model_dump(), valid=valid, reason=reason))
    return checked
