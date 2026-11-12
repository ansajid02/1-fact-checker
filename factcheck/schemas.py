"""JSON contracts. Agents exchange these, never free text."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["TRUE", "FALSE", "NOT_ENOUGH_INFO"]
LABELS: list[str] = ["TRUE", "FALSE", "NOT_ENOUGH_INFO"]
Lang = Literal["en", "ar"]

# Shared by the single agent and the team judge. Without these rules the model treats
# "not mentioned" as FALSE and partial matches as TRUE, and almost never says NOT_ENOUGH_INFO.
LABEL_RULES = """How to choose the label. Check EVERY part of the claim (who, what, when, where, numbers,
words like "only", "first", "never"):
- TRUE: the evidence explicitly states every part of the claim. A partial match is not enough:
  evidence "the album was released in 2004" does not support "the album was released in March 2004".
- FALSE: the evidence explicitly states something that cannot be true at the same time as the claim,
  e.g. a different date or number for the same event.
- NOT_ENOUGH_INFO: everything else, in particular when
  - one part of the claim is not mentioned in the evidence, even if the rest matches;
  - the evidence describes the subject differently but both could be true
    (being described as a novelist does not mean someone is not also a poet);
  - the claim sounds unlikely or absurd, but no evidence directly contradicts it;
  - deciding would need your own knowledge or a guess.
Not mentioned is NOT the same as contradicted."""


# ---------------------------------------------------------------- tool outputs


class Sentence(BaseModel):
    idx: int
    text: str


class Document(BaseModel):
    """A source the agent has read. Only these sentences can be cited."""

    source_id: str  # e.g. "wiki:en:Burj Khalifa", "openalex:W2741809807"
    kind: Literal["wikipedia", "paper", "kb"]
    title: str
    url: str
    lang: str
    sentences: list[Sentence]
    total_sentences: int


class SearchHit(BaseModel):
    title: str
    snippet: str
    lang: str


# ---------------------------------------------------------------- verdicts


class Citation(BaseModel):
    source_id: str = Field(description="source_id exactly as shown in an observation")
    sent_idx: int = Field(description="the [number] of the cited sentence")
    quote: str = Field(description="the cited sentence, copied exactly")


class CheckedCitation(Citation):
    valid: bool
    reason: str = ""


class VerdictDraft(BaseModel):
    """What the LLM returns when it decides."""

    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(description="1-3 sentences, in the claim's language")
    citations: list[Citation] = Field(default_factory=list)


class CheckResult(BaseModel):
    """Final output of one fact-check, plus bookkeeping for benchmarks."""

    claim: str
    mode: str
    label: Label | Literal["ERROR"]
    confidence: float = 0.0
    explanation: str = ""
    citations: list[CheckedCitation] = Field(default_factory=list)
    sources_read: list[str] = Field(default_factory=list)
    steps: int = 0
    error: str | None = None
    extra: dict = Field(default_factory=dict)

    @property
    def fake_citations(self) -> int:
        return sum(not c.valid for c in self.citations)
