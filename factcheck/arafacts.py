"""AraFacts (Sheikh Ali et al., WANLP 2021): Arabic claims verified by professional fact-checkers.

Source: https://gitlab.com/bigirqu/AraFacts (v2, 18,598 claims). Licence: CC BY-NC 4.0 (non-commercial).

Sampling rules (documented in the README):
- text claims only (claim_type 0); image/video claims cannot be checked from text
- questions dropped ("هل ...؟"): they have no truth value as written
- labels: TRUE -> TRUE, FALSE -> FALSE, Unverifiable -> NOT_ENOUGH_INFO;
  Partly-false and Sarcasm are left out (no clean 3-way mapping)
- the `description` column is NEVER used: it contains the fact-checker's answer
Most AraFacts claims are social-media rumours that Wikipedia does not cover, so expect many
NOT_ENOUGH_INFO answers: report that honestly.
"""

from __future__ import annotations

import csv
import random

from pydantic import BaseModel

from .fever import download_file
from .paths import DATA_DIR, RAW_DIR
from .schemas import Label

ARAFACTS_URL = "https://gitlab.com/bigirqu/AraFacts/-/raw/master/Dataset/AraFacts%202/AraFacts%202.csv"
RAW_CSV = RAW_DIR / "arafacts.csv"
SAMPLE_PATH = DATA_DIR / "arafacts_sample.jsonl"
LABELS = {"TRUE": "TRUE", "FALSE": "FALSE", "UNVERIFIABLE": "NOT_ENOUGH_INFO"}


class AraFactsClaim(BaseModel):
    id: str
    claim: str
    label: Label
    source: str
    date: str
    url: str
    dataset: str = "arafacts"
    evidence_sets: list[list[tuple[str, int]]] = []

    @property
    def evidence_titles(self) -> list[str]:
        return []


def is_question(text: str) -> bool:
    t = text.strip()
    return t.endswith(("؟", "?")) or t.startswith("هل ")


def load_rows() -> list[AraFactsClaim]:
    download_file(ARAFACTS_URL, RAW_CSV)
    out = []
    with RAW_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = LABELS.get(r["normalized_label"].strip().upper())
            claim = r["claim"].strip()
            if r["claim_type"] != "0" or not label or not claim or is_question(claim):
                continue
            out.append(AraFactsClaim(id=r["ClaimID"], claim=claim, label=label, source=r["source"],
                                     date=r["date"][:10], url=r["source_url"]))
    return out


def make_sample(per_class: int = 50, max_nei: int = 25, seed: int = 42) -> list[AraFactsClaim]:
    rows = load_rows()
    rng = random.Random(seed)
    by = {lab: sorted((r for r in rows if r.label == lab), key=lambda r: r.id) for lab in LABELS.values()}
    picked = {
        "TRUE": rng.sample(by["TRUE"], min(per_class, len(by["TRUE"]))),
        "FALSE": rng.sample(by["FALSE"], min(per_class, len(by["FALSE"]))),
        "NOT_ENOUGH_INFO": rng.sample(by["NOT_ENOUGH_INFO"], min(max_nei, len(by["NOT_ENOUGH_INFO"]))),
    }
    sample, i = [], 0
    while any(i < len(v) for v in picked.values()):  # interleave so --limit stays mixed
        sample += [v[i] for v in picked.values() if i < len(v)]
        i += 1
    SAMPLE_PATH.write_text("".join(c.model_dump_json() + "\n" for c in sample), encoding="utf-8")
    return sample


def load_sample() -> list[AraFactsClaim]:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"{SAMPLE_PATH} not found. Run `make data` first.")
    return [AraFactsClaim.model_validate_json(l) for l in SAMPLE_PATH.read_text(encoding="utf-8").splitlines() if l]
