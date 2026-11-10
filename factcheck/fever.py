"""FEVER test set: a balanced, fixed-seed sample from the labelled shared-task dev split.

FEVER (Thorne et al., 2018) is distributed by fever.ai; the claims are derived from a
June 2017 Wikipedia snapshot (CC BY-SA). Live Wikipedia has changed since then, so a
correct answer from today's Wikipedia can occasionally disagree with the gold label.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict

import httpx
from pydantic import BaseModel

from .paths import DATA_DIR, RAW_DIR
from .schemas import Label

FEVER_DEV_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"
FEVER_TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"
FEVER_WIKI_URL = "https://fever.ai/download/fever/wiki-pages.zip"
RAW_DEV = RAW_DIR / "shared_task_dev.jsonl"
RAW_TRAIN = RAW_DIR / "train.jsonl"
RAW_WIKI = RAW_DIR / "wiki-pages.zip"
SAMPLE_PATH = DATA_DIR / "fever_300.jsonl"
TRAIN_SAMPLE_PATH = DATA_DIR / "fever_train_sample.jsonl"

LABEL_MAP: dict[str, Label] = {
    "SUPPORTS": "TRUE",
    "REFUTES": "FALSE",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
}
_TITLE_TOKENS = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "-COLON-": ":",
}


class FeverClaim(BaseModel):
    id: int
    claim: str
    label: Label
    fever_label: str
    evidence_sets: list[list[tuple[str, int]]]  # each set: [(page title, sentence id), ...]
    dataset: str = "fever"

    @property
    def evidence_titles(self) -> list[str]:
        return sorted({t for s in self.evidence_sets for t, _ in s})


def decode_title(title: str) -> str:
    for token, char in _TITLE_TOKENS.items():
        title = title.replace(token, char)
    return title.replace("_", " ")


def download_file(url: str, dest, force: bool = False) -> None:
    if dest.exists() and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    tmp.rename(dest)


def download(force: bool = False) -> None:
    download_file(FEVER_DEV_URL, RAW_DEV, force)


def _parse(row: dict) -> FeverClaim:
    sets = []
    for ev_set in row["evidence"]:
        pairs = [(decode_title(t), int(sid)) for _, _, t, sid in ev_set if t is not None]
        if pairs:
            sets.append(pairs)
    return FeverClaim(id=row["id"], claim=row["claim"], label=LABEL_MAP[row["label"]],
                      fever_label=row["label"], evidence_sets=sets)


def _balanced(raw_path, per_class: int, seed: int) -> list[FeverClaim]:
    """Balanced sample, interleaved by label so any prefix (--limit N) stays balanced."""
    by_label: dict[str, list[FeverClaim]] = defaultdict(list)
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            c = _parse(json.loads(line))
            by_label[c.label].append(c)
    rng = random.Random(seed)
    picked = {}
    for label in LABEL_MAP.values():
        pool = sorted(by_label[label], key=lambda c: c.id)
        if len(pool) < per_class:
            raise ValueError(f"only {len(pool)} {label} claims available")
        picked[label] = rng.sample(pool, per_class)
    return [picked[lab][i] for i in range(per_class) for lab in LABEL_MAP.values()]


def _write(path, claims: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(c.model_dump_json() + "\n" for c in claims), encoding="utf-8")


def make_sample(per_class: int = 100, seed: int = 42) -> list[FeverClaim]:
    """The test set: from the labelled shared-task dev split."""
    download()
    sample = _balanced(RAW_DEV, per_class, seed)
    _write(SAMPLE_PATH, sample)
    return sample


def make_train_sample(per_class: int = 1500, seed: int = 42) -> list[FeverClaim]:
    """LoRA training claims: from the train split (disjoint from dev, so no test leakage)."""
    download_file(FEVER_TRAIN_URL, RAW_TRAIN)
    sample = _balanced(RAW_TRAIN, per_class, seed)
    _write(TRAIN_SAMPLE_PATH, sample)
    return sample


def load_sample(path=SAMPLE_PATH) -> list[FeverClaim]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    return [FeverClaim.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------- title matching

_STOP = {"the", "a", "an", "of", "in", "on", "and", "or", "is", "was", "to", "by", "for", "at", "it",
         "he", "she", "they", "this", "that", "as", "with", "from", "his", "her", "there", "be"}
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_TOKEN = re.compile(r"[\w'’.\-]+", re.UNICODE)


def base_title(title: str) -> str:
    """'Soul Food (film)' -> 'soul food' (lowercase, no disambiguation suffix)."""
    return _PAREN.sub("", title).strip().lower()


def title_ngrams(claim: str, max_n: int = 6) -> dict[str, int]:
    """Capitalised n-grams of a claim that could be Wikipedia titles -> their length in tokens.

    This is the classic FEVER document-retrieval trick: entity names in claims are page titles.
    """
    tokens = [t.strip(".") for t in _TOKEN.findall(claim)]
    tokens = [t for t in tokens if t]
    out: dict[str, int] = {}
    for i, first in enumerate(tokens):
        if not (first[:1].isupper() or first[:1].isdigit()):
            continue
        for n in range(1, max_n + 1):
            gram = tokens[i:i + n]
            if len(gram) < n:
                break
            if all(g.lower() in _STOP for g in gram) or len(" ".join(gram)) < 3:
                continue
            out[" ".join(gram).lower()] = n
    return out
