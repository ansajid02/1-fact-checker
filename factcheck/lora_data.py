"""LoRA judge data (step 11).

`make lora-data` writes:
  data/lora/train.jsonl, val.jsonl   chat-format SFT examples from the FEVER *train* split
  data/lora/judge_eval.jsonl         the 300 *test* claims with fixed evidence, for `make judge-eval`

Evidence construction (identical for training and evaluation, so judges are compared fairly):
- SUPPORTS / REFUTES: the first gold evidence set + up to 2 other sentences from the gold page(s)
  (so the model must find the relevant sentence, not just trust whatever it is shown)
- NOT ENOUGH INFO: the first sentences of pages whose title appears in the claim (the entity is
  there, the fact is not) - or of a random page if none matches
"""

from __future__ import annotations

import json
import random

from .fever import TRAIN_SAMPLE_PATH, FeverClaim, base_title, load_sample, title_ngrams
from .judge_lora import SYSTEM, prompt, target
from .kb.build import KB_PAGES, LORA_DIR, LORA_PAGES, load_pages
from .kb.wikidump import WikiPage

TRAIN = LORA_DIR / "train.jsonl"
VAL = LORA_DIR / "val.jsonl"
JUDGE_EVAL = LORA_DIR / "judge_eval.jsonl"


def build_evidence(c: FeverClaim, pages: dict[str, WikiPage], by_base: dict[str, list[str]],
                   rng: random.Random) -> list[str] | None:
    if c.evidence_sets:
        sents = []
        for title, idx in c.evidence_sets[0]:
            page = pages.get(title)
            text = next((s.text for s in page.sentences if s.idx == idx), None) if page else None
            if text is None:
                return None  # gold sentence missing from the dump: skip the example
            sents.append(f"{title}: {text}")
        gold_ids = set(map(tuple, c.evidence_sets[0]))
        extra = [f"{t}: {s.text}" for t in c.evidence_titles if t in pages
                 for s in pages[t].sentences if (t, s.idx) not in gold_ids]
        sents += rng.sample(extra, min(2, len(extra)))
        rng.shuffle(sents)
        return sents
    titles = [t for g in title_ngrams(c.claim) for t in by_base.get(g, [])][:2]
    if not titles:
        titles = [rng.choice(sorted(pages))]
    return [f"{t}: {s.text}" for t in titles for s in pages[t].sentences[:2]]


def _index(pages: dict[str, WikiPage]) -> dict[str, list[str]]:
    by_base: dict[str, list[str]] = {}
    for t in sorted(pages):
        by_base.setdefault(base_title(t), []).append(t)
    return by_base


def build(val_fraction: float = 0.05, seed: int = 42) -> dict:
    rng = random.Random(seed)
    stats = {}

    train_pages = load_pages(LORA_PAGES)
    by_base = _index(train_pages)
    examples = []
    for c in load_sample(TRAIN_SAMPLE_PATH):
        ev = build_evidence(c, train_pages, by_base, rng)
        if ev is None:
            continue
        examples.append({"id": c.id, "label": c.label, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt(c.claim, ev)},
            {"role": "assistant", "content": target(c.label)},
        ]})
    rng.shuffle(examples)
    n_val = int(len(examples) * val_fraction)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in ((VAL, examples[:n_val]), (TRAIN, examples[n_val:])):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats |= {"train": len(examples) - n_val, "val": n_val}

    kb_pages = load_pages(KB_PAGES)
    kb_base = _index(kb_pages)
    evals = []
    for c in load_sample():
        ev = build_evidence(c, kb_pages, kb_base, rng)
        if ev is not None:
            evals.append({"id": c.id, "claim": c.claim, "label": c.label, "evidence": ev})
    JUDGE_EVAL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in evals), encoding="utf-8")
    stats["judge_eval"] = len(evals)
    return stats
