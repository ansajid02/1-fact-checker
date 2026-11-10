"""Build the local knowledge base (and the LoRA evidence pages) from the FEVER 2017 dump.

One pass over the 5.4M pages selects:
  KB pages   = gold evidence pages of the 300 test claims
             + pages whose title appears in a test claim (hard distractors, classic FEVER doc retrieval)
             + N random pages (easy distractors)
  LoRA pages = gold evidence pages of the train sample + title-matched pages for its NOT ENOUGH INFO claims

Honest caveat (goes in the README): the KB is "closed world" - it contains the gold pages,
so it measures retrieval + reasoning, not open-web search.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict

from rich.console import Console

from ..fever import RAW_WIKI, TRAIN_SAMPLE_PATH, base_title, load_sample, title_ngrams
from ..paths import DATA_DIR
from .wikidump import WikiPage, iter_lines, parse_line, raw_title

KB_DIR = DATA_DIR / "kb"
LORA_DIR = DATA_DIR / "lora"
KB_PAGES = KB_DIR / "pages.jsonl"
LORA_PAGES = LORA_DIR / "pages.jsonl"

console = Console()


def write_pages(path, pages: dict[str, WikiPage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for title in sorted(pages):
            f.write(pages[title].model_dump_json() + "\n")


def load_pages(path) -> dict[str, WikiPage]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make kb` first.")
    pages = (WikiPage.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l)
    return {p.title: p for p in pages}


def build_pages(n_random: int = 3000, max_matched_per_claim: int = 10, seed: int = 42,
                zip_path=RAW_WIKI) -> dict:
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found. Run `make data` (downloads the 1.7 GB dump).")
    test = load_sample()
    train = load_sample(TRAIN_SAMPLE_PATH) if TRAIN_SAMPLE_PATH.exists() else []
    kb_gold = {t for c in test for t in c.evidence_titles}
    lora_gold = {t for c in train for t in c.evidence_titles}

    grams: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for i, c in enumerate(test):
        for g, n in title_ngrams(c.claim).items():
            grams[g].append(("kb", i, n))
    for i, c in enumerate(train):
        if c.label == "NOT_ENOUGH_INFO":
            for g, n in title_ngrams(c.claim).items():
                grams[g].append(("lora", i, n))

    kb_pages: dict[str, WikiPage] = {}
    lora_pages: dict[str, WikiPage] = {}
    matched_pages: dict[str, WikiPage] = {}
    matched: dict[tuple[str, int], list[tuple[int, bool, str]]] = defaultdict(list)
    reservoir: list[str] = []
    rng = random.Random(seed)
    seen, t0 = 0, time.time()

    for line in iter_lines(zip_path):
        title = raw_title(line)
        if not title:
            continue
        seen += 1
        hits = grams.get(base_title(title))
        if title in kb_gold or title in lora_gold or hits:
            page = parse_line(line)
            if page:
                if title in kb_gold:
                    kb_pages[title] = page
                if title in lora_gold:
                    lora_pages[title] = page
                if hits:
                    matched_pages[title] = page
                    for group, i, n in hits:
                        matched[(group, i)].append((n, "(" in title, title))
        if len(reservoir) < n_random:
            reservoir.append(line)
        else:
            j = rng.randrange(seen)
            if j < n_random:
                reservoir[j] = line
        if seen % 500_000 == 0:
            console.print(f"  scanned {seen:,} pages ({time.time() - t0:.0f}s)")

    # longest title match first, plain titles before "(disambiguated)" ones
    for (group, _), cands in matched.items():
        target = kb_pages if group == "kb" else lora_pages
        for _, _, title in sorted(cands, key=lambda x: (-x[0], x[1], x[2]))[:max_matched_per_claim]:
            target.setdefault(title, matched_pages[title])
    n_matched_kb = len(kb_pages)
    for line in reservoir:
        page = parse_line(line)
        if page:
            kb_pages.setdefault(page.title, page)

    write_pages(KB_PAGES, kb_pages)
    write_pages(LORA_PAGES, lora_pages)
    stats = {
        "pages_scanned": seen,
        "kb_pages": len(kb_pages),
        "kb_gold_pages_found": sum(t in kb_pages for t in kb_gold),
        "kb_gold_pages_wanted": len(kb_gold),
        "kb_gold_plus_matched": n_matched_kb,
        "lora_pages": len(lora_pages),
        "n_random": n_random,
        "max_matched_per_claim": max_matched_per_claim,
        "seed": seed,
        "seconds": round(time.time() - t0),
    }
    (KB_DIR / "build_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
