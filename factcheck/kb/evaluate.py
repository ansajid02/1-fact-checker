"""`make kb-eval`: retrieval quality on the verifiable test claims (gold evidence known).

page recall@k     - a gold evidence page is among the top-k distinct pages
evidence recall@k - a complete gold evidence set is inside the top-k chunks' sentences
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from agentcore import trace

from ..fever import load_sample
from ..paths import RESULTS_DIR
from .retriever import KnowledgeBase

console = Console()


def _pages(hits) -> list[str]:
    return list(dict.fromkeys(h.chunk.title for h in hits))


def evaluate(ks=(1, 3, 5, 10), rerank: bool = False) -> dict:
    kb = KnowledgeBase()
    claims = [c for c in load_sample() if c.evidence_sets]
    configs = ["bm25"] + (["dense", "hybrid"] if kb.vectors is not None else [])
    if rerank:
        configs.append(f"{configs[-1]}+llm-rerank")
    results = {}
    with trace.run("kb-eval"), trace.collect() as events:
        for cfg in configs:
            mode, _, rr = cfg.partition("+")
            page_hits = {k: 0 for k in ks}
            ev_hits = {k: 0 for k in ks}
            for c in claims:
                hits = kb.search(c.claim, k=max(ks), mode=mode, rerank="llm" if rr else "none")
                gold = set(c.evidence_titles)
                for k in ks:
                    page_hits[k] += bool(gold & set(_pages(hits)[:k]))
                    got = {(h.chunk.title, s) for h in hits[:k] for s in h.chunk.sent_ids}
                    ev_hits[k] += any(all(tuple(p) in got for p in es) for es in c.evidence_sets)
            results[cfg] = {"page_recall": {k: v / len(claims) for k, v in page_hits.items()},
                            "evidence_recall": {k: v / len(claims) for k, v in ev_hits.items()}}
    cost = trace.summarize(events)["total"]["cost_usd"]
    t = Table(title=f"KB retrieval on {len(claims)} verifiable claims ({kb.meta['n_chunks']} chunks, "
                    f"embeddings: {kb.meta.get('embed_model') or 'none'})")
    t.add_column("config")
    for k in ks:
        t.add_column(f"page@{k}", justify="right")
    for k in ks:
        t.add_column(f"evid@{k}", justify="right")
    for cfg, r in results.items():
        t.add_row(cfg, *[f"{r['page_recall'][k]:.0%}" for k in ks], *[f"{r['evidence_recall'][k]:.0%}" for k in ks])
    console.print(t)
    console.print(f"[dim]query embedding / rerank cost: ${cost:.4f}[/dim]")
    out = {"n_claims": len(claims), "meta": kb.meta, "results": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "kb_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
