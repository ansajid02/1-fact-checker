"""Hybrid retrieval over the local KB: BM25 + dense embeddings, fused with RRF, optional LLM rerank.

KB_MODE   = hybrid (default) | bm25 | dense      (hybrid falls back to bm25 if no embeddings)
KB_RERANK = none (default) | llm                  (`make kb-eval RERANK=1` shows whether it helps)
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from urllib.parse import quote

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from rich.console import Console

from agentcore import embed, llm
from agentcore.trace import traced_tool

from ..schemas import Document, Sentence
from ..text import normalize
from .build import KB_PAGES, load_pages
from .index import CHUNKS, META, VECTORS, Chunk

RRF_K = 60
CANDIDATES = 50
RERANK_POOL = 20
console = Console(stderr=True)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


class RerankOut(BaseModel):
    ranked: list[int] = Field(description="candidate numbers, most relevant first")


def rrf(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, float, dict[str, int]]]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ids in rankings.items():
        for rank, cid in enumerate(ids, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(cid, {})[name] = rank
    return sorted(((cid, s, ranks[cid]) for cid, s in scores.items()), key=lambda x: -x[1])


class KnowledgeBase:
    def __init__(self, mode: str | None = None) -> None:
        if not CHUNKS.exists():
            raise FileNotFoundError(f"{CHUNKS} not found. Run `make kb` first.")
        self.chunks = [Chunk.model_validate_json(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l]
        self.pages = load_pages(KB_PAGES)
        self.meta = json.loads(META.read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([normalize(c.text).split() or ["_"] for c in self.chunks])
        self.vectors = np.load(VECTORS) if VECTORS.exists() else None
        self.mode = mode or os.environ.get("KB_MODE", "hybrid")
        if self.mode in ("hybrid", "dense") and self.vectors is None:
            console.print("[yellow]KB has no embeddings: using BM25 only (run `make kb` with credits).[/]")
            self.mode = "bm25"

    def bm25_top(self, query: str, n: int = CANDIDATES) -> list[int]:
        scores = self.bm25.get_scores(normalize(query).split())
        top = np.argsort(-scores)[:n]
        return [int(i) for i in top if scores[i] > 0]

    def dense_top(self, query: str, n: int = CANDIDATES) -> list[int]:
        qv = embed.embed([query], kind="query", backend=self.meta["embed_backend"],
                         model=self.meta["embed_model"], agent="kb.query")[0]
        sims = self.vectors @ qv
        return [int(i) for i in np.argsort(-sims)[:n]]

    def search(self, query: str, k: int = 8, rerank: str = "none", mode: str | None = None) -> list[Hit]:
        mode = mode or self.mode
        rankings: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid"):
            rankings["bm25"] = self.bm25_top(query)
        if mode in ("dense", "hybrid"):
            rankings["dense"] = self.dense_top(query)
        fused = rrf(rankings)
        hits = [Hit(self.chunks[cid], score, r) for cid, score, r in fused[:max(k, RERANK_POOL)]]
        if rerank == "llm" and len(hits) > 1:
            hits = self._llm_rerank(query, hits[:RERANK_POOL])
        return hits[:k]

    def _llm_rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        listing = "\n".join(f"{i}. {h.chunk.text[:400]}" for i, h in enumerate(hits))
        out = llm.generate(
            f"Query: {query}\n\nCandidates:\n{listing}\n\nRank the candidates by how useful they are "
            "for verifying the query (most useful first). Return candidate numbers only.",
            schema=RerankOut, agent="kb.rerank")
        order = [i for i in dict.fromkeys(out.ranked) if 0 <= i < len(hits)]
        order += [i for i in range(len(hits)) if i not in order]
        return [hits[i] for i in order]

    def to_documents(self, hits: list[Hit]) -> list[Document]:
        """Group hits by page; always include the page's first sentence (it defines the entity)."""
        by_title: dict[str, set[int]] = {}
        for h in hits:
            by_title.setdefault(h.chunk.title, set()).update(h.chunk.sent_ids)
        docs = []
        for title, ids in by_title.items():
            page = self.pages[title]
            first = page.sentences[0].idx
            wanted = ids | {first}
            docs.append(Document(
                source_id=f"fever:{title}", kind="kb", title=title,
                url=f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                lang="en", total_sentences=len(page.sentences),
                sentences=[Sentence(idx=s.idx, text=s.text) for s in page.sentences if s.idx in wanted],
            ))
        return docs


_kb: KnowledgeBase | None = None
_kb_lock = threading.Lock()


def get_kb() -> KnowledgeBase:
    global _kb
    with _kb_lock:
        if _kb is None:
            _kb = KnowledgeBase()
        return _kb


def kb_available() -> bool:
    return CHUNKS.exists()


@traced_tool("kb_search")
def kb_search(query: str, k: int = 6, rerank: str | None = None) -> list[Document]:
    """Search the local KB (FEVER 2017 Wikipedia intros). Returns citable sentences, grouped by page."""
    kb = get_kb()
    return kb.to_documents(kb.search(query, k=k, rerank=rerank or os.environ.get("KB_RERANK", "none")))
