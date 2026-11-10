"""Chunk the KB pages and embed the chunks (one-off, stored as local files - no vector DB)."""

from __future__ import annotations

import json

import numpy as np
from pydantic import BaseModel

from agentcore import embed, trace

from .build import KB_DIR, KB_PAGES, load_pages

CHUNKS = KB_DIR / "chunks.jsonl"
VECTORS = KB_DIR / "embeddings.npy"
META = KB_DIR / "index_meta.json"


class Chunk(BaseModel):
    id: int
    title: str
    sent_ids: list[int]
    text: str  # "Title: s1 s2 s3" - the title resolves pronouns in FEVER sentences


def make_chunks(window: int = 3) -> list[Chunk]:
    chunks: list[Chunk] = []
    for title, page in sorted(load_pages(KB_PAGES).items()):
        for i in range(0, len(page.sentences), window):
            sents = page.sentences[i:i + window]
            chunks.append(Chunk(id=len(chunks), title=title, sent_ids=[s.idx for s in sents],
                                text=f"{title}: " + " ".join(s.text for s in sents)))
    return chunks


def build_index(window: int = 3, with_embeddings: bool = True) -> dict:
    chunks = make_chunks(window)
    CHUNKS.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    meta: dict = {"n_chunks": len(chunks), "window": window, "embed_backend": None, "embed_model": None}
    if with_embeddings:
        backend, model = embed.current_model()
        with trace.run("kb-index"), trace.collect() as events:
            vecs = embed.embed([c.text for c in chunks], kind="document", agent="kb.index")
        np.save(VECTORS, vecs)
        meta |= {"embed_backend": backend, "embed_model": model, "dim": int(vecs.shape[1]),
                 "embed_cost_usd": trace.summarize(events)["total"]["cost_usd"]}
    elif VECTORS.exists():
        VECTORS.unlink()  # stale vectors would no longer line up with the chunks
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
