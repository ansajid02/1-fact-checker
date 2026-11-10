"""Text embeddings, through one function (like llm.generate), traced.

    vecs = embed.embed(["text", ...], kind="document")   # -> float32 array, unit-normalised

Backends (EMBED_BACKEND in .env):
- "openrouter": e.g. baai/bge-m3 (multilingual, good Arabic) - default
- "vertex":     Vertex AI embedding models (google-genai SDK)
- "local":      fastembed on CPU, no API (`uv sync --extra local-embed`)
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Callable, Literal

import httpx
import numpy as np

from . import trace
from .config import settings

Kind = Literal["document", "query"]
BATCH = 64


class EmbedError(RuntimeError):
    pass


def _prefix(model: str, kind: Kind, texts: list[str]) -> list[str]:
    if "e5" in model.lower():  # e5 models expect role prefixes
        return [f"{'query' if kind == 'query' else 'passage'}: {t}" for t in texts]
    return texts


def _openrouter(model: str, texts: list[str], kind: Kind) -> tuple[np.ndarray, int, float | None]:
    from .llm import _openrouter_client

    delay = 2.0
    for attempt in range(5):
        r = _openrouter_client().post("/embeddings", json={"model": model, "input": _prefix(model, kind, texts)})
        if r.status_code not in (408, 429, 500, 502, 503, 504) or attempt == 4:
            break
        time.sleep(delay)
        delay = min(delay * 2, 30)
    if r.status_code != 200:
        raise EmbedError(f"OpenRouter embeddings HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    if "data" not in data:
        raise EmbedError(f"OpenRouter embeddings error: {str(data)[:300]}")
    rows = sorted(data["data"], key=lambda d: d["index"])
    usage = data.get("usage") or {}
    return np.array([d["embedding"] for d in rows], dtype=np.float32), usage.get("prompt_tokens", 0), usage.get("cost")


def _vertex(model: str, texts: list[str], kind: Kind) -> tuple[np.ndarray, int, float | None]:
    from google.genai import types

    from .llm import _vertex_client

    task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
    resp = _vertex_client().models.embed_content(
        model=model, contents=texts, config=types.EmbedContentConfig(task_type=task))
    return np.array([e.values for e in resp.embeddings], dtype=np.float32), 0, None


@lru_cache(maxsize=2)
def _fastembed(model: str):
    try:
        from fastembed import TextEmbedding
    except ImportError as e:
        raise EmbedError("local embeddings need fastembed: uv sync --extra local-embed") from e
    return TextEmbedding(model_name=model, cache_dir=str(settings().local_models_dir / "fastembed"))


def _local(model: str, texts: list[str], kind: Kind) -> tuple[np.ndarray, int, float | None]:
    m = _fastembed(model)
    vecs = m.query_embed(texts) if kind == "query" else m.embed(texts)
    return np.array(list(vecs), dtype=np.float32), 0, 0.0


BACKENDS: dict[str, Callable] = {"openrouter": _openrouter, "vertex": _vertex, "local": _local}


def current_model(backend: str | None = None) -> tuple[str, str]:
    s = settings()
    backend = backend or s.embed_backend
    model = {"openrouter": s.embed_model, "vertex": s.vertex_embed_model, "local": s.local_embed_model}.get(backend, "")
    if not model:
        raise EmbedError(f"no embedding model configured for backend '{backend}'")
    return backend, model


def embed(texts: list[str], kind: Kind = "document", *, backend: str | None = None,
          model: str | None = None, agent: str = "embed") -> np.ndarray:
    if model is None:
        backend, model = current_model(backend)
    backend = backend or settings().embed_backend
    out = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        with trace.span("embed", agent, backend=backend, model=model, n=len(batch), embed_kind=kind) as sp:
            vecs, tokens, cost = BACKENDS[backend](model, batch, kind)
            sp.update(input_tokens=tokens, output_tokens=0, cost_usd=cost)
        out.append(vecs)
    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.vstack(out)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1, norms)
