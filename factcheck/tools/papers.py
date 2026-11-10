"""Paper search via OpenAlex (covers arXiv and most journals). Abstracts are citable.

OpenAlex works without a key on a small shared daily budget. For your own budget, get a
free key at https://openalex.org/settings/api and set OPENALEX_API_KEY in .env.
"""

from __future__ import annotations

from agentcore.config import settings
from agentcore.trace import traced_tool

from ..schemas import Document, Sentence
from ..text import split_sentences
from .http import ToolError, get_json

OPENALEX = "https://api.openalex.org/works"
MAX_ABSTRACT_SENTENCES = 15


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = {pos: word for word, plist in index.items() for pos in plist}
    return " ".join(positions[i] for i in sorted(positions))


@traced_tool("paper_search")
def paper_search(query: str, limit: int = 3) -> list[Document]:
    if not query.strip():
        raise ToolError("empty query")
    params = {
        "search": query, "per_page": limit,
        "select": "id,display_name,publication_year,doi,abstract_inverted_index",
    }
    if settings().openalex_api_key:
        params["api_key"] = settings().openalex_api_key
    data = get_json(OPENALEX, params)
    docs = []
    for w in data.get("results", []):
        abstract = abstract_from_inverted_index(w.get("abstract_inverted_index"))
        sentences = split_sentences(abstract)
        work_id = (w.get("id") or "").rsplit("/", 1)[-1]
        title = w.get("display_name") or "(untitled)"
        year = w.get("publication_year")
        docs.append(Document(
            source_id=f"openalex:{work_id}",
            kind="paper",
            title=f"{title} ({year})" if year else title,
            url=w.get("doi") or w.get("id") or "",
            lang="en",
            sentences=[Sentence(idx=i, text=s) for i, s in enumerate(sentences[:MAX_ABSTRACT_SENTENCES])],
            total_sentences=len(sentences),
        ))
    return docs
