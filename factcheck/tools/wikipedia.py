"""Wikipedia tools (English + Arabic), via the MediaWiki Action API."""

from __future__ import annotations

import html
import re
from urllib.parse import quote

from agentcore.trace import traced_tool

from ..schemas import Document, SearchHit, Sentence
from ..text import split_sentences
from .http import ToolError, get_json

_TAG = re.compile(r"<[^>]+>")


def _api(lang: str) -> str:
    if lang not in ("en", "ar"):
        raise ToolError(f"unsupported language '{lang}' (use 'en' or 'ar')")
    return f"https://{lang}.wikipedia.org/w/api.php"


@traced_tool("wikipedia_search")
def wikipedia_search(query: str, lang: str = "en", limit: int = 5) -> list[SearchHit]:
    """Full-text search. Returns titles + short snippets (snippets are NOT citable)."""
    if not query.strip():
        raise ToolError("empty query")
    data = get_json(_api(lang), {
        "action": "query", "list": "search", "srsearch": query, "srlimit": limit,
        "srprop": "snippet", "format": "json", "formatversion": 2,
    })
    return [
        SearchHit(title=h["title"], snippet=html.unescape(_TAG.sub("", h.get("snippet", ""))), lang=lang)
        for h in data.get("query", {}).get("search", [])
    ]


def _page_sentences(title: str, lang: str) -> tuple[str, list[str]]:
    data = get_json(_api(lang), {
        "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
        "titles": title, "format": "json", "formatversion": 2,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or "extract" not in pages[0]:
        raise ToolError(f"no Wikipedia ({lang}) page titled '{title}'. Search first, then use an exact title.")
    page = pages[0]
    return page["title"], split_sentences(page["extract"])


@traced_tool("fetch_page")
def fetch_page(title: str, lang: str = "en", start: int = 0, max_sentences: int = 40) -> Document:
    """Read a page as numbered sentences, from sentence `start`. These sentences are citable."""
    canonical, sentences = _page_sentences(title, lang)
    start = max(0, start)
    if sentences and start >= len(sentences):
        raise ToolError(f"'{canonical}' has only {len(sentences)} sentences (start={start})")
    chunk = sentences[start:start + max_sentences]
    return Document(
        source_id=f"wiki:{lang}:{canonical}",
        kind="wikipedia",
        title=canonical,
        url=f"https://{lang}.wikipedia.org/wiki/{quote(canonical.replace(' ', '_'))}",
        lang=lang,
        sentences=[Sentence(idx=start + i, text=s) for i, s in enumerate(chunk)],
        total_sentences=len(sentences),
    )
