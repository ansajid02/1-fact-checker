"""Tools: the only way agents touch data. Each returns compact, validated JSON."""

from .http import AccessBlocked, ToolError, get_json
from .papers import paper_search
from .wikipedia import fetch_page, wikipedia_search


def preflight() -> None:
    """Fail fast (before spending LLM money) if a data source refuses us."""
    for lang in ("en", "ar"):
        get_json(f"https://{lang}.wikipedia.org/w/api.php",
                 {"action": "query", "meta": "siteinfo", "format": "json"}, use_cache=False)

TOOL_LINES = {
    "wikipedia_search": 'wikipedia_search(query, lang)  -> page titles + snippets. lang = "en" or "ar". Snippets are NOT citable.',
    "fetch_page": "fetch_page(title, lang, start) -> numbered sentences of a Wikipedia page, from sentence `start`\n"
                  "                                  (40 per call; call again with a higher start to read further).",
    "kb_search": "kb_search(query)              -> numbered sentences from the local knowledge base (Wikipedia 2017 intros).",
    "paper_search": "paper_search(query)            -> up to 3 scientific papers (OpenAlex) with numbered abstract sentences.",
    "finish": "finish                         -> stop searching and give the verdict.",
}
# searcher name (shared with the team) -> tools it enables
SEARCHER_TOOLS = {"wiki": ["wikipedia_search", "fetch_page"], "kb": ["kb_search"], "papers": ["paper_search"]}


def tool_docs(searchers: list[str]) -> str:
    names = [t for s in searchers for t in SEARCHER_TOOLS[s]] + ["finish"]
    return "\n".join(TOOL_LINES[n] for n in names)


TOOL_DOCS = tool_docs(["wiki", "kb", "papers"])

__all__ = ["AccessBlocked", "SEARCHER_TOOLS", "ToolError", "TOOL_DOCS", "tool_docs", "fetch_page", "paper_search", "preflight",
           "wikipedia_search"]
