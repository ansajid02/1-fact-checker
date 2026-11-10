"""`make models [FILTER=gemini]`: list models the current backend can use.

OpenRouter: public model list with prices and structured-output support.
Vertex AI:  Gemini base models your Google Cloud project can call.
"""

from __future__ import annotations

import sys

import httpx
from rich.console import Console
from rich.table import Table

from .config import settings

console = Console()


BEDROCK_CLAUDE = [  # Claude in Amazon Bedrock (no list endpoint there); prices = Anthropic list, $/1M in / out
    ("anthropic.claude-haiku-5-5", "Claude Haiku 5.5", "0.10 / 0.50"),
    ("anthropic.claude-sonnet-5-5", "Claude Sonnet 5.5", "2.00 / 10.00"),
    ("anthropic.claude-opus-5-5", "Claude Opus 5.5", "4.00 / 20.00"),
    ("anthropic.claude-fable-5-1", "Claude Fable 5.1", "10.00 / 50.00"),
]


def list_bedrock(filter_text: str) -> None:
    s = settings()
    t = Table(title=f"Claude in Amazon Bedrock ({s.bedrock_region})")
    for col in ("model id", "name", "$ / 1M in / out"):
        t.add_column(col)
    for row in BEDROCK_CLAUDE:
        t.add_row(*row)
    console.print(t)
    console.print("Model access is per account: check the Bedrock console (Model access).")
    console.print(f"Current .env: BEDROCK_MODEL_FLASH={s.bedrock_model_flash}  BEDROCK_MODEL_PRO={s.bedrock_model_pro}")


def list_openrouter(filter_text: str) -> None:
    s = settings()
    data = httpx.get(f"{s.openrouter_base_url}/models", timeout=30).json()["data"]
    t = Table(title=f"OpenRouter models matching '{filter_text}' (USD per 1M tokens)")
    for col in ("model id", "structured", "input $", "output $", "context"):
        t.add_column(col, justify="left" if col == "model id" else "right")
    for m in sorted(data, key=lambda m: m["id"]):
        if filter_text.lower() not in m["id"].lower():
            continue
        p = m.get("pricing", {})
        structured = "structured_outputs" in m.get("supported_parameters", [])
        t.add_row(m["id"], "yes" if structured else "[red]no[/]",
                  f"{float(p.get('prompt', 0)) * 1e6:.3f}", f"{float(p.get('completion', 0)) * 1e6:.3f}",
                  f"{m.get('context_length') or '?'}")
    console.print(t)
    console.print(f"Current .env: OPENROUTER_MODEL_FLASH={s.openrouter_model_flash or '(unset)'}  "
                  f"OPENROUTER_MODEL_PRO={s.openrouter_model_pro or '(unset)'}")
    console.print("[dim]Agents need structured = yes.[/dim]")


def list_vertex(filter_text: str) -> None:
    from .llm import _vertex_client

    s = settings()
    t = Table(title=f"Vertex AI base models  (project={s.gcp_project}, location={s.gcp_location})")
    t.add_column("model")
    t.add_column("display name")
    t.add_column("in / out token limit", justify="right")
    for m in _vertex_client().models.list(config={"query_base": True, "page_size": 100}):
        name = (m.name or "").split("/")[-1]
        if filter_text.lower() not in name.lower():
            continue
        t.add_row(name, m.display_name or "", f"{m.input_token_limit or '?'} / {m.output_token_limit or '?'}")
    console.print(t)
    console.print(f"Current .env: VERTEX_MODEL_FLASH={s.vertex_model_flash or '(unset)'}  "
                  f"VERTEX_MODEL_PRO={s.vertex_model_pro or '(unset)'}")


def main() -> None:
    filter_text = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    backend = sys.argv[2] if len(sys.argv) > 2 else settings().llm_backend
    {"bedrock": list_bedrock, "openrouter": list_openrouter, "vertex": list_vertex}[backend](filter_text)


if __name__ == "__main__":
    main()
