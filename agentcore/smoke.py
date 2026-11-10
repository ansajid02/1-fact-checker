"""`make smoke`: step-1 check. One structured Gemini call through core/llm, traced."""

from __future__ import annotations

import sys

from pydantic import BaseModel
from rich.console import Console

from . import llm, trace


class Capital(BaseModel):
    country: str
    capital: str
    capital_in_arabic: str


def main() -> None:
    tier = sys.argv[1] if len(sys.argv) > 1 else "flash"
    console = Console()
    with trace.run("smoke") as run_id, trace.collect() as events:
        answer = llm.generate("What is the capital of Saudi Arabia?", schema=Capital,
                              tier=tier, agent="smoke")
    console.print(answer)
    e = events[-1]
    console.print(f"model={e['model']} in={e['input_tokens']} out={e['output_tokens']} "
                  f"latency={e['latency_ms']}ms cost={e['cost_usd']}")
    console.print(f"trace: {trace.trace_path(run_id)}")


if __name__ == "__main__":
    main()
