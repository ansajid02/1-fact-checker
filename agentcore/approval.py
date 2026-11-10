"""Human approval gate: nothing risky (publish, send, book, trade) happens without a yes."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Literal

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import trace

console = Console()


class Decision(BaseModel):
    action: Literal["approve", "edit", "reject"]
    content: str  # final content (edited if action == "edit")
    reason: str = ""


def _edit(text: str) -> str:
    editor = os.environ.get("EDITOR")
    if editor:
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(text)
            path = f.name
        subprocess.call([*editor.split(), path])
        with open(path, encoding="utf-8") as f:
            return f.read()
    console.print("[dim]Type the new text. Finish with a line containing only '.'[/dim]")
    lines = []
    for line in sys.stdin:
        if line.rstrip("\n") == ".":
            break
        lines.append(line)
    return "".join(lines) or text


def ask(title: str, content: str, *, what: str = "publish") -> Decision:
    """Show content, ask approve / edit / reject. Non-interactive -> reject (safe default)."""
    # Text(): show LLM output verbatim - as markup, '[/x]' would crash and '[MEETS]' would vanish
    console.print(Panel(Text(content), title=Text(title), expand=False))
    if not sys.stdin.isatty():
        decision = Decision(action="reject", content=content, reason="no interactive terminal")
        console.print(f"[yellow]Not a terminal: not approving '{what}'.[/yellow]")
    else:
        while True:
            choice = console.input(f"[bold]{what}?[/bold] \\[a]pprove / \\[e]dit / \\[r]eject: ").strip().lower()
            if choice in ("a", "approve"):
                decision = Decision(action="approve", content=content)
                break
            if choice in ("e", "edit"):
                decision = Decision(action="edit", content=_edit(content))
                break
            if choice in ("r", "reject"):
                reason = console.input("reason (optional): ").strip()
                decision = Decision(action="reject", content=content, reason=reason)
                break
    trace.log("approval", what, action=decision.action, reason=decision.reason)
    return decision
