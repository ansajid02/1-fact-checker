"""`make compare`: one table from every benchmark summary -> terminal + results/RESULTS.md
(and the README's results section, between the RESULTS markers)."""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from .paths import PROJECT_ROOT, RESULTS_DIR

COLUMNS = [
    ("dataset", lambda s: s["dataset"]),
    ("mode", lambda s: s["mode"]),
    ("model", lambda s: s.get("model", s["tier"])),
    ("judge", lambda s: s["judge"]),
    ("searchers", lambda s: ",".join(s["searchers"])),
    ("n", lambda s: str(s["n"])),
    ("accuracy", lambda s: f"{s['report']['accuracy']:.1%}"),
    ("macro-F1", lambda s: f"{s['report']['macro_f1']:.3f}"),
    ("FEVER score", lambda s: _pct(s.get("fever_score"))),
    ("gold page cited", lambda s: _pct(s.get("gold_page_cited"))),
    ("fake citations", lambda s: f"{s['fake_citation_rate']:.1%}"),
    ("$ / claim", lambda s: f"{s['avg_cost_usd']:.4f}" + ("" if s["cost_complete"] else "?")),
    ("s / claim", lambda s: f"{s['avg_latency_s']:.1f}"),
    ("errors", lambda s: str(s["errors"])),
]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _pct(x) -> str:
    return "-" if x is None else f"{x:.1%}"


def load_summaries() -> list[dict]:
    out = []
    for p in sorted(RESULTS_DIR.glob("*.summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        if "dataset" not in s:  # summaries from before datasets existed were FEVER runs
            s |= {"dataset": "fever", "judge": "llm", "searchers": ["wiki", "papers"], "n": s["report"]["n"]}
        out.append(s)
    return out


def markdown(summaries: list[dict]) -> str:
    if not summaries:
        return "_No benchmark runs yet. Run `make bench`._\n"
    head = "| " + " | ".join(c for c, _ in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    rows = ["| " + " | ".join(f(s) for _, f in COLUMNS) + " |" for s in summaries]
    note = ("\n`?` after a cost = some calls had no known price. FEVER score = label correct and a complete "
            "gold evidence set cited (KB citations only).\n")
    return "\n".join([head, sep, *rows]) + "\n" + note


def compare() -> str:
    summaries = load_summaries()
    t = Table(title="All benchmark runs")
    for c, _ in COLUMNS:
        t.add_column(c, justify="left" if c in ("dataset", "mode", "judge", "searchers") else "right")
    for s in summaries:
        t.add_row(*[f(s) for _, f in COLUMNS])
    Console().print(t)
    md = markdown(summaries)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "RESULTS.md").write_text("# Benchmark results\n\n" + md, encoding="utf-8")
    readme = PROJECT_ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if START in text and END in text:
            before, rest = text.split(START, 1)
            after = rest.split(END, 1)[1]
            readme.write_text(f"{before}{START}\n{md}{END}{after}", encoding="utf-8")
    return md
