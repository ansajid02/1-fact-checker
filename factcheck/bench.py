"""Benchmark a mode (single agent / team) on a dataset; score labels, citations, cost and time.

Per-claim rows are appended to results/<run_id>.jsonl as they finish (a crash loses nothing;
--resume <file> continues), and a summary goes to results/<run_id>.summary.json.
`make compare` turns all summaries into one table (results/RESULTS.md).

Evidence metrics (FEVER only, where gold evidence is known):
- fever_score        label correct AND (NEI, or a complete gold evidence set is cited) - the official
                     FEVER metric; sentence ids only line up for KB citations (fever:<page>)
- fever_score_page   label correct AND (NEI, or a gold page is cited, KB or live Wikipedia)
- gold_page_cited    % of verifiable claims where a cited source is a gold evidence page
- kb_citation_precision  % of valid KB citations that are gold evidence sentences
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from agentcore import evals, trace
from agentcore.config import settings

from . import baseline, team
from .paths import RESULTS_DIR
from .schemas import LABELS, CheckResult
from .tools import AccessBlocked, preflight

console = Console()

MODES: dict[str, Callable[..., CheckResult]] = {"single": baseline.check, "team": team.check}


def load_dataset(name: str) -> list:
    if name == "fever":
        from .fever import load_sample
    elif name == "arafacts":
        from .arafacts import load_sample
    else:
        raise ValueError(f"unknown dataset '{name}' (fever | arafacts)")
    return load_sample()


def source_title(source_id: str) -> tuple[str, str]:
    """'fever:Title' -> ('kb', 'Title'); 'wiki:en:Title' -> ('wiki', 'Title'); else (kind, id)."""
    if source_id.startswith("fever:"):
        return "kb", source_id[6:]
    if source_id.startswith("wiki:en:"):
        return "wiki", source_id[8:]
    return source_id.split(":", 1)[0], source_id


def evidence_metrics(item, r: CheckResult) -> dict:
    valid = [c for c in r.citations if c.valid]
    gold_pairs = {tuple(p) for es in item.evidence_sets for p in es}
    gold_pages = set(item.evidence_titles)
    cited_kb = {(source_title(c.source_id)[1], c.sent_idx) for c in valid if source_title(c.source_id)[0] == "kb"}
    cited_pages = {source_title(c.source_id)[1] for c in valid if source_title(c.source_id)[0] in ("kb", "wiki")}
    correct = r.label == item.label
    verifiable = bool(item.evidence_sets)
    full_set = any(all(tuple(p) in cited_kb for p in es) for es in item.evidence_sets)
    return {
        "verifiable": verifiable,
        "gold_page_cited": bool(cited_pages & gold_pages) if verifiable else None,
        "kb_citations": len(cited_kb),
        "kb_citations_gold": len(cited_kb & gold_pairs),
        "fever_score": correct and (item.label == "NOT_ENOUGH_INFO" or full_set),
        "fever_score_page": correct and (item.label == "NOT_ENOUGH_INFO" or bool(cited_pages & gold_pages)),
    }


def _run_one(item, mode: str, tier: str, backend: str | None, judge: str, searchers: list[str]) -> dict:
    with trace.tags(claim_id=item.id, mode=mode), trace.collect() as events:
        t0 = time.perf_counter()
        try:
            r = MODES[mode](item.claim, tier=tier, backend=backend, judge=judge, searchers=searchers)
        except AccessBlocked:
            raise
        except Exception as e:  # one bad claim must not kill the run
            r = CheckResult(claim=item.claim, mode=mode, label="ERROR", error=f"{type(e).__name__}: {e}"[:500])
        elapsed = time.perf_counter() - t0
    tot = trace.summarize(events)["total"]
    return {
        "id": item.id, "dataset": item.dataset, "claim": item.claim, "gold": item.label, "pred": r.label,
        "correct": r.label == item.label, "confidence": r.confidence, "explanation": r.explanation,
        "citations": [x.model_dump() for x in r.citations],
        "n_citations": len(r.citations),
        "n_fake_citations": r.fake_citations + r.extra.get("invalid_judge_ids", 0),
        "sources_read": r.sources_read, "gold_pages": item.evidence_titles, "steps": r.steps,
        **(evidence_metrics(item, r) if item.dataset == "fever" else {}),
        "latency_s": round(elapsed, 2), "llm_calls": tot["llm_calls"], "tool_calls": tot["tool_calls"],
        "input_tokens": tot["input_tokens"], "output_tokens": tot["output_tokens"],
        "cost_usd": tot["cost_usd"], "cost_unknown_calls": tot["cost_unknown"],
        "error": r.error, "extra": r.extra,
    }


def _rate(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows) or 1
    report = evals.classification_report([r["gold"] for r in rows], [r["pred"] for r in rows], LABELS)
    total_cites = sum(r["n_citations"] for r in rows)
    kb_cites = sum(r.get("kb_citations", 0) for r in rows if r.get("verifiable"))
    return {
        "report": report,
        "errors": sum(r["pred"] == "ERROR" for r in rows),
        "citations_total": total_cites,
        "fake_citation_rate": sum(r["n_fake_citations"] for r in rows) / total_cites if total_cites else 0.0,
        "fever_score": _rate(rows, "fever_score"),
        "fever_score_page": _rate(rows, "fever_score_page"),
        "gold_page_cited": _rate([r for r in rows if r.get("verifiable")], "gold_page_cited"),
        "kb_citation_precision": (sum(r.get("kb_citations_gold", 0) for r in rows if r.get("verifiable")) / kb_cites
                                  if kb_cites else None),
        "avg_steps": sum(r["steps"] for r in rows) / n,
        "avg_latency_s": sum(r["latency_s"] for r in rows) / n,
        "avg_cost_usd": sum(r["cost_usd"] for r in rows) / n,
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
        "cost_complete": all(r["cost_unknown_calls"] == 0 for r in rows),
        "avg_input_tokens": sum(r["input_tokens"] for r in rows) / n,
        "avg_output_tokens": sum(r["output_tokens"] for r in rows) / n,
    }


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def print_summary(s: dict, title: str) -> None:
    evals.print_report(s["report"], title=title, console=console)
    t = Table(title="Evidence, cost and time")
    t.add_column("metric")
    t.add_column("value", justify="right")
    cost_note = "" if s["cost_complete"] else "  (incomplete: fill prices.toml)"
    for k, v in [
        ("errors", str(s["errors"])),
        ("FEVER score (label + full gold evidence, KB)", _pct(s["fever_score"])),
        ("FEVER score, page level", _pct(s["fever_score_page"])),
        ("gold page cited (verifiable claims)", _pct(s["gold_page_cited"])),
        ("KB citation precision", _pct(s["kb_citation_precision"])),
        ("fake citation rate", f"{s['fake_citation_rate']:.1%} of {s['citations_total']}"),
        ("avg rounds / steps", f"{s['avg_steps']:.1f}"),
        ("avg seconds / claim", f"{s['avg_latency_s']:.1f}"),
        ("avg tokens / claim (in / out)", f"{s['avg_input_tokens']:.0f} / {s['avg_output_tokens']:.0f}"),
        ("avg cost / claim", f"${s['avg_cost_usd']:.5f}{cost_note}"),
        ("total cost", f"${s['total_cost_usd']:.4f}{cost_note}"),
    ]:
        t.add_row(k, v)
    console.print(t)


def run(mode: str = "single", dataset: str = "fever", limit: int | None = None, workers: int = 4,
        tier: str = "flash", backend: str | None = None, judge: str = "llm",
        searchers: list[str] | None = None, resume: Path | None = None) -> dict:
    backend = backend or settings().llm_backend
    searchers = searchers or ["wiki", "kb", "papers"]
    if "wiki" in searchers:
        preflight()
    items = load_dataset(dataset)
    items = items[:limit] if limit else items
    done: list[dict] = []
    if resume:
        done = [json.loads(l) for l in resume.read_text(encoding="utf-8").splitlines() if l.strip()]
        done_ids = {r["id"] for r in done}
        items = [c for c in items if c.id not in done_ids]

    label = f"{dataset}-{mode}" + (f"-{judge.replace(':', '_')}" if judge != "llm" else "")
    with trace.run(f"bench-{label}") as run_id:
        out_path = resume or RESULTS_DIR / f"{run_id}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()
        rows = list(done)
        console.print(f"[bold]{mode}[/bold] on {dataset}: {len(items)} claims ({len(done)} already done), "
                      f"tier={tier}, backend={backend}, judge={judge}, searchers={','.join(searchers)}, "
                      f"workers={workers}\n  results: {out_path}\n  trace:   {trace.trace_path(run_id)}")
        with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                      TimeElapsedColumn(), console=console) as progress, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            task = progress.add_task("checking", total=len(items))
            # each task needs its own copy of the context (run id / tags live in contextvars)
            futures = [pool.submit(contextvars.copy_context().run, _run_one, c, mode, tier, backend, judge, searchers)
                       for c in items]
            correct = 0
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                correct += row["correct"]
                with lock:
                    rows.append(row)
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                progress.update(task, advance=1, description=f"checking (acc so far {correct / i:.0%})")

    try:
        model = settings().model_for(backend, tier)
    except RuntimeError:
        model = f"{backend}-{tier}"
    summary = {"run_id": run_id, "mode": mode, "dataset": dataset, "tier": tier, "backend": backend, "model": model,
               "judge": judge, "searchers": searchers, "n": len(rows), "results_file": str(out_path),
               **summarize_rows(rows)}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, title=f"{dataset} / {mode} / {model} / judge={judge}")
    return summary
