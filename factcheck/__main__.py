"""CLI:  uv run python -m factcheck <command>   (the Makefile wraps every command)"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from agentcore import trace

console = Console()
COLORS = {"TRUE": "green", "FALSE": "red", "NOT_ENOUGH_INFO": "yellow", "ERROR": "magenta"}


def cmd_prepare_data(a: argparse.Namespace) -> None:
    from . import arafacts, fever

    for name, sample in [("FEVER test (dev split)", fever.make_sample(per_class=a.per_class, seed=a.seed)),
                         ("FEVER LoRA train sample", fever.make_train_sample(seed=a.seed)),
                         ("AraFacts", arafacts.make_sample(seed=a.seed))]:
        console.print(f"{name}: {len(sample)} claims {dict(Counter(c.label for c in sample))}")


def cmd_kb(a: argparse.Namespace) -> None:
    from .fever import FEVER_WIKI_URL, RAW_WIKI, download_file
    from .kb.build import build_pages
    from .kb.index import build_index

    if not RAW_WIKI.exists():
        console.print(f"Downloading FEVER Wikipedia dump (1.7 GB) to {RAW_WIKI} ...")
        download_file(FEVER_WIKI_URL, RAW_WIKI)
    if not a.skip_pages:
        console.print("Selecting KB pages from 5.4M (one pass, ~1 min) ...")
        console.print(build_pages(n_random=a.random_pages))
    console.print("Chunking" + ("" if a.no_embed else " and embedding") + " ...")
    console.print(build_index(with_embeddings=not a.no_embed))


def cmd_kb_eval(a: argparse.Namespace) -> None:
    from .kb.evaluate import evaluate

    evaluate(rerank=a.rerank)


def _print_result(r, tot, run_id) -> None:
    # LLM / source text is escaped: it may contain [brackets] that rich would parse as markup
    body = [f"[bold {COLORS[r.label]}]{r.label}[/]  (confidence {r.confidence:.2f})", "", escape(r.explanation), ""]
    for c in r.citations:
        mark = "[green]✓[/]" if c.valid else f"[red]✗ {c.reason}[/]"
        body.append(f"{mark} {escape(c.source_id)} \\[{c.sent_idx}]: “{escape(c.quote)}”")
    if not r.citations:
        body.append("[dim](no citations)[/dim]")
    cost = "unknown (fill prices.toml)" if tot["cost_unknown"] else f"${tot['cost_usd']:.5f}"
    body += ["", f"[dim]{r.steps} steps · {tot['llm_calls']} LLM calls · {tot['tool_calls']} tool calls · "
                 f"{tot['input_tokens']}/{tot['output_tokens']} tokens · cost {cost}[/dim]",
             f"[dim]trace: {trace.trace_path(run_id)}[/dim]"]
    console.print(Panel("\n".join(body), title=escape(r.claim), expand=False))


def cmd_check(a: argparse.Namespace) -> None:
    searchers = a.searchers.split(",")
    with trace.run(f"check-{a.mode}") as run_id, trace.collect() as events:
        if a.mode == "single":
            from . import baseline

            r = baseline.check(a.claim, tier=a.tier, backend=a.backend, searchers=searchers)
            _print_result(r, trace.summarize(events)["total"], run_id)
            return
        from . import team

        cfg = team.TeamConfig(tier=a.tier, backend=a.backend, judge=a.judge, searchers=searchers,
                              publish=not a.no_publish)
        state = team.run(a.claim, cfg)
    tot = trace.summarize(events)["total"]
    if a.no_publish:
        console.print(Panel(Text(state["report"]), title="report", expand=False))
    cost = "unknown" if tot["cost_unknown"] else f"${tot['cost_usd']:.5f}"
    console.print(f"[dim]{len(state['claims'])} claim(s) · {state['round']} round(s) · {tot['llm_calls']} LLM calls · "
                  f"{tot['tool_calls']} tool calls · cost {cost}\ntrace: {trace.trace_path(run_id)}[/dim]")
    if state.get("published_path"):
        console.print(f"[green]Published:[/] {state['published_path']}")
    elif state.get("decision"):
        console.print(f"[yellow]Not published ({state['decision'].get('action')}).[/]")


def cmd_bench(a: argparse.Namespace) -> None:
    from . import bench

    bench.run(mode=a.mode, dataset=a.dataset, limit=a.limit, workers=a.workers, tier=a.tier,
              backend=a.backend, judge=a.judge, searchers=a.searchers.split(","),
              resume=Path(a.resume) if a.resume else None)


def cmd_compare(a: argparse.Namespace) -> None:
    from .report import compare

    compare()


def cmd_lora_data(a: argparse.Namespace) -> None:
    from .lora_data import build

    console.print(build())


def cmd_judge_eval(a: argparse.Namespace) -> None:
    from .judge_eval import evaluate

    evaluate(a.judges.split(), limit=a.limit, workers=a.workers)


def cmd_trace(a: argparse.Namespace) -> None:
    s = trace.summarize(trace.load(a.run_id))
    t = Table(title=f"trace {a.run_id}")
    for col in ("name", "calls", "errors", "in tok", "out tok", "cost $", "avg ms"):
        t.add_column(col, justify="left" if col == "name" else "right")
    for name, r in sorted(s["by_name"].items()):
        cost = f"{r['cost_usd']:.5f}" + ("?" if r["cost_unknown"] else "")
        t.add_row(name, str(r["calls"]), str(r["errors"]), str(r["input_tokens"]),
                  str(r["output_tokens"]), cost, f"{r['latency_ms'] / r['calls']:.0f}")
    console.print(t)
    console.print(json.dumps(s["total"]))


def main() -> None:
    p = argparse.ArgumentParser(prog="factcheck")
    sub = p.add_subparsers(required=True)

    def llm_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--tier", choices=["flash", "pro"], default="flash")
        sp.add_argument("--backend", default=None, help="bedrock | openrouter | vertex | ollama (default: LLM_BACKEND in .env)")
        sp.add_argument("--judge", default="llm", help="llm | llm:pro | lora:<ollama model>  (team only)")
        sp.add_argument("--searchers", default="wiki,kb,papers", help="comma list of wiki, kb, papers")

    s = sub.add_parser("prepare-data", help="FEVER test + LoRA train samples, AraFacts sample")
    s.add_argument("--per-class", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_prepare_data)

    s = sub.add_parser("kb", help="build the local knowledge base (download dump, select pages, chunk, embed)")
    s.add_argument("--no-embed", action="store_true", help="BM25 only (no embedding cost)")
    s.add_argument("--skip-pages", action="store_true", help="reuse selected pages, only re-chunk/embed")
    s.add_argument("--random-pages", type=int, default=3000)
    s.set_defaults(fn=cmd_kb)

    s = sub.add_parser("kb-eval", help="retrieval recall on the test claims")
    s.add_argument("--rerank", action="store_true", help="also evaluate LLM rerank (costs a little)")
    s.set_defaults(fn=cmd_kb_eval)

    s = sub.add_parser("check", help="fact-check a claim or short article")
    s.add_argument("claim")
    s.add_argument("--mode", choices=["team", "single"], default="team")
    s.add_argument("--no-publish", action="store_true", help="skip the approval/publish step")
    llm_args(s)
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("bench", help="benchmark a mode on a dataset")
    s.add_argument("--mode", default="single", choices=["single", "team"])
    s.add_argument("--dataset", default="fever", choices=["fever", "arafacts"])
    s.add_argument("--limit", type=int, default=None, help="first N claims (stays label-balanced)")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--resume", default=None, help="results .jsonl to continue")
    llm_args(s)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("compare", help="table of all benchmark runs -> results/RESULTS.md + README")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("lora-data", help="build LoRA train/val data and the judge-eval set")
    s.set_defaults(fn=cmd_lora_data)

    s = sub.add_parser("judge-eval", help="compare judges on fixed evidence")
    s.add_argument("--judges", default="llm:flash", help='space-separated, e.g. "llm:flash ollama:factjudge"')
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(fn=cmd_judge_eval)

    s = sub.add_parser("trace", help="summarise a trace file by run id")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_trace)

    a = p.parse_args()
    from agentcore.embed import EmbedError
    from agentcore.llm import LLMError

    from .tools import AccessBlocked

    try:
        a.fn(a)
    except (AccessBlocked, LLMError, EmbedError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[bold red]{type(e).__name__}:[/] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
