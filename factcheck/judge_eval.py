"""`make judge-eval JUDGES="..."`: compare judges on the SAME claims with the SAME fixed evidence.

Judge specs:
  llm:flash / llm:pro      the cloud LLM (LLM_BACKEND), same prompt + output format as the LoRA model
  ollama:<model>           a local Ollama model, e.g. the base model or the LoRA model (factjudge)
This isolates the judge: retrieval is held constant, so differences are the judge's alone.
"""

from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console
from rich.table import Table

from agentcore import evals, llm, trace

from . import judge_lora
from .lora_data import JUDGE_EVAL
from .paths import RESULTS_DIR
from .schemas import LABELS

console = Console()


def run_judge(spec: str, item: dict) -> str:
    kind, _, name = spec.partition(":")
    try:
        if kind == "ollama":
            return judge_lora.judge(item["claim"], item["evidence"], model=name, backend="ollama")
        if kind == "llm":
            out = llm.generate(judge_lora.prompt(item["claim"], item["evidence"]), schema=judge_lora.LoraVerdict,
                               system=judge_lora.SYSTEM, tier=name or "flash", agent="judge.eval")
            return out.label
        raise ValueError(f"bad judge spec '{spec}' (llm:flash | llm:pro | ollama:<model>)")
    except llm.LLMError:
        return "ERROR"


def evaluate(judges: list[str], limit: int | None = None, workers: int = 4) -> dict:
    if not JUDGE_EVAL.exists():
        raise FileNotFoundError(f"{JUDGE_EVAL} not found. Run `make lora-data` first.")
    items = [json.loads(l) for l in JUDGE_EVAL.read_text(encoding="utf-8").splitlines() if l]
    items = items[:limit] if limit else items
    results = {}
    with trace.run("judge-eval"):
        for spec in judges:
            with trace.tags(judge=spec), trace.collect() as events, ThreadPoolExecutor(workers) as pool:
                preds = list(pool.map(lambda it: contextvars.copy_context().run(run_judge, spec, it), items))
            tot = trace.summarize(events)["total"]
            report = evals.classification_report([i["label"] for i in items], preds, LABELS)
            results[spec] = {"report": report, "cost_usd": tot["cost_usd"],
                             "avg_latency_ms": sum(e.get("latency_ms", 0) for e in events if e["kind"] == "llm") / max(1, len(items)),
                             "errors": preds.count("ERROR")}
    t = Table(title=f"Judges on {len(items)} claims with fixed evidence")
    for col in ("judge", "accuracy", "macro-F1", "TRUE F1", "FALSE F1", "NEI F1", "errors", "total $", "ms / claim"):
        t.add_column(col, justify="left" if col == "judge" else "right")
    for spec, r in results.items():
        pc = r["report"]["per_class"]
        t.add_row(spec, f"{r['report']['accuracy']:.1%}", f"{r['report']['macro_f1']:.3f}",
                  *[f"{pc[l]['f1']:.2f}" for l in LABELS], str(r["errors"]), f"{r['cost_usd']:.4f}",
                  f"{r['avg_latency_ms']:.0f}")
    console.print(t)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "judge_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
