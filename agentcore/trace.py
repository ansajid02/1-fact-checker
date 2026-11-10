"""Tracing: every LLM and tool call becomes one JSON line (tokens, cost, latency).

Usage:
    with trace.run("bench-single"):            # one JSONL file per run
        with trace.tags(claim_id=42):          # extra fields on every event inside
            with trace.collect() as events:    # also keep events in memory
                ...                            # llm.generate / @traced_tool calls
        print(trace.summarize(events))
"""

from __future__ import annotations

import functools
import json
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import settings

_run_id: ContextVar[str | None] = ContextVar("trace_run_id", default=None)
_tags: ContextVar[dict[str, Any]] = ContextVar("trace_tags", default={})
_collectors: ContextVar[tuple[list[dict], ...]] = ContextVar("trace_collectors", default=())
_write_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id(name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{name}_{uuid.uuid4().hex[:6]}"


def current_run_id() -> str:
    run_id = _run_id.get()
    if run_id is None:
        run_id = f"adhoc-{datetime.now().strftime('%Y%m%d')}"
    return run_id


def trace_path(run_id: str | None = None) -> Path:
    return settings().trace_dir / f"{run_id or current_run_id()}.jsonl"


@contextmanager
def run(name: str) -> Iterator[str]:
    """Start a traced run; all events inside go to traces/<run_id>.jsonl."""
    run_id = new_run_id(name)
    token = _run_id.set(run_id)
    try:
        yield run_id
    finally:
        _run_id.reset(token)


@contextmanager
def tags(**kv: Any) -> Iterator[None]:
    token = _tags.set({**_tags.get(), **kv})
    try:
        yield
    finally:
        _tags.reset(token)


@contextmanager
def collect() -> Iterator[list[dict]]:
    """Capture events emitted inside this block (in addition to writing them)."""
    bucket: list[dict] = []
    token = _collectors.set((*_collectors.get(), bucket))
    try:
        yield bucket
    finally:
        _collectors.reset(token)


def log(kind: str, name: str, **fields: Any) -> dict:
    event = {
        "ts": _now(),
        "run_id": current_run_id(),
        "kind": kind,
        "name": name,
        **_tags.get(),
        **fields,
    }
    for bucket in _collectors.get():
        bucket.append(event)
    path = trace_path(event["run_id"])
    line = json.dumps(event, ensure_ascii=False, default=str)
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return event


@contextmanager
def span(kind: str, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block. Add fields to the yielded dict; they are logged on exit."""
    extra: dict[str, Any] = {}
    start = time.perf_counter()
    error: str | None = None
    try:
        yield extra
    except BaseException as e:
        error = f"{type(e).__name__}: {e}"[:500]
        raise
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        log(kind, name, latency_ms=latency_ms, ok=error is None, error=error, **fields, **extra)


def traced_tool(name: str | None = None) -> Callable:
    """Decorator: log every call of a tool function (args, latency, ok/error)."""

    def deco(fn: Callable) -> Callable:
        tool_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span("tool", tool_name, args=_short_args(args, kwargs)):
                return fn(*args, **kwargs)

        return wrapper

    return deco


def _short_args(args: tuple, kwargs: dict) -> dict:
    out = {f"arg{i}": a for i, a in enumerate(args)} | kwargs
    return {k: (v if isinstance(v, (int, float, bool)) or v is None else str(v)[:200]) for k, v in out.items()}


def load(run_id: str) -> list[dict]:
    path = trace_path(run_id)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(events: list[dict]) -> dict[str, Any]:
    """Totals plus a per-(kind, name) breakdown."""
    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "errors": 0, "input_tokens": 0, "output_tokens": 0,
                 "cost_usd": 0.0, "cost_unknown": 0, "latency_ms": 0.0}
    )
    for e in events:
        r = rows[f"{e['kind']}:{e['name']}"]
        r["calls"] += 1
        r["errors"] += 0 if e.get("ok", True) else 1
        r["input_tokens"] += e.get("input_tokens") or 0
        r["output_tokens"] += e.get("output_tokens") or 0
        r["latency_ms"] += e.get("latency_ms") or 0.0
        if e["kind"] in ("llm", "embed"):
            if e.get("cost_usd") is None:
                r["cost_unknown"] += 1
            else:
                r["cost_usd"] += e["cost_usd"]
    total = {k: sum(r[k] for r in rows.values()) for k in
             ("calls", "errors", "input_tokens", "output_tokens", "cost_usd", "cost_unknown")}
    total["llm_calls"] = sum(r["calls"] for k, r in rows.items() if k.startswith("llm:"))
    total["embed_calls"] = sum(r["calls"] for k, r in rows.items() if k.startswith("embed:"))
    total["tool_calls"] = sum(r["calls"] for k, r in rows.items() if k.startswith("tool:"))
    return {"total": total, "by_name": dict(rows)}
