"""Evaluation helpers: accuracy, per-class precision/recall/F1, confusion matrix, tables."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from rich.console import Console
from rich.table import Table


def classification_report(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """Predictions outside `labels` (e.g. "ERROR") count as wrong."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    n = len(gold)
    pairs = Counter(zip(gold, pred))
    per_class = {}
    for lab in labels:
        tp = pairs[(lab, lab)]
        fp = sum(c for (g, p), c in pairs.items() if p == lab and g != lab)
        fn = sum(c for (g, p), c in pairs.items() if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    pred_labels = list(labels) + sorted({p for p in pred if p not in labels})
    confusion = {g: {p: pairs[(g, p)] for p in pred_labels} for g in labels}
    correct = sum(pairs[(lab, lab)] for lab in labels)
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(labels) if labels else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }


def print_report(report: dict[str, Any], title: str = "Results", console: Console | None = None) -> None:
    console = console or Console()
    t = Table(title=f"{title}  (n={report['n']}, accuracy={report['accuracy']:.1%}, "
                    f"macro-F1={report['macro_f1']:.3f})")
    t.add_column("class")
    for col in ("precision", "recall", "f1", "support"):
        t.add_column(col, justify="right")
    for lab, c in report["per_class"].items():
        t.add_row(lab, f"{c['precision']:.2f}", f"{c['recall']:.2f}", f"{c['f1']:.2f}", str(c["support"]))
    console.print(t)

    conf = report["confusion"]
    pred_labels = list(next(iter(conf.values())).keys()) if conf else []
    ct = Table(title="Confusion (rows = gold, cols = predicted)")
    ct.add_column("gold \\ pred")
    for p in pred_labels:
        ct.add_column(p, justify="right")
    for g, row in conf.items():
        ct.add_row(g, *[str(row[p]) for p in pred_labels])
    console.print(ct)
