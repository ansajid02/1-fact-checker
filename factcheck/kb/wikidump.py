"""Stream FEVER's June-2017 Wikipedia dump (wiki-pages.zip, intro sections only).

Each line: {"id": "Soul_Food_-LRB-film-RRB-", "text": "...", "lines": "0\\tSentence\\tlink...\\n1\\t..."}.
Sentence numbers are kept, so citations can be compared with FEVER's gold evidence.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from ..fever import decode_title
from ..schemas import Sentence

_ID = re.compile(r'^\{"id": "((?:[^"\\]|\\.)*)"')
_DETOK = [
    (re.compile(r"-LRB-"), "("), (re.compile(r"-RRB-"), ")"), (re.compile(r"-LSB-"), "["),
    (re.compile(r"-RSB-"), "]"), (re.compile(r"-LCB-"), "{"), (re.compile(r"-RCB-"), "}"),
    (re.compile(r"-COLON-"), ":"), (re.compile(r"``|''"), '"'),
    (re.compile(r"\s+([,.;:!?%)\]}])"), r"\1"), (re.compile(r"([(\[{$])\s+"), r"\1"),
    (re.compile(r"\s+('s|'re|'ve|'ll|'d|n't)\b"), r"\1"),
]


class WikiPage(BaseModel):
    title: str
    sentences: list[Sentence]  # original FEVER sentence ids, empty lines dropped


def detok(text: str) -> str:
    for pattern, repl in _DETOK:
        text = pattern.sub(repl, text)
    return text.strip()


def raw_title(line: str) -> str | None:
    """Cheap title extraction without parsing the whole JSON line."""
    m = _ID.match(line)
    return decode_title(json.loads(f'"{m.group(1)}"')) if m else None


def parse_line(line: str) -> WikiPage | None:
    row = json.loads(line)
    if not row.get("id") or not row.get("lines"):
        return None
    sentences = []
    for entry in row["lines"].split("\n"):
        parts = entry.split("\t")
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].strip():
            continue
        sentences.append(Sentence(idx=int(parts[0]), text=detok(parts[1])))
    if not sentences:
        return None
    return WikiPage(title=decode_title(row["id"]), sentences=sentences)


def iter_lines(zip_path: Path) -> Iterator[str]:
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".jsonl")):
            with zf.open(name) as f:
                for raw in f:
                    yield raw.decode("utf-8", errors="replace")  # a few bad bytes exist in the dump
