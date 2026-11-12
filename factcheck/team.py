"""Agent team (LangGraph).

  planner -> [wiki | kb | papers searchers, in parallel per claim] -> reader -> judge
          -> (confidence < 0.6 and round < 2: judge's follow-up queries go back to the searchers)
          -> writer -> [human approval] -> publish

- Every agent returns validated JSON (Pydantic).
- Reader quotes are checked against the sentences actually retrieved (hallucination control);
  the judge can only cite evidence the reader produced (by number).
- The judge can be the LLM (default) or the local LoRA model in Ollama ("lora:<model>").
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from agentcore import approval, llm

from . import judge_lora
from .citations import EvidenceStore, verify
from .paths import PROJECT_ROOT
from .schemas import LABEL_RULES, CheckedCitation, CheckResult, Citation, Document, Label, Lang
from .text import is_arabic, normalize
from .tools import AccessBlocked, ToolError, fetch_page, paper_search, wikipedia_search

ALL_SEARCHERS = ("wiki", "kb", "papers")
CONFIDENCE_THRESHOLD = 0.6
MAX_ROUNDS = 2
MAX_CLAIMS = 5
READER_MAX_SENTENCES = 60
# 6 quoted sentences need ~1,000 tokens. A lower cap makes a stuck (whitespace-looping) reply fail
# fast and cheap, and llm.generate then retries without constrained decoding.
READER_MAX_OUTPUT_TOKENS = 2000
PUBLISHED_DIR = PROJECT_ROOT / "published"


# ================================================================ agent I/O schemas


class Query(BaseModel):
    query: str
    lang: Lang = "en"


class PlannedClaim(BaseModel):
    text: str = Field(description="one checkable claim, self-contained, in the input's language")
    english: str = Field(description="the claim in English (same as text if already English)")
    queries: list[Query] = Field(description="1-3 short search queries; entity names work best")
    scientific: bool = Field(False, description="true only for scientific/medical claims")


class PlannerOut(BaseModel):
    claims: list[PlannedClaim]


class ReaderItem(Citation):
    stance: Literal["supports", "refutes", "context"]


class ReaderOut(BaseModel):
    evidence: list[ReaderItem] = Field(description="at most 6 sentences, the most relevant first")


class JudgeOut(BaseModel):
    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(description="1-2 sentences, English")
    evidence_ids: list[int] = Field(description="numbers of the evidence items the verdict relies on")
    follow_up_queries: list[str] = Field(default_factory=list,
                                         description="if unsure: up to 2 searches that could settle it")


class WriterClaim(BaseModel):
    claim_id: int
    explanation: str = Field(description="1-3 sentences in the output language, no new facts")


class WriterOut(BaseModel):
    headline: str
    claims: list[WriterClaim]


# ================================================================ state records


class EvidenceItem(CheckedCitation):
    stance: Literal["supports", "refutes", "context"] = "context"
    url: str = ""
    title: str = ""


class DocBatch(BaseModel):
    claim_id: int
    round: int
    searcher: str
    docs: list[Document] = Field(default_factory=list)
    error: str | None = None


class Reading(BaseModel):
    claim_id: int
    round: int
    evidence: list[EvidenceItem]  # valid AND invalid (invalid = reader hallucinations)
    n_candidates: int


class Judgment(BaseModel):
    claim_id: int
    round: int
    judge: str
    label: Label
    confidence: float
    explanation: str
    citations: list[EvidenceItem]
    follow_up_queries: list[str] = Field(default_factory=list)
    invalid_ids: int = 0


class TeamConfig(BaseModel):
    tier: llm.Tier = "flash"
    backend: str | None = None
    judge: str = "llm"  # "llm" | "llm:pro" | "lora:<ollama model>"
    searchers: list[str] = Field(default_factory=lambda: list(ALL_SEARCHERS))
    publish: bool = False
    max_rounds: int = MAX_ROUNDS


class TeamState(TypedDict, total=False):
    text: str
    config: TeamConfig
    lang: str
    claims: list[PlannedClaim]
    round: int
    pending: list[int]
    queries: list[list[Query]]
    docs: Annotated[list[DocBatch], operator.add]
    readings: Annotated[list[Reading], operator.add]
    judgments: Annotated[list[Judgment], operator.add]
    headline: str
    explanations: dict[str, str]
    report: str
    decision: dict
    published_path: str


def latest(items: list, claim_id: int):
    found = [x for x in items if x.claim_id == claim_id]
    return max(found, key=lambda x: x.round) if found else None


# ================================================================ planner

PLANNER_SYSTEM = """You are the planner of a fact-checking team. Split the input into separate,
checkable factual claims (max 5). If the input is a single claim, return exactly that one claim,
unchanged. Ignore opinions and questions. For each claim write 1-3 short search queries
(entity names first). For Arabic input give queries in Arabic AND English."""


def planner(state: TeamState) -> dict:
    cfg, text = state["config"], state["text"]
    out = llm.generate(f"INPUT:\n{text}", schema=PlannerOut, system=PLANNER_SYSTEM, tier=cfg.tier,
                       backend=cfg.backend, agent="team.planner")
    claims = out.claims[:MAX_CLAIMS] or [PlannedClaim(text=text, english=text, queries=[Query(query=text[:200])])]
    return {
        "lang": "ar" if is_arabic(text) else "en",  # deterministic beats asking the LLM
        "claims": claims,
        "round": 1,
        "pending": list(range(len(claims))),
        "queries": [c.queries or [Query(query=c.english[:200])] for c in claims],
    }


# ================================================================ searchers (parallel)


class SearchTask(TypedDict):
    claim_id: int
    round: int
    searcher: str
    claim: PlannedClaim
    queries: list[Query]
    seen: list[str]


def dispatch_search(state: TeamState):
    cfg = state["config"]
    sends = []
    for cid in state["pending"]:
        claim = state["claims"][cid]
        seen = [d.source_id for b in state.get("docs", []) if b.claim_id == cid for d in b.docs]
        for s in cfg.searchers:
            if s == "papers" and not claim.scientific:
                continue
            sends.append(Send("search", SearchTask(claim_id=cid, round=state["round"], searcher=s, claim=claim,
                                                   queries=state["queries"][cid], seen=seen)))
    return sends or "gather"


def _wiki(task: SearchTask) -> list[Document]:
    docs, seen = [], set(task["seen"])
    for q in task["queries"][:3]:
        for hit in wikipedia_search(q.query, lang=q.lang, limit=3)[:2]:
            sid = f"wiki:{q.lang}:{hit.title}"
            if sid in seen or len(docs) >= 3:
                continue
            seen.add(sid)
            try:
                docs.append(fetch_page(hit.title, lang=q.lang))
            except ToolError:
                continue
    return docs


def _kb(task: SearchTask) -> list[Document]:
    from .kb.retriever import kb_search

    query = " ".join([task["claim"].english] + [q.query for q in task["queries"] if q.lang == "en"][:2])
    return kb_search(query, k=6)


def _papers(task: SearchTask) -> list[Document]:
    return paper_search(task["queries"][0].query if task["queries"] else task["claim"].english)


SEARCHERS = {"wiki": _wiki, "kb": _kb, "papers": _papers}


def search(task: SearchTask) -> dict:
    batch = DocBatch(claim_id=task["claim_id"], round=task["round"], searcher=task["searcher"])
    try:
        batch.docs = SEARCHERS[task["searcher"]](task)
    except AccessBlocked:
        raise
    except (ToolError, FileNotFoundError) as e:  # one searcher failing must not sink the team
        batch.error = str(e)[:300]
    return {"docs": [batch]}


def barrier(state: TeamState) -> dict:
    return {}


# ================================================================ reader

READER_SYSTEM = """You are the reader of a fact-checking team. From the numbered sentences, pick the
ones that support or refute the claim (or give essential context). Copy each sentence exactly with
its source_id and [number]. Only use sentences shown. Pick at most 6. If nothing is relevant, return
an empty list."""


def prefilter(claim: str, docs: list[Document], limit: int = READER_MAX_SENTENCES) -> list[Document]:
    """Keep the sentences most lexically related to the claim (+ each page's first sentence)."""
    rows = [(d, s) for d in docs for s in d.sentences]
    if len(rows) <= limit:
        return docs
    bm25 = BM25Okapi([normalize(s.text).split() or ["_"] for _, s in rows])
    scores = bm25.get_scores(normalize(claim).split())
    keep = {id(rows[i][1]) for i in sorted(range(len(rows)), key=lambda i: -scores[i])[:limit]}
    out = []
    for d in docs:
        sents = [s for j, s in enumerate(d.sentences) if id(s) in keep or j == 0]
        out.append(d.model_copy(update={"sentences": sents}))
    return out


def render_docs(docs: list[Document]) -> str:
    blocks = []
    for d in docs:
        lines = [f"source_id: {d.source_id}", f"title: {d.title}"] + [f"[{s.idx}] {s.text}" for s in d.sentences]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class ReadTask(TypedDict):
    claim_id: int
    round: int
    claim: PlannedClaim
    docs: list[Document]
    config: TeamConfig


def dispatch_read(state: TeamState):
    sends = []
    for cid in state["pending"]:
        docs: dict[str, Document] = {}
        for b in state.get("docs", []):
            if b.claim_id == cid:
                for d in b.docs:
                    if d.source_id in docs:  # same page from two rounds: merge sentences
                        merged = {s.idx: s for s in docs[d.source_id].sentences + d.sentences}
                        d = d.model_copy(update={"sentences": [merged[i] for i in sorted(merged)]})
                    docs[d.source_id] = d
        sends.append(Send("reader", ReadTask(claim_id=cid, round=state["round"], claim=state["claims"][cid],
                                             docs=list(docs.values()), config=state["config"])))
    return sends


def reader(task: ReadTask) -> dict:
    cfg, claim = task["config"], task["claim"]
    docs = prefilter(claim.english, task["docs"])
    store = EvidenceStore()
    for d in docs:
        store.add(d)
    n = sum(len(d.sentences) for d in docs)
    if n == 0:
        return {"readings": [Reading(claim_id=task["claim_id"], round=task["round"], evidence=[], n_candidates=0)]}
    out = llm.generate(f"CLAIM: {claim.text}\n(English: {claim.english})\n\nSENTENCES:\n{render_docs(docs)}",
                       schema=ReaderOut, system=READER_SYSTEM, tier=cfg.tier, backend=cfg.backend,
                       agent="team.reader", max_output_tokens=READER_MAX_OUTPUT_TOKENS)
    items = out.evidence[:6]
    checked = verify([Citation(source_id=i.source_id, sent_idx=i.sent_idx, quote=i.quote) for i in items], store)
    meta = {d.source_id: d for d in docs}
    evidence = [EvidenceItem(**c.model_dump(), stance=i.stance,
                             url=meta[c.source_id].url if c.source_id in meta else "",
                             title=store.titles.get(c.source_id, ""))
                for c, i in zip(checked, items)]
    return {"readings": [Reading(claim_id=task["claim_id"], round=task["round"], evidence=evidence, n_candidates=n)]}


# ================================================================ judge

JUDGE_SYSTEM = f"""You are the judge of a fact-checking team. Decide if the claim is TRUE, FALSE or
NOT_ENOUGH_INFO. Use ONLY the numbered evidence, never your own knowledge. Cite evidence by number.

{LABEL_RULES}

confidence: how sure you are (0-1). If below 0.6, suggest up to 2 follow-up searches."""


class JudgeTask(TypedDict):
    claim_id: int
    round: int
    claim: PlannedClaim
    evidence: list[EvidenceItem]
    config: TeamConfig


def dispatch_judge(state: TeamState):
    return [Send("judge", JudgeTask(claim_id=cid, round=state["round"], claim=state["claims"][cid],
                                    evidence=[e for e in latest(state["readings"], cid).evidence if e.valid],
                                    config=state["config"]))
            for cid in state["pending"]]


def judge(task: JudgeTask) -> dict:
    cfg, claim, evidence = task["config"], task["claim"], task["evidence"]
    base = dict(claim_id=task["claim_id"], round=task["round"], judge=cfg.judge)
    if cfg.judge.startswith("lora:"):
        label = judge_lora.judge(claim.english, [e.quote for e in evidence], model=cfg.judge.split(":", 1)[1])
        cited = [e for e in evidence if e.stance != "context"] or evidence
        return {"judgments": [Judgment(**base, label=label, confidence=1.0, explanation="",
                                       citations=cited if label != "NOT_ENOUGH_INFO" else [])]}
    tier = "pro" if cfg.judge == "llm:pro" else cfg.tier
    listing = "\n".join(f"{i}. [{e.stance}] ({e.title}) {e.quote}" for i, e in enumerate(evidence)) or "(no evidence found)"
    out = llm.generate(f"CLAIM: {claim.text}\n(English: {claim.english})\n\nEVIDENCE:\n{listing}",
                       schema=JudgeOut, system=JUDGE_SYSTEM, tier=tier, backend=cfg.backend, agent="team.judge")
    valid_ids = [i for i in dict.fromkeys(out.evidence_ids) if 0 <= i < len(evidence)]
    return {"judgments": [Judgment(**base, label=out.label, confidence=out.confidence, explanation=out.explanation,
                                   citations=[evidence[i] for i in valid_ids],
                                   follow_up_queries=out.follow_up_queries[:2],
                                   invalid_ids=len(set(out.evidence_ids)) - len(valid_ids))]}


def after_judge(state: TeamState) -> dict:
    """Pick the claims that need another search round, with the judge's follow-up queries."""
    cfg, rnd = state["config"], state["round"]
    retry, queries = [], list(state["queries"])
    for cid in state["pending"]:
        j = latest(state["judgments"], cid)
        if j.confidence < CONFIDENCE_THRESHOLD and rnd < cfg.max_rounds and j.follow_up_queries:
            retry.append(cid)
            lang = state["lang"]
            queries[cid] = [Query(query=q, lang="ar" if lang == "ar" and is_arabic(q) else "en")
                            for q in j.follow_up_queries]
    return {"pending": retry, "round": rnd + 1 if retry else rnd, "queries": queries}


def route_after_judge(state: TeamState):
    return dispatch_search(state) if state["pending"] else "writer"


# ================================================================ writer + approval

WRITER_SYSTEM = """You are the writer of a fact-checking team. Write a short headline and, for each
claim, a 1-3 sentence explanation of the verdict for a general reader, in the requested language.
Use only the verdicts and evidence given; add no new facts."""

BADGE = {"TRUE": "✅ TRUE", "FALSE": "❌ FALSE", "NOT_ENOUGH_INFO": "⚠️ NOT ENOUGH INFO"}
BADGE_AR = {"TRUE": "✅ صحيح", "FALSE": "❌ خطأ", "NOT_ENOUGH_INFO": "⚠️ معلومات غير كافية"}


def writer(state: TeamState) -> dict:
    cfg, lang = state["config"], state["lang"]
    finals = [latest(state["judgments"], cid) for cid in range(len(state["claims"]))]
    listing = "\n\n".join(
        f"claim_id {cid}: {c.text}\nverdict: {j.label} (confidence {j.confidence:.2f})\nreason: {j.explanation}\n"
        "evidence:\n" + "\n".join(f"- {e.quote}" for e in j.citations)
        for cid, (c, j) in enumerate(zip(state["claims"], finals)))
    out = llm.generate(f"OUTPUT LANGUAGE: {'Arabic' if lang == 'ar' else 'English'}\n\n{listing}",
                       schema=WriterOut, system=WRITER_SYSTEM, tier=cfg.tier, backend=cfg.backend,
                       agent="team.writer")
    explanations = {str(w.claim_id): w.explanation for w in out.claims}
    return {"headline": out.headline, "explanations": explanations,
            "report": render_report(out.headline, state["claims"], finals, explanations, lang)}


def render_report(headline: str, claims: list[PlannedClaim], finals: list[Judgment],
                  explanations: dict[str, str], lang: str) -> str:
    """Citations are rendered from verified evidence, never written by the LLM."""
    badge = BADGE_AR if lang == "ar" else BADGE
    lines = [f"# {headline}", ""]
    for cid, (c, j) in enumerate(zip(claims, finals)):
        lines += [f"## {cid + 1}. {c.text}", "", f"**{badge[j.label]}** (confidence {j.confidence:.2f})", "",
                  explanations.get(str(cid)) or j.explanation, ""]
        for e in j.citations:
            lines.append(f"- “{e.quote}” — [{e.title or e.source_id}]({e.url}), sentence {e.sent_idx}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def approve(state: TeamState) -> dict:
    decision = interrupt({"report": state["report"]})  # pauses the graph until a human answers
    out: dict = {"decision": decision}
    if decision.get("action") in ("approve", "edit"):
        PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
        path = PUBLISHED_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.md"
        path.write_text(decision.get("content") or state["report"], encoding="utf-8")
        out["published_path"] = str(path)
    return out


# ================================================================ graph


def build_graph(checkpointer=None):
    g = StateGraph(TeamState)
    for name, fn in [("planner", planner), ("search", search), ("gather", barrier), ("reader", reader),
                     ("read_done", barrier), ("judge", judge), ("after_judge", after_judge),
                     ("writer", writer), ("approve", approve)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", dispatch_search, ["search", "gather"])
    g.add_edge("search", "gather")
    g.add_conditional_edges("gather", dispatch_read, ["reader"])
    g.add_edge("reader", "read_done")
    g.add_conditional_edges("read_done", dispatch_judge, ["judge"])
    g.add_edge("judge", "after_judge")
    g.add_conditional_edges("after_judge", route_after_judge, ["search", "gather", "writer"])
    g.add_conditional_edges("writer", lambda s: "approve" if s["config"].publish else END, ["approve", END])
    g.add_edge("approve", END)
    return g.compile(checkpointer=checkpointer)


_STATE_TYPES = [(__name__, n) for n in ("TeamConfig", "PlannedClaim", "Query", "DocBatch", "Reading",
                                        "Judgment", "EvidenceItem")] + [
    ("factcheck.schemas", "Document"), ("factcheck.schemas", "Sentence")]


def checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_STATE_TYPES))


# ================================================================ entry points


def run(text: str, config: TeamConfig | None = None, *, thread_id: str = "cli") -> TeamState:
    """Run the team. With config.publish, asks for approval in the terminal before publishing."""
    config = config or TeamConfig()
    graph = build_graph(checkpointer() if config.publish else None)
    cfg = {"configurable": {"thread_id": thread_id}, "max_concurrency": 8}
    state = graph.invoke({"text": text, "config": config}, cfg)
    while state.get("__interrupt__"):
        report = state["__interrupt__"][0].value["report"]
        d = approval.ask("Fact-check report", report, what="publish")
        state = graph.invoke(Command(resume=d.model_dump()), cfg)
    return state


AGGREGATE = {"FALSE": 0, "NOT_ENOUGH_INFO": 1, "TRUE": 2}


def check(claim: str, *, tier: llm.Tier = "flash", backend: str | None = None, judge: str = "llm",
          searchers: list[str] | None = None) -> CheckResult:
    """Benchmark adapter: one input -> one label (several sub-claims: any FALSE -> FALSE, all TRUE -> TRUE)."""
    cfg = TeamConfig(tier=tier, backend=backend, judge=judge, searchers=searchers or list(ALL_SEARCHERS))
    state = run(claim, cfg)
    finals = [latest(state["judgments"], cid) for cid in range(len(state["claims"]))]
    label = min((j.label for j in finals), key=lambda lab: AGGREGATE[lab])
    citations = [e for j in finals for e in j.citations]
    hallucinated = [e for r in state["readings"] for e in r.evidence if not e.valid]
    sources = list(dict.fromkeys(d.source_id for b in state["docs"] for d in b.docs))
    return CheckResult(
        claim=claim, mode="team", label=label,
        confidence=min(j.confidence for j in finals),
        explanation=" ".join(j.explanation for j in finals),
        citations=[CheckedCitation(**{k: getattr(e, k) for k in CheckedCitation.model_fields}) for e in citations + hallucinated],
        sources_read=sources, steps=state["round"],
        extra={"n_claims": len(finals), "invalid_judge_ids": sum(j.invalid_ids for j in finals),
               "searcher_errors": [b.error for b in state["docs"] if b.error]},
    )
