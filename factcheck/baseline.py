"""Single-agent baseline: ONE ReAct agent with all the tools.

Each step the LLM returns an AgentStep (validated JSON): a thought plus one tool call.
When it chooses "finish" (or runs out of steps) it returns a VerdictDraft. Citations are
then checked against the sentences it actually read.

The agent team (step 7) must beat this on the same claims.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentcore import llm

from .citations import EvidenceStore, verify
from .text import is_arabic
from .schemas import CheckResult, Document, SearchHit, VerdictDraft
from .tools import SEARCHER_TOOLS, ToolError, fetch_page, paper_search, tool_docs, wikipedia_search

MAX_STEPS = 8
MAX_OBSERVATION_CHARS = 6000


class AgentStep(BaseModel):
    thought: str = Field(description="short reasoning about what to do next")
    action: Literal["wikipedia_search", "fetch_page", "kb_search", "paper_search", "finish"]
    query: str = Field("", description="search query (wikipedia_search, kb_search, paper_search)")
    title: str = Field("", description="exact page title (fetch_page)")
    lang: Literal["en", "ar"] = "en"
    start: int = Field(0, description="first sentence index to read (fetch_page)")


SYSTEM_TEMPLATE = """You are a careful fact-checker. You decide whether a claim is TRUE, FALSE or
NOT_ENOUGH_INFO using ONLY evidence you read with your tools, never your own memory.

Tools:
{tools}

Method: search Wikipedia for the main entity, fetch the most relevant page, read the sentences.
Use lang "ar" for Arabic claims (and English Wikipedia as a fallback). Use paper_search only
for scientific or medical claims. Do not repeat an identical call.

Labels:
- TRUE: the evidence you read supports the claim.
- FALSE: the evidence you read contradicts the claim.
- NOT_ENOUGH_INFO: you could not find evidence that settles it either way.
"""
ALL_SEARCHERS = ["wiki", "kb", "papers"]
SYSTEM = SYSTEM_TEMPLATE.format(tools=tool_docs(ALL_SEARCHERS))

FINAL_INSTRUCTIONS = """Now give your verdict as JSON.
- Base the label only on the observations above.
- Cite the sentences that support or contradict the claim: source_id exactly as shown,
  sent_idx = the [number] before the sentence, quote = that sentence copied exactly.
  Cite at most 3. Only cite sentences that appear in the observations.
- For NOT_ENOUGH_INFO, citations may be empty.
- confidence: 0-1, how sure you are of the label.
- explanation: 1-3 sentences, written in {language} (the language of the claim)."""


def _render_call(step: AgentStep) -> str:
    if step.action == "fetch_page":
        return f'fetch_page(title={step.title!r}, lang="{step.lang}", start={step.start})'
    if step.action == "wikipedia_search":
        return f'wikipedia_search(query={step.query!r}, lang="{step.lang}")'
    if step.action in ("paper_search", "kb_search"):
        return f"{step.action}(query={step.query!r})"
    return "finish"


def _render_doc(d: Document) -> str:
    # source_id on its own line: models copy it verbatim into citations
    lines = [f"source_id: {d.source_id}"]
    if not d.sentences:
        return "\n".join(lines + [f"title: {d.title} (no text)"])
    first, last = d.sentences[0].idx, d.sentences[-1].idx
    lines.append(f"title: {d.title} (showing sentences {first}-{last} of {d.total_sentences})")
    lines += [f"[{s.idx}] {s.text}" for s in d.sentences]
    return "\n".join(lines)


def _render_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "no results"
    return "\n".join(f"- {h.title} ({h.lang}): {h.snippet}" for h in hits)


def run_tool(step: AgentStep, store: EvidenceStore, allowed: set[str] | None = None) -> str:
    """Execute one tool call; register citable sentences; return the observation text."""
    if allowed is not None and step.action not in allowed:
        return f"ERROR: tool '{step.action}' is not available in this run."
    try:
        if step.action == "wikipedia_search":
            obs = _render_hits(wikipedia_search(step.query, lang=step.lang))
        elif step.action == "fetch_page":
            doc = fetch_page(step.title, lang=step.lang, start=step.start)
            store.add(doc)
            obs = _render_doc(doc)
        elif step.action == "kb_search":
            from .kb.retriever import kb_search

            docs = kb_search(step.query)
            for d in docs:
                store.add(d)
            obs = "\n\n".join(_render_doc(d) for d in docs) or "no results"
        elif step.action == "paper_search":
            docs = paper_search(step.query)
            for d in docs:
                store.add(d)
            obs = "\n\n".join(_render_doc(d) for d in docs) or "no results"
        else:
            obs = ""
    except (ToolError, FileNotFoundError) as e:
        obs = f"ERROR: {e}"
    if len(obs) > MAX_OBSERVATION_CHARS:
        obs = obs[:MAX_OBSERVATION_CHARS] + "\n...(truncated; use fetch_page with a higher start)"
    return obs


def check(claim: str, *, tier: llm.Tier = "flash", backend: str | None = None,
          max_steps: int = MAX_STEPS, searchers: list[str] | None = None, judge: str = "llm") -> CheckResult:
    """`judge` is accepted for interface parity with the team; the single agent always judges itself."""
    searchers = searchers or ALL_SEARCHERS
    allowed = {t for s in searchers for t in SEARCHER_TOOLS[s]} | {"finish"}
    system = SYSTEM_TEMPLATE.format(tools=tool_docs(searchers))
    store = EvidenceStore()
    history: list[str] = []
    seen_calls: set[str] = set()
    steps = 0

    for steps in range(1, max_steps + 1):
        prompt = (
            f"CLAIM: {claim}\n\nHISTORY:\n" + ("\n\n".join(history) or "(nothing yet)") +
            f"\n\nYou have {max_steps - steps + 1} step(s) left. Choose the next action."
        )
        step = llm.generate(prompt, schema=AgentStep, system=system, tier=tier,
                            backend=backend, agent="baseline.step")
        call = _render_call(step)
        if step.action == "finish":
            history.append(f"Step {steps}\nthought: {step.thought}\naction: finish")
            break
        if call in seen_calls:
            obs = "You already made this exact call. Try something different, or finish."
        else:
            seen_calls.add(call)
            obs = run_tool(step, store, allowed)
        history.append(f"Step {steps}\nthought: {step.thought}\naction: {call}\nobservation:\n{obs}")

    final_prompt = f"CLAIM: {claim}\n\nOBSERVATIONS:\n" + "\n\n".join(history) + "\n\n" + FINAL_INSTRUCTIONS.format(
        language="Arabic" if is_arabic(claim) else "English")
    draft = llm.generate(final_prompt, schema=VerdictDraft, system=system, tier=tier,
                         backend=backend, agent="baseline.verdict")
    return CheckResult(
        claim=claim,
        mode="single",
        label=draft.label,
        confidence=draft.confidence,
        explanation=draft.explanation,
        citations=verify(draft.citations, store),
        sources_read=store.sources(),
        steps=steps,
    )
