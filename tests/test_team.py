"""Agent team end-to-end with a fake LLM and fake searchers (no network, no cost)."""

import json

import pytest

from agentcore import llm, trace
from agentcore.llm import RawResult
from factcheck import team
from factcheck.schemas import Document, Sentence

BURJ = Document(source_id="wiki:en:Burj Khalifa", kind="wikipedia", title="Burj Khalifa", url="https://w/Burj",
                lang="en", total_sentences=3, sentences=[
                    Sentence(idx=0, text="The Burj Khalifa is a skyscraper in Dubai."),
                    Sentence(idx=1, text="Construction began in 2004."),
                    Sentence(idx=2, text="The building was completed in 2009 and opened in January 2010.")])
KB = Document(source_id="fever:Burj Khalifa", kind="kb", title="Burj Khalifa", url="https://w/Burj", lang="en",
              total_sentences=9, sentences=[Sentence(idx=0, text="Burj Khalifa is a megatall skyscraper in Dubai.")])


class FakeLLM:
    """Answers by schema name; judge is unsure in round 1, sure in round 2."""

    def __init__(self, judge_rounds=({"conf": 0.4, "follow": ["Burj Khalifa opening date"]}, {"conf": 0.95, "follow": []})):
        self.calls = []
        self.judge_rounds = list(judge_rounds)

    def __call__(self, model, prompt, system, schema, temperature, max_tokens):
        name = schema["title"]
        self.calls.append((name, prompt))
        if name == "PlannerOut":
            out = {"claims": [{"text": "اكتمل برج خليفة في 2015", "english": "Burj Khalifa was completed in 2015",
                               "queries": [{"query": "برج خليفة", "lang": "ar"}, {"query": "Burj Khalifa", "lang": "en"}],
                               "scientific": False}]}
        elif name == "ReaderOut":
            out = {"evidence": [
                {"source_id": "wiki:en:Burj Khalifa", "sent_idx": 2, "stance": "refutes",
                 "quote": "The building was completed in 2009 and opened in January 2010."},
                {"source_id": "wiki:en:Burj Khalifa", "sent_idx": 1, "stance": "refutes",
                 "quote": "It was completed in 2015."}]}  # hallucinated quote -> must be caught
        elif name == "JudgeOut":
            r = self.judge_rounds.pop(0)
            out = {"label": "FALSE", "confidence": r["conf"], "explanation": "Completed in 2009.",
                   "evidence_ids": [0, 7], "follow_up_queries": r["follow"]}  # 7 does not exist
        elif name == "WriterOut":
            out = {"headline": "برج خليفة لم يكتمل في 2015", "claims": [{"claim_id": 0, "explanation": "اكتمل في 2009."}]}
        else:
            raise AssertionError(name)
        return RawResult(text=json.dumps(out, ensure_ascii=False), model=model, input_tokens=10, output_tokens=5)


@pytest.fixture
def fake(monkeypatch):
    f = FakeLLM()
    monkeypatch.setitem(llm.BACKENDS, "fake", f)
    searched = []
    monkeypatch.setitem(team.SEARCHERS, "wiki", lambda t: searched.append(("wiki", t["round"], [q.query for q in t["queries"]])) or [BURJ])
    monkeypatch.setitem(team.SEARCHERS, "kb", lambda t: searched.append(("kb", t["round"], [])) or [KB])
    monkeypatch.setitem(team.SEARCHERS, "papers", lambda t: pytest.fail("papers searched for a non-scientific claim"))
    f.searched = searched
    return f


def test_team_full_flow(fake):
    with trace.collect() as events:
        state = team.run("اكتمل برج خليفة في 2015", team.TeamConfig(backend="fake"))
    # parallel searchers, two rounds (judge unsure -> follow-up queries)
    assert sorted((s, r) for s, r, _ in fake.searched) == [("kb", 1), ("kb", 2), ("wiki", 1), ("wiki", 2)]
    assert ["Burj Khalifa opening date"] in [q for s, r, q in fake.searched if r == 2]
    assert state["lang"] == "ar" and state["round"] == 2
    # reader hallucination caught; judge's bad evidence id dropped
    r1 = state["readings"][0]
    assert [e.valid for e in r1.evidence] == [True, False]
    final = team.latest(state["judgments"], 0)
    assert final.label == "FALSE" and final.confidence == 0.95 and final.invalid_ids == 1
    assert [e.sent_idx for e in final.citations] == [2]
    # judge only ever saw verified evidence
    judge_prompts = [p for n, p in fake.calls if n == "JudgeOut"]
    assert all("It was completed in 2015" not in p for p in judge_prompts)
    # report: Arabic badge, citation rendered from verified evidence with URL
    assert "❌ خطأ" in state["report"] and "(https://w/Burj), sentence 2" in state["report"]
    assert {e["name"] for e in events if e["kind"] == "llm"} == {"team.planner", "team.reader", "team.judge", "team.writer"}
    assert "approve" not in state.get("decision", {})


def test_team_check_adapter_counts_hallucinations(fake):
    r = team.check("Burj Khalifa was completed in 2015", backend="fake")
    assert r.label == "FALSE" and r.mode == "team" and r.steps == 2
    assert r.fake_citations == 2  # one hallucinated reader quote per round
    assert r.extra["invalid_judge_ids"] == 1
    assert set(r.sources_read) == {"wiki:en:Burj Khalifa", "fever:Burj Khalifa"}


def test_team_stops_after_max_rounds(monkeypatch, fake):
    fake.judge_rounds = [{"conf": 0.3, "follow": ["x"]}, {"conf": 0.3, "follow": ["y"]}]
    state = team.run("claim", team.TeamConfig(backend="fake"))
    assert state["round"] == 2 and len(state["judgments"]) == 2


def test_approval_interrupt_and_publish(fake, monkeypatch, tmp_path):
    from agentcore import approval

    monkeypatch.setattr(team, "PUBLISHED_DIR", tmp_path)
    monkeypatch.setattr(approval, "ask", lambda title, content, what: approval.Decision(action="edit", content=content + "\nedited"))
    state = team.run("claim", team.TeamConfig(backend="fake", publish=True), thread_id="t-approve")
    assert state["decision"]["action"] == "edit"
    published = (tmp_path / state["published_path"].split("/")[-1]).read_text()
    assert published.endswith("edited")


def test_rejected_report_is_not_published(fake, monkeypatch, tmp_path):
    from agentcore import approval

    out = tmp_path / "published"
    monkeypatch.setattr(team, "PUBLISHED_DIR", out)
    monkeypatch.setattr(approval, "ask", lambda *a, **k: approval.Decision(action="reject", content=""))
    state = team.run("claim", team.TeamConfig(backend="fake", publish=True), thread_id="t-reject")
    assert "published_path" not in state and not out.exists()


def test_lora_judge_path(fake, monkeypatch):
    seen = {}

    def ollama_fake(model, prompt, system, schema, temperature, max_tokens):
        seen["model"], seen["prompt"] = model, prompt
        return RawResult(text='{"label": "FALSE"}', model=model)

    monkeypatch.setitem(llm.BACKENDS, "ollama", ollama_fake)
    r = team.check("Burj Khalifa was completed in 2015", backend="fake", judge="lora:factjudge")
    assert r.label == "FALSE" and seen["model"] == "factjudge"
    assert seen["prompt"].startswith("Claim: Burj Khalifa was completed in 2015\nEvidence:\n1. The building")


def test_prefilter_keeps_relevant_and_first_sentences():
    doc = Document(source_id="s", kind="kb", title="t", url="", lang="en", total_sentences=100,
                   sentences=[Sentence(idx=i, text=f"filler sentence number {i}") for i in range(100)]
                   + [Sentence(idx=100, text="Burj Khalifa completed 2009")])
    out = team.prefilter("Burj Khalifa completed", [doc], limit=5)
    ids = [s.idx for s in out[0].sentences]
    assert 0 in ids and 100 in ids and len(ids) <= 6
