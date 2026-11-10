"""Steps 6, 9, 10, 11 offline: retrieval fusion, benchmark metrics, AraFacts rules, LoRA data."""

import json
import random

import numpy as np
import pytest

from agentcore import embed, llm
from agentcore.llm import RawResult
from factcheck import bench, report
from factcheck.arafacts import is_question
from factcheck.fever import FeverClaim
from factcheck.kb.retriever import rrf
from factcheck.kb.wikidump import detok, parse_line, raw_title
from factcheck.lora_data import build_evidence
from factcheck.schemas import CheckedCitation, CheckResult, Sentence
from factcheck.kb.wikidump import WikiPage


# ---------------------------------------------------------------- step 6: KB


def test_wikidump_line_parsing():
    line = json.dumps({"id": "Soul_Food_-LRB-film-RRB-", "text": "x",
                       "lines": "0\tSoul Food is a 1997 film -LRB- comedy -RRB- .\tlink\n1\t\n2\tIt grossed $ 43 million ."})
    assert raw_title(line) == "Soul Food (film)"
    page = parse_line(line)
    assert page.title == "Soul Food (film)"
    assert [(s.idx, s.text) for s in page.sentences] == [(0, "Soul Food is a 1997 film (comedy)."), (2, "It grossed $43 million.")]
    assert detok("Burj Khalifa , known as `` Burj Dubai '' , is tall .") == 'Burj Khalifa, known as " Burj Dubai ", is tall.'


def test_rrf_rewards_agreement():
    fused = rrf({"bm25": [1, 2, 3], "dense": [3, 1, 4]})
    assert [cid for cid, _, _ in fused][:2] == [1, 3]
    assert fused[0][2] == {"bm25": 1, "dense": 2}


def test_embed_normalises_and_batches(monkeypatch):
    calls = []

    def fake(model, texts, kind):
        calls.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32) * 3, 7, 0.001

    monkeypatch.setitem(embed.BACKENDS, "fake", fake)
    v = embed.embed(["a"] * 70, backend="fake", model="m")
    assert calls == [64, 6] and v.shape == (70, 4)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


# ---------------------------------------------------------------- step 9: benchmark


def _claim(label="FALSE", sets=None):
    return FeverClaim(id=1, claim="c", label=label, fever_label="x",
                      evidence_sets=sets if sets is not None else [[("Burj Khalifa", 2)], [("Dubai", 0), ("Burj Khalifa", 5)]])


def _result(label, cites):
    return CheckResult(claim="c", mode="team", label=label, citations=[
        CheckedCitation(source_id=s, sent_idx=i, quote="q", valid=v) for s, i, v in cites])


def test_fever_score_needs_full_evidence_set():
    m = bench.evidence_metrics(_claim(), _result("FALSE", [("fever:Burj Khalifa", 2, True)]))
    assert m["fever_score"] and m["fever_score_page"] and m["gold_page_cited"] and m["kb_citations_gold"] == 1
    # right label, gold page via live Wikipedia only: page-level yes, strict no
    m = bench.evidence_metrics(_claim(), _result("FALSE", [("wiki:en:Burj Khalifa", 7, True)]))
    assert not m["fever_score"] and m["fever_score_page"]
    # invalid (hallucinated) citations never count
    m = bench.evidence_metrics(_claim(), _result("FALSE", [("fever:Burj Khalifa", 2, False)]))
    assert not m["fever_score"] and not m["gold_page_cited"]
    # half of a 2-sentence set is not enough
    m = bench.evidence_metrics(_claim(sets=[[("Dubai", 0), ("Burj Khalifa", 5)]]),
                               _result("FALSE", [("fever:Dubai", 0, True)]))
    assert not m["fever_score"]
    # NEI: label alone is enough
    m = bench.evidence_metrics(_claim("NOT_ENOUGH_INFO", sets=[]), _result("NOT_ENOUGH_INFO", []))
    assert m["fever_score"] and m["gold_page_cited"] is None


def test_bench_run_end_to_end(monkeypatch, tmp_path):
    items = [_claim("FALSE"), _claim("TRUE").model_copy(update={"id": 2})]

    def fake_mode(claim, **kw):
        return _result("FALSE", [("fever:Burj Khalifa", 2, True), ("fever:X", 1, False)])

    monkeypatch.setitem(bench.MODES, "fake", fake_mode)
    monkeypatch.setattr(bench, "load_dataset", lambda name: items)
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path)
    s = bench.run(mode="fake", searchers=["kb"], workers=2)
    assert s["report"]["accuracy"] == 0.5 and s["fake_citation_rate"] == 0.5
    assert s["fever_score"] == 0.5 and s["kb_citation_precision"] == 1.0  # invalid citation excluded
    rows = [json.loads(l) for l in open(s["results_file"])]
    assert {r["id"] for r in rows} == {1, 2}
    # resume skips finished claims
    s2 = bench.run(mode="fake", searchers=["kb"], resume=tmp_path / s["results_file"].split("/")[-1])
    assert s2["n"] == 2


def test_compare_writes_markdown(monkeypatch, tmp_path):
    summary = {"dataset": "fever", "mode": "team", "tier": "flash", "model": "g/flash", "judge": "llm", "searchers": ["kb"], "n": 3,
               "report": {"accuracy": 0.5, "macro_f1": 0.4}, "fever_score": 0.3, "gold_page_cited": 0.9,
               "fake_citation_rate": 0.01, "avg_cost_usd": 0.002, "cost_complete": True, "avg_latency_s": 4.2,
               "errors": 0}
    (tmp_path / "x.summary.json").write_text(json.dumps(summary))
    readme = tmp_path / "README.md"
    readme.write_text(f"a\n{report.START}\nold\n{report.END}\nb\n")
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(report, "PROJECT_ROOT", tmp_path)
    md = report.compare()
    assert "| fever | team | g/flash | llm | kb | 3 | 50.0% | 0.400 | 30.0% | 90.0% | 1.0% | 0.0020 | 4.2 | 0 |" in md
    assert "\nold\n" not in readme.read_text() and readme.read_text().startswith("a\n")


# ---------------------------------------------------------------- step 10 / 11


def test_arafacts_questions_are_dropped():
    assert is_question("هل يخالف كازينو لبنان القانون؟") and is_question("هل هذا صحيح")
    assert not is_question("وفاة قاسم الملاّك")


def test_lora_evidence_construction():
    pages = {"Burj Khalifa": WikiPage(title="Burj Khalifa", sentences=[
        Sentence(idx=0, text="A tower in Dubai."), Sentence(idx=1, text="Opened 2010."), Sentence(idx=2, text="Tallest.")])}
    by_base = {"burj khalifa": ["Burj Khalifa"]}
    ev = build_evidence(_claim(sets=[[("Burj Khalifa", 1)]]), pages, by_base, random.Random(0))
    assert "Burj Khalifa: Opened 2010." in ev and len(ev) == 3
    assert build_evidence(_claim(sets=[[("Missing Page", 0)]]), pages, by_base, random.Random(0)) is None
    nei = FeverClaim(id=3, claim="Burj Khalifa is purple.", label="NOT_ENOUGH_INFO", fever_label="x", evidence_sets=[])
    assert build_evidence(nei, pages, by_base, random.Random(0)) == ["Burj Khalifa: A tower in Dubai.", "Burj Khalifa: Opened 2010."]


def test_judge_eval_same_prompt_for_every_judge(monkeypatch):
    from factcheck import judge_eval, judge_lora

    prompts = []

    def fake(model, prompt, system, schema, temperature, max_tokens):
        prompts.append((model, prompt, system))
        return RawResult(text='{"label": "TRUE"}', model=model)

    monkeypatch.setitem(llm.BACKENDS, "ollama", fake)
    monkeypatch.setattr(llm, "_model_name", lambda backend, tier, model: model or f"cloud-{tier}")
    monkeypatch.setitem(llm.BACKENDS, "openrouter", fake)
    monkeypatch.setitem(llm.BACKENDS, "bedrock", fake)
    item = {"claim": "c", "evidence": ["e1"], "label": "TRUE"}
    assert judge_eval.run_judge("ollama:factjudge", item) == "TRUE"
    assert judge_eval.run_judge("llm:flash", item) == "TRUE"
    assert prompts[0][1:] == prompts[1][1:] == (judge_lora.prompt("c", ["e1"]), judge_lora.SYSTEM)
    with pytest.raises(ValueError):
        judge_eval.run_judge("nope:x", item)
