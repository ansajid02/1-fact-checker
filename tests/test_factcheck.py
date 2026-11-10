import pytest

from agentcore import llm
from agentcore.llm import RawResult
from factcheck import baseline
from factcheck.citations import EvidenceStore, verify
from factcheck.fever import _parse, decode_title
from factcheck.schemas import Citation, Document, SearchHit, Sentence
from factcheck.text import is_arabic, normalize, split_sentences
from factcheck.tools.papers import abstract_from_inverted_index


def test_split_sentences_english():
    text = ("== History ==\nDr. Smith built it in 2010. It is 828 m tall! J. R. R. Tolkien wrote "
            "books. The U.S. economy grew.\n\nShort.")
    assert split_sentences(text) == [
        "Dr. Smith built it in 2010.", "It is 828 m tall!",
        "J. R. R. Tolkien wrote books.", "The U.S. economy grew.", "Short.",
    ]


def test_split_sentences_arabic():
    text = "بُرج خليفة ناطحة سحاب في دبي. افتتح في عام 2010. هل هو الأطول؟ نعم."
    assert len(split_sentences(text)) == 4


def test_normalize_arabic():
    assert normalize("بُرْجُ خَلِيفَة") == normalize("برج خليفه")
    assert normalize("إمارة") == normalize("امارة")
    assert is_arabic("برج خليفة 2010") and not is_arabic("Burj Khalifa")


def test_fever_parsing():
    assert decode_title("Soul_Food_-LRB-film-RRB-") == "Soul Food (film)"
    row = {"id": 1, "label": "REFUTES", "claim": "x",
           "evidence": [[[1, 2, "Telemundo", 0]], [[1, 3, "Telemundo", 4], [1, 3, "Hispanic_and_Latino_Americans", 0]]]}
    c = _parse(row)
    assert c.label == "FALSE" and c.evidence_titles == ["Hispanic and Latino Americans", "Telemundo"]
    nei = _parse({"id": 2, "label": "NOT ENOUGH INFO", "claim": "y", "evidence": [[[5, None, None, None]]]})
    assert nei.label == "NOT_ENOUGH_INFO" and nei.evidence_sets == []


def test_openalex_abstract():
    assert abstract_from_inverted_index({"Coffee": [0], "is": [1], "good.": [2]}) == "Coffee is good."


def _doc():
    return Document(source_id="wiki:en:Burj Khalifa", kind="wikipedia", title="Burj Khalifa", url="u",
                    lang="en", total_sentences=2,
                    sentences=[Sentence(idx=0, text="The Burj Khalifa is a skyscraper in Dubai."),
                               Sentence(idx=1, text="It was completed in 2010.")])


def test_citation_verification():
    store = EvidenceStore()
    store.add(_doc())
    out = verify([
        Citation(source_id="wiki:en:Burj Khalifa", sent_idx=1, quote="It was completed in 2010"),
        Citation(source_id="wiki:en:Burj Khalifa", sent_idx=0, quote="It was completed in 2010."),
        Citation(source_id="wiki:en:Burj Khalifa", sent_idx=1, quote="It opened in 2015."),
        Citation(source_id="wiki:en:Burj Khalifa", sent_idx=9, quote="x"),
        Citation(source_id="wiki:en:Nope", sent_idx=0, quote="x"),
    ], store)
    assert [c.valid for c in out] == [True, False, False, False, False]
    assert [c.reason for c in out[1:]] == [
        "quote exists but under a different sentence index", "quote does not match the sentence",
        "sentence index was never read", "source was never read"]


def test_baseline_loop_with_fake_llm(monkeypatch):
    """Full ReAct loop: search -> fetch -> finish -> verdict, with fake LLM and tools."""
    replies = iter([
        '{"thought": "search", "action": "wikipedia_search", "query": "Burj Khalifa"}',
        '{"thought": "read", "action": "fetch_page", "title": "Burj Khalifa"}',
        '{"thought": "read again", "action": "fetch_page", "title": "Burj Khalifa"}',
        '{"thought": "done", "action": "finish"}',
        '{"label": "FALSE", "confidence": 0.9, "explanation": "Completed in 2010.",'
        ' "citations": [{"source_id": "wiki:en:Burj Khalifa", "sent_idx": 1, "quote": "It was completed in 2010."},'
        ' {"source_id": "wiki:en:Burj Khalifa", "sent_idx": 1, "quote": "Completed in 2015."}]}',
    ])
    prompts = []

    def fake(model, prompt, system, schema, temperature, max_tokens):
        prompts.append(prompt)
        return RawResult(text=next(replies), model=model)

    monkeypatch.setitem(llm.BACKENDS, "fake", fake)
    monkeypatch.setattr(baseline, "wikipedia_search",
                        lambda q, lang="en": [SearchHit(title="Burj Khalifa", snippet="tall", lang="en")])
    fetches = []
    monkeypatch.setattr(baseline, "fetch_page", lambda t, lang="en", start=0: fetches.append(t) or _doc())

    r = baseline.check("The Burj Khalifa was completed in 2015.", backend="fake")
    assert r.label == "FALSE" and r.steps == 4
    assert len(fetches) == 1  # repeated identical call is blocked
    assert "already made this exact call" in prompts[3]
    assert "[1] It was completed in 2010." in prompts[-1]
    assert [c.valid for c in r.citations] == [True, False] and r.fake_citations == 1
    assert r.sources_read == ["wiki:en:Burj Khalifa"]


@pytest.mark.parametrize("bad", ['{"label": "MAYBE", "confidence": 0.5, "explanation": "x"}',
                                 '{"label": "TRUE", "confidence": 1.5, "explanation": "x"}'])
def test_verdict_schema_rejects_bad_values(bad):
    from pydantic import ValidationError

    from factcheck.schemas import VerdictDraft

    with pytest.raises(ValidationError):
        VerdictDraft.model_validate_json(bad)


def test_wikimedia_block_is_fatal_not_a_tool_error(monkeypatch):
    import httpx

    from factcheck.tools import http

    blocked = httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(403, text="Error: 403, Please respect our robot policy")))
    monkeypatch.setattr(http, "client", lambda: blocked)
    with pytest.raises(http.AccessBlocked, match="HTTP_USER_AGENT"):
        http.get_json("https://en.wikipedia.org/w/api.php", {"q": 1}, use_cache=False)
