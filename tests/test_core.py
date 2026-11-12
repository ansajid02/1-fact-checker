import json

import pytest
from pydantic import BaseModel

from agentcore import evals, llm, trace
from agentcore.llm import RawResult


class Answer(BaseModel):
    value: int


def fake_backend(replies):
    calls = []

    def call(model, prompt, system, json_schema, temperature, max_tokens):
        calls.append({"prompt": prompt, "schema": json_schema})
        return RawResult(text=replies[len(calls) - 1], model=model, input_tokens=10, output_tokens=5)

    return call, calls


def test_structured_output_and_trace(monkeypatch, tmp_traces):
    call, calls = fake_backend(['```json\n{"value": 7}\n```'])
    monkeypatch.setitem(llm.BACKENDS, "fake", call)
    with trace.run("t") as run_id, trace.collect() as events:
        out = llm.generate("q", schema=Answer, backend="fake", agent="tester")
    assert out == Answer(value=7)
    assert calls[0]["schema"]["properties"]["value"]["type"] == "integer"
    assert events[0]["kind"] == "llm" and events[0]["input_tokens"] == 10 and events[0]["ok"]
    on_disk = [json.loads(l) for l in (tmp_traces / f"{run_id}.jsonl").read_text().splitlines()]
    assert on_disk[0]["name"] == "tester"


def test_retry_on_invalid_json(monkeypatch):
    call, calls = fake_backend(['{"value": "seven"}', '{"value": 7}'])
    monkeypatch.setitem(llm.BACKENDS, "fake", call)
    with trace.collect() as events:
        assert llm.generate("q", schema=Answer, backend="fake").value == 7
    assert "did not match the required JSON schema" in calls[1]["prompt"]
    assert "schema_error" in events[0] and "schema_error" not in events[1]


def test_cut_off_reply_retries_without_constrained_decoding(monkeypatch):
    """Qwen on Bedrock can loop on whitespace under response_format until max_tokens."""
    calls = []

    def call(model, prompt, system, json_schema, temperature, max_tokens):
        calls.append({"prompt": prompt, "schema": json_schema})
        if json_schema:
            return RawResult(text='{"value": 7' + " " * 50, model=model, input_tokens=10,
                             output_tokens=4096, finish_reason="length")
        return RawResult(text='```json\n{"value": 7}\n```', model=model, input_tokens=10, output_tokens=5)

    monkeypatch.setitem(llm.BACKENDS, "fake", call)
    with trace.collect() as events:
        assert llm.generate("q", schema=Answer, backend="fake").value == 7
    assert calls[0]["schema"] is not None and calls[1]["schema"] is None
    assert '"value"' in calls[1]["prompt"] and "JSON schema" in calls[1]["prompt"]
    assert events[0]["constrained"] is True and events[1]["constrained"] is False


def test_gives_up_after_retries(monkeypatch):
    call, _ = fake_backend(["nope", "still nope"])
    monkeypatch.setitem(llm.BACKENDS, "fake", call)
    with pytest.raises(llm.LLMError):
        llm.generate("q", schema=Answer, backend="fake")


def test_tags_and_summary():
    with trace.collect() as events, trace.tags(claim_id=1):
        trace.log("llm", "a", input_tokens=3, output_tokens=2, cost_usd=0.5)
        trace.log("llm", "a", input_tokens=1, output_tokens=1, cost_usd=None)
        trace.log("tool", "search", ok=False)
    assert all(e["claim_id"] == 1 for e in events)
    s = trace.summarize(events)
    assert s["total"]["input_tokens"] == 4 and s["total"]["cost_usd"] == 0.5
    assert s["total"]["cost_unknown"] == 1 and s["total"]["errors"] == 1
    assert s["total"]["llm_calls"] == 2 and s["total"]["tool_calls"] == 1


def test_classification_report():
    r = evals.classification_report(["A", "A", "B", "B"], ["A", "B", "B", "ERROR"], ["A", "B"])
    assert r["accuracy"] == 0.5
    assert r["per_class"]["A"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2}
    assert r["confusion"]["B"]["ERROR"] == 1


def test_openrouter_request_and_usage(monkeypatch):
    import httpx

    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "model": "google/test-flash",
            "choices": [{"message": {"content": '{"value": 3}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "cost": 0.00042},
        })

    client = httpx.Client(base_url="https://openrouter.test/api/v1", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm, "_openrouter_client", lambda: client)
    with trace.collect() as events:
        out = llm.generate("q", schema=Answer, backend="openrouter", model="google/test-flash", system="sys")
    assert out.value == 3
    body = seen["body"]
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["title"] == "Answer"
    assert body["provider"] == {"require_parameters": True}
    assert events[0]["cost_usd"] == 0.00042 and events[0]["input_tokens"] == 12


def test_openrouter_error_body_raises(monkeypatch):
    import httpx

    client = httpx.Client(base_url="https://x/api/v1", transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"error": {"message": "no endpoints"}})))
    monkeypatch.setattr(llm, "_openrouter_client", lambda: client)
    with pytest.raises(llm.LLMError, match="no endpoints"):
        llm.generate("q", backend="openrouter", model="m")


def test_empty_env_value_means_default(monkeypatch):
    from agentcore import config

    monkeypatch.setenv("LLM_BACKEND", "")
    monkeypatch.setenv("EMBED_MODEL", "  ")
    config.settings.cache_clear()
    try:
        assert config.settings().llm_backend == "bedrock"
        assert config.settings().embed_model == "baai/bge-m3"
    finally:
        config.settings.cache_clear()


def test_approval_shows_llm_text_verbatim(monkeypatch, capsys):
    from agentcore import approval

    monkeypatch.setattr(approval.sys.stdin, "isatty", lambda: False)
    d = approval.ask("t [x]", "verdict [MEETS] and a stray [/x] tag", what="publish")
    assert d.action == "reject"  # no terminal -> safe default
    out = capsys.readouterr().out
    assert "[MEETS]" in out and "[/x]" in out


def test_bedrock_request_and_usage(monkeypatch):
    from types import SimpleNamespace as NS

    sent = {}

    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            return NS(stop_reason="end_turn", stop_details=None,
                      content=[NS(type="thinking", thinking=""), NS(type="text", text='{"value": 42}')],
                      usage=NS(input_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=None,
                               output_tokens=5))

    monkeypatch.setattr(llm, "_bedrock_client", lambda: NS(messages=FakeMessages()))
    with trace.run("t-bedrock"), trace.collect() as events:
        out = llm.generate("q", schema=Answer, backend="bedrock", model="anthropic.claude-haiku-5-5", system="sys")
    assert out.value == 42
    assert sent["system"] == "sys" and "temperature" not in sent
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["format"]["schema"]["additionalProperties"] is False
    span = [e for e in events if e["kind"] == "llm"][0]
    assert span["input_tokens"] == 10 and span["output_tokens"] == 5 and span["backend"] == "bedrock"


def test_bedrock_refusal_raises(monkeypatch):
    from types import SimpleNamespace as NS

    resp = NS(stop_reason="refusal", stop_details=NS(category="cyber"), content=[], usage=None)
    monkeypatch.setattr(llm, "_bedrock_client", lambda: NS(messages=NS(create=lambda **kw: resp)))
    with pytest.raises(llm.LLMError, match="refused"):
        llm.generate("q", backend="bedrock", model="anthropic.test")


def test_bedrock_non_claude_uses_openai_route(monkeypatch):
    import httpx

    def handler(req):
        body = json.loads(req.content)
        assert req.url.path == "/v1/chat/completions" and body["model"] == "qwen.test"
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"value": 7}'}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    client = httpx.Client(base_url="https://bedrock-mantle.test/v1", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm, "_bedrock_openai_client", lambda: client)
    with trace.run("t-bedrock-oai"):
        assert llm.generate("q", schema=Answer, backend="bedrock", model="qwen.test").value == 7
