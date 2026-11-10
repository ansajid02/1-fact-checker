"""The only way agent code talks to an LLM.

    from agentcore import llm
    verdict = llm.generate(prompt, schema=Verdict, tier="flash", agent="judge")

- backend "bedrock": Claude in Amazon Bedrock (Anthropic SDK, Mantle Messages API) - current default
- backend "openrouter": any model on OpenRouter (OpenAI-compatible API)
- backend "vertex": Google Vertex AI (Gemini), via the google-genai SDK
- backend "ollama": a local model (used later for the LoRA judge)
- backend=None -> LLM_BACKEND from .env
- schema: a Pydantic model -> JSON-schema-constrained output, validated, retried once on failure
- every call is traced (tokens, cost, latency) via agentcore.trace
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, TypeVar, overload

import httpx
from pydantic import BaseModel, ValidationError

from . import trace
from .config import settings
from .pricing import cost_usd

T = TypeVar("T", bound=BaseModel)
Tier = Literal["flash", "pro"]


class LLMError(RuntimeError):
    pass


@dataclass
class RawResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    cost_usd: float | None = None  # set when the provider reports the real cost (OpenRouter)


# A backend takes (model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult
Backend = Callable[[str, str, str | None, dict | None, float, int | None], RawResult]


# ---------------------------------------------------------------- Amazon Bedrock (Claude)


@lru_cache(maxsize=1)
def _bedrock_client():
    from anthropic import AnthropicBedrockMantle

    import os

    s = settings()
    has_aws_creds = (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")
                     or (Path.home() / ".aws" / "credentials").exists() or (Path.home() / ".aws" / "config").exists())
    if not s.bedrock_api_key and not has_aws_creds:
        raise LLMError("AWS_BEARER_TOKEN_BEDROCK is not set in .env (and no AWS credentials found). "
                       "Create a Bedrock API key in the AWS console (Bedrock -> API keys) and paste it there.")
    # With no key, the SDK falls back to the normal AWS credential chain (IAM / SSO profile).
    return AnthropicBedrockMantle(aws_region=s.bedrock_region, api_key=s.bedrock_api_key or None,
                                  timeout=120.0, max_retries=4)


@lru_cache(maxsize=1)
def _bedrock_openai_client() -> httpx.Client:
    s = settings()
    if not s.bedrock_api_key:
        raise LLMError("AWS_BEARER_TOKEN_BEDROCK is not set in .env (needed for non-Claude Bedrock models).")
    return httpx.Client(base_url=f"https://bedrock-mantle.{s.bedrock_region}.api.aws/v1",
                        headers={"Authorization": f"Bearer {s.bedrock_api_key}"}, timeout=httpx.Timeout(120.0))


def _call_bedrock_openai(model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult:
    """Non-Claude Bedrock models (Qwen, Mistral, Gemma, ...) via Bedrock's OpenAI-compatible endpoint."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body: dict = {"model": model, "messages": messages, "temperature": temperature,
                  "max_tokens": max_output_tokens or 4096}
    if json_schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": json_schema.get("title", "output"), "strict": False, "schema": json_schema}}
    delay = 2.0
    for attempt in range(5):
        r = _bedrock_openai_client().post("/chat/completions", json=body)
        if r.status_code not in _RETRY_STATUS or attempt == 4:
            break
        time.sleep(delay)
        delay = min(delay * 2, 30)
    if r.status_code != 200:
        raise LLMError(f"Bedrock HTTP {r.status_code} for '{model}': {r.text[:500]}")
    data = r.json()
    if not data.get("choices"):
        raise LLMError(f"Bedrock error for '{model}': {json.dumps(data)[:500]}")
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return RawResult(
        text=choice.get("message", {}).get("content") or "",
        model=model,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        finish_reason=choice.get("finish_reason"),
    )


def _call_bedrock(model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult:
    """anthropic.* models -> Claude Messages API; every other Bedrock model -> OpenAI-compatible API."""
    if not model.startswith("anthropic."):
        return _call_bedrock_openai(model, prompt, system, json_schema, temperature, max_output_tokens)
    import anthropic

    s = settings()
    kw: dict = {"model": model, "max_tokens": max_output_tokens or 4096,
                "messages": [{"role": "user", "content": prompt}]}
    if system:
        kw["system"] = system
    # temperature is not sent: current Claude models reject sampling parameters.
    # Thinking stays at the model default; effort bounds how much it thinks (and costs).
    output_config: dict = {}
    if s.bedrock_effort and s.bedrock_effort != "none":
        output_config["effort"] = s.bedrock_effort
    if json_schema:
        # transform_schema drops constraints the API can't enforce (pydantic still validates them)
        output_config["format"] = {"type": "json_schema", "schema": anthropic.transform_schema(json_schema)}
    if output_config:
        kw["output_config"] = output_config
    try:
        resp = _bedrock_client().messages.create(**kw)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise LLMError(f"Bedrock {e.status_code}: check AWS_BEARER_TOKEN_BEDROCK / AWS_REGION in .env and that "
                       f"model access for '{model}' is enabled in the Bedrock console. {str(e)[:300]}") from e
    except anthropic.NotFoundError as e:
        raise LLMError(f"Bedrock 404 for model '{model}' in {s.bedrock_region}: {str(e)[:300]}") from e
    except anthropic.APIStatusError as e:  # 429 / 5xx were already retried by the SDK
        raise LLMError(f"Bedrock HTTP {e.status_code}: {str(e)[:500]}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(f"Bedrock connection error: {e}") from e
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        raise LLMError(f"Bedrock: model refused (category={cat})")
    u = resp.usage
    return RawResult(
        text="".join(b.text for b in resp.content if b.type == "text"),
        model=model,
        input_tokens=(u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
        output_tokens=u.output_tokens or 0,  # includes thinking tokens
        finish_reason=resp.stop_reason,
    )


# ---------------------------------------------------------------- OpenRouter

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


@lru_cache(maxsize=1)
def _openrouter_client() -> httpx.Client:
    s = settings()
    if not s.openrouter_api_key:
        raise LLMError("OPENROUTER_API_KEY is not set in .env")
    return httpx.Client(
        base_url=s.openrouter_base_url,
        headers={"Authorization": f"Bearer {s.openrouter_api_key}", "X-Title": "agent-lab"},
        timeout=httpx.Timeout(120.0),
    )


def _call_openrouter(model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body: dict = {"model": model, "messages": messages, "temperature": temperature}
    if max_output_tokens:
        body["max_tokens"] = max_output_tokens
    if json_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": json_schema.get("title", "output"), "strict": False, "schema": json_schema},
        }
        # only route to providers that actually honour response_format
        body["provider"] = {"require_parameters": True}

    delay = 2.0
    for attempt in range(5):
        r = _openrouter_client().post("/chat/completions", json=body)
        if r.status_code not in _RETRY_STATUS or attempt == 4:
            break
        time.sleep(delay)
        delay = min(delay * 2, 30)
    if r.status_code != 200:
        raise LLMError(f"OpenRouter HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    if "error" in data or not data.get("choices"):
        raise LLMError(f"OpenRouter error: {json.dumps(data.get('error', data))[:500]}")
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return RawResult(
        text=choice.get("message", {}).get("content") or "",
        model=data.get("model", model),
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),  # includes reasoning tokens
        finish_reason=choice.get("finish_reason"),
        cost_usd=usage.get("cost"),
    )


# ---------------------------------------------------------------- Vertex AI


@lru_cache(maxsize=1)
def _vertex_client():
    from google import genai
    from google.genai import types

    s = settings()
    if not s.gcp_project:
        raise LLMError("GOOGLE_CLOUD_PROJECT is not set in .env (see PLAN.txt section 8).")
    return genai.Client(
        enterprise=True,  # = Vertex AI ("vertexai=True" is the legacy name in google-genai v3)
        project=s.gcp_project,
        location=s.gcp_location,
        http_options=types.HttpOptions(
            timeout=120_000,  # ms
            retry_options=types.HttpRetryOptions(
                attempts=5, initial_delay=2.0, max_delay=30.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )


def _call_vertex(model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult:
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json" if json_schema else None,
        response_json_schema=json_schema,
    )
    resp = _vertex_client().models.generate_content(model=model, contents=prompt, config=config)
    um = resp.usage_metadata
    finish = None
    if resp.candidates and resp.candidates[0].finish_reason is not None:
        finish = str(resp.candidates[0].finish_reason)
    return RawResult(
        text=resp.text or "",
        model=model,
        input_tokens=(um.prompt_token_count or 0) if um else 0,
        # thinking tokens are billed as output tokens
        output_tokens=((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0,
        finish_reason=finish,
    )


# ---------------------------------------------------------------- Ollama (local)


def _call_ollama(model, prompt, system, json_schema, temperature, max_output_tokens) -> RawResult:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    options: dict = {"temperature": temperature}
    if max_output_tokens:
        options["num_predict"] = max_output_tokens
    body = {"model": model, "messages": messages, "stream": False, "options": options}
    if json_schema:
        body["format"] = json_schema
    r = httpx.post(f"{settings().ollama_url}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    data = r.json()
    return RawResult(
        text=data.get("message", {}).get("content", ""),
        model=model,
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        finish_reason=data.get("done_reason"),
    )


BACKENDS: dict[str, Backend] = {
    "bedrock": _call_bedrock,
    "openrouter": _call_openrouter,
    "vertex": _call_vertex,
    "ollama": _call_ollama,
}


def _model_name(backend: str, tier: Tier, model: str | None) -> str:
    if model:
        return model
    if backend in ("bedrock", "openrouter", "vertex", "ollama"):
        return settings().model_for(backend, tier)
    return f"{backend}-{tier}"  # test / custom backends


# ---------------------------------------------------------------- public API

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _clean_json(text: str) -> str:
    m = _FENCE.match(text)
    return m.group(1) if m else text.strip()


@overload
def generate(prompt: str, schema: type[T], **kw) -> T: ...
@overload
def generate(prompt: str, schema: None = None, **kw) -> str: ...


def generate(
    prompt: str,
    schema: type[T] | None = None,
    *,
    system: str | None = None,
    tier: Tier = "flash",
    backend: str | None = None,
    model: str | None = None,
    agent: str = "unknown",
    temperature: float = 0.0,
    max_output_tokens: int | None = 4096,  # bounded cost; OpenRouter reserves credit for this
    retries: int = 1,
) -> T | str:
    """Call an LLM. With `schema`, returns a validated instance of it."""
    backend = backend or settings().llm_backend
    if backend not in BACKENDS:
        raise LLMError(f"unknown backend '{backend}' (choose from {', '.join(BACKENDS)})")
    call = BACKENDS[backend]
    model_name = _model_name(backend, tier, model)
    json_schema = schema.model_json_schema() if schema else None
    current_prompt = prompt
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        with trace.span("llm", agent, backend=backend, model=model_name, tier=tier,
                        attempt=attempt, schema=schema.__name__ if schema else None) as sp:
            raw = call(model_name, current_prompt, system, json_schema, temperature, max_output_tokens)
            sp.update(
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
                cost_usd=raw.cost_usd if raw.cost_usd is not None
                else cost_usd(backend, raw.model, raw.input_tokens, raw.output_tokens),
                finish_reason=raw.finish_reason,
            )
            if schema is None:
                if not raw.text:
                    raise LLMError(f"Empty response (finish_reason={raw.finish_reason})")
                return raw.text
            try:
                return schema.model_validate_json(_clean_json(raw.text))
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_error = e
                sp["schema_error"] = str(e)[:500]
        current_prompt = (
            f"{prompt}\n\n---\nYour previous reply did not match the required JSON schema.\n"
            f"Error: {str(last_error)[:1500]}\nReply again with valid JSON only."
        )

    raise LLMError(f"{agent}: no valid {schema.__name__} after {retries + 1} attempts: {last_error}")
