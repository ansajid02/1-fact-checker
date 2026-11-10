"""Shared HTTP client: polite User-Agent, retries with backoff, and an on-disk cache.

The cache makes benchmark reruns cheap and reproducible (same pages -> same evidence).
Delete data/cache/http/ to refetch.
"""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from typing import Any

import httpx

from agentcore.config import settings

from ..paths import CACHE_DIR

HTTP_CACHE = CACHE_DIR / "http"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_SECRET_PARAMS = {"api_key"}


class ToolError(RuntimeError):
    """A tool failed. The agent sees the message as its observation."""


class AccessBlocked(RuntimeError):
    """The data source refused us (e.g. Wikimedia robot policy). Fatal: fix config, don't retry."""


UA_HELP = (
    "Wikimedia rejected the request under its robot policy. Set HTTP_USER_AGENT in .env to a "
    "descriptive name with YOUR contact info, e.g. 'agent-lab/0.1 (https://github.com/<you>/agent-lab)' "
    "or '... (<your email>)'. Policy: https://meta.wikimedia.org/wiki/User-Agent_policy"
)


@lru_cache(maxsize=1)
def client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": settings().http_user_agent},
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
    )


def _cache_key(url: str, params: dict[str, Any]) -> str:
    public = {k: v for k, v in sorted(params.items()) if k not in _SECRET_PARAMS}
    return hashlib.sha256(json.dumps([url, public], ensure_ascii=False).encode()).hexdigest()


def get_json(url: str, params: dict[str, Any], *, use_cache: bool = True, retries: int = 3) -> Any:
    path = HTTP_CACHE / f"{_cache_key(url, params)}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    delay = 1.0
    for attempt in range(retries + 1):
        try:
            r = client().get(url, params=params)
        except httpx.TransportError as e:
            if attempt == retries:
                raise ToolError(f"network error: {e}") from e
        else:
            if r.status_code == 200:
                data = r.json()
                if use_cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return data
            if r.status_code == 403 and "robot policy" in r.text.lower():
                raise AccessBlocked(UA_HELP)
            if r.status_code not in _RETRY_STATUS or attempt == retries:
                raise ToolError(f"HTTP {r.status_code} from {url}: {r.text[:200]}")
            retry_after = r.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))
        time.sleep(min(delay, 30))
        delay *= 2
    raise ToolError("unreachable")
