"""Token prices, read from prices.toml at the repo root. Unknown model -> cost None."""

from __future__ import annotations

import tomllib
from functools import lru_cache

from .config import PROJECT_ROOT

PRICES_FILE = PROJECT_ROOT / "prices.toml"


@lru_cache(maxsize=1)
def _prices() -> dict[str, dict[str, float]]:
    if not PRICES_FILE.exists():
        return {}
    return tomllib.loads(PRICES_FILE.read_text(encoding="utf-8")).get("models", {})


def cost_usd(backend: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    if backend == "ollama":
        return 0.0  # local model: no per-token cost
    p = _prices().get(model)
    if not p:
        return None
    return round(
        input_tokens / 1e6 * p.get("input", 0.0) + output_tokens / 1e6 * p.get("output", 0.0), 8
    )
