"""Settings shared by every project. Values come from the repo-root .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # this project folder (self-contained)

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    """Unset OR empty (`KEY=` in .env) both mean: use the default."""
    return os.environ.get(name, "").strip() or default


@dataclass(frozen=True)
class Settings:
    llm_backend: str  # default backend: "bedrock" | "openrouter" | "vertex" | "ollama"
    bedrock_api_key: str
    bedrock_region: str
    bedrock_model_flash: str
    bedrock_model_pro: str
    bedrock_effort: str
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_model_flash: str
    openrouter_model_pro: str
    gcp_project: str
    gcp_location: str
    vertex_model_flash: str
    vertex_model_pro: str
    ollama_url: str
    ollama_model: str
    embed_backend: str
    embed_model: str
    vertex_embed_model: str
    local_embed_model: str
    http_user_agent: str
    openalex_api_key: str
    trace_dir: Path
    local_models_dir: Path

    def model_for(self, backend: str, tier: str) -> str:
        if backend == "ollama":
            if not self.ollama_model:
                raise RuntimeError("OLLAMA_MODEL is not set in .env")
            return self.ollama_model
        model = getattr(self, f"{backend}_model_{tier}", "")
        if not model:
            raise RuntimeError(
                f"No model set for backend '{backend}', tier '{tier}'. Set "
                f"{backend.upper()}_MODEL_{tier.upper()} in .env (`make models` lists options)."
            )
        return model


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings(
        llm_backend=_env("LLM_BACKEND", "bedrock"),
        # Claude in Amazon Bedrock (Messages API). Bedrock API key = bearer token.
        bedrock_api_key=_env("AWS_BEARER_TOKEN_BEDROCK"),
        bedrock_region=_env("AWS_REGION", "us-east-1"),
        bedrock_model_flash=_env("BEDROCK_MODEL_FLASH", "anthropic.claude-haiku-5-5"),
        bedrock_model_pro=_env("BEDROCK_MODEL_PRO", "anthropic.claude-opus-5-5"),
        bedrock_effort=_env("BEDROCK_EFFORT", "low"),
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        openrouter_base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_model_flash=_env("OPENROUTER_MODEL_FLASH"),
        openrouter_model_pro=_env("OPENROUTER_MODEL_PRO"),
        gcp_project=_env("GOOGLE_CLOUD_PROJECT"),
        gcp_location=_env("GOOGLE_CLOUD_LOCATION", "us-central1"),
        vertex_model_flash=_env("VERTEX_MODEL_FLASH"),
        vertex_model_pro=_env("VERTEX_MODEL_PRO"),
        ollama_url=_env("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL"),
        embed_backend=_env("EMBED_BACKEND", "openrouter"),
        embed_model=_env("EMBED_MODEL", "baai/bge-m3"),
        vertex_embed_model=_env("VERTEX_EMBED_MODEL"),
        local_embed_model=_env("LOCAL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        # Wikimedia requires contact info in the User-Agent: set HTTP_USER_AGENT in .env
        http_user_agent=_env("HTTP_USER_AGENT", "agent-lab/0.1 (personal research project)"),
        openalex_api_key=_env("OPENALEX_API_KEY"),
        trace_dir=Path(_env("TRACE_DIR") or PROJECT_ROOT / "traces"),
        # the ONLY thing shared between projects: local model files, downloaded once
        local_models_dir=Path(_env("LOCAL_MODELS_DIR") or Path.home() / ".cache" / "agent-lab-models").expanduser(),
    )
