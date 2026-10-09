"""Central configuration, loaded once from the environment / .env file.

Everything the sandbox needs to talk to GCP (and to a local Ollama server,
if that runs the models) lives here so the rest of the codebase never reads
os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root if present. Real env vars always win over .env.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

LLM_BACKENDS = ("vertex", "gemini", "ollama")


@dataclass(frozen=True)
class Settings:
    # --- GCP / Vertex ---
    project_id: str
    vertex_location: str
    # --- Earth Engine service account ---
    ee_service_account: str | None
    credentials_path: str | None
    # --- Models (bare Gemini ids, or Ollama tags when llm_backend='ollama') ---
    orchestrator_model: str
    worker_model: str
    # --- LLM backend: 'vertex' (default), 'gemini' (AI Studio) or 'ollama' ---
    llm_backend: str
    vertex_api_key: str | None   # API-key auth for Vertex (no service account needed)
    gemini_api_key: str | None   # API key for the AI Studio / Gemini API path
    # --- Ollama (used only when llm_backend='ollama') ---
    ollama_host: str
    ollama_num_ctx: int          # context window; Ollama's own default is often 4096
    ollama_think: bool | str | None  # None = the model's default; 'low'/'medium'/'high' for gpt-oss

    @property
    def credentials_abspath(self) -> str | None:
        if not self.credentials_path:
            return None
        p = Path(self.credentials_path)
        return str(p if p.is_absolute() else (_REPO_ROOT / p))


def _get(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, default)
    return val.strip() if isinstance(val, str) else val


def _ollama_think(value: str | None) -> bool | str | None:
    """OLLAMA_THINK -> the API's `think` value. Blank means the model's default."""
    if not value:
        return None
    value = value.lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    if value in ("low", "medium", "high"):  # gpt-oss takes an effort level
        return value
    raise RuntimeError(
        f"OLLAMA_THINK={value!r} is not understood. Use true, false, low, medium, "
        "high, or leave it blank for the model's default."
    )


def load_settings() -> Settings:
    project_id = _get("GCP_PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "GCP_PROJECT_ID is not set. Copy .env.example to .env and fill it in."
        )

    llm_backend = (_get("LLM_BACKEND", "vertex") or "vertex").lower()
    if llm_backend not in LLM_BACKENDS:
        raise RuntimeError(
            f"LLM_BACKEND={llm_backend!r} is not supported. Use one of: "
            + ", ".join(LLM_BACKENDS) + "."
        )
    if llm_backend == "ollama":
        # No sensible default: it has to be a model this machine has pulled.
        # One model serving both roles keeps a single copy in memory.
        orchestrator_model = _get("ORCHESTRATOR_MODEL")
        if not orchestrator_model:
            raise RuntimeError(
                "LLM_BACKEND=ollama needs ORCHESTRATOR_MODEL set to an Ollama model "
                "tag in .env, e.g. qwen2.5-coder:14b (`ollama list` shows yours)."
            )
        worker_model = _get("WORKER_MODEL") or orchestrator_model
    else:
        orchestrator_model = _get("ORCHESTRATOR_MODEL") or "gemini-2.5-pro"
        worker_model = _get("WORKER_MODEL") or "gemini-2.5-flash"

    num_ctx = _get("OLLAMA_NUM_CTX") or "32768"
    try:
        ollama_num_ctx = int(num_ctx)
    except ValueError:
        raise RuntimeError(f"OLLAMA_NUM_CTX={num_ctx!r} must be a whole number of tokens.") from None

    return Settings(
        project_id=project_id,
        vertex_location=_get("VERTEX_LOCATION", "us-central1"),
        ee_service_account=_get("EE_SERVICE_ACCOUNT"),
        credentials_path=_get("GOOGLE_APPLICATION_CREDENTIALS"),
        orchestrator_model=orchestrator_model,
        worker_model=worker_model,
        llm_backend=llm_backend,
        vertex_api_key=_get("VERTEX_API_KEY"),
        gemini_api_key=_get("GEMINI_API_KEY"),
        ollama_host=_get("OLLAMA_HOST") or "http://localhost:11434",
        ollama_num_ctx=ollama_num_ctx,
        ollama_think=_ollama_think(_get("OLLAMA_THINK")),
    )


# Import-time singleton is intentionally avoided so that importing the package
# never fails just because .env isn't filled in yet. Call load_settings().
