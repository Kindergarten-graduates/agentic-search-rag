"""Central configuration, driven entirely by environment variables.

Defaults are chosen so the whole project runs **fully offline with zero setup**
(LLM_BACKEND=stub, retriever=bm25 over a tiny local corpus). To get real
numbers, point LLM_BACKEND=openai at any OpenAI-compatible endpoint:

  * a vLLM server on the remote 5x4090D box  (recommended for a real reasoning model)
  * a local Ollama / vLLM instance
  * a hosted API such as DeepSeek-R1

See .env.example for concrete settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal, dependency-free .env loader (does not override real env vars)."""
    path = Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # --- LLM backend -------------------------------------------------------
    llm_backend: str = os.getenv("LLM_BACKEND", "stub")          # "stub" | "openai"
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "EMPTY")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen2.5-7B-Instruct")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # --- retrieval ---------------------------------------------------------
    retriever: str = os.getenv("RETRIEVER", "bm25")              # "bm25" | "dense" (TODO)
    top_k: int = int(os.getenv("TOP_K", "5"))
    use_reranker: bool = _flag("USE_RERANKER", "0")              # TODO: cross-encoder rerank

    # --- agentic loop ------------------------------------------------------
    max_search_rounds: int = int(os.getenv("MAX_SEARCH_ROUNDS", "3"))
    use_reason_in_documents: bool = _flag("USE_RID", "1")        # Search-o1's distillation

    def describe(self) -> str:
        return (
            f"backend={self.llm_backend} model={self.llm_model} "
            f"retriever={self.retriever} top_k={self.top_k} "
            f"rounds={self.max_search_rounds} rid={self.use_reason_in_documents} "
            f"reranker={self.use_reranker}"
        )


CONFIG = Config()
