"""LLM backends.

Two implementations behind a tiny common interface:

* ``StubLLM``   — deterministic, dependency-free mock. It is NOT a real model;
  it exists so the full pipeline (agent loop -> retrieval -> distillation ->
  metrics -> table) runs offline with zero setup. It does crude extractive QA
  so the comparison table shows a sensible pattern (no-retrieval fails,
  retrieval helps). Swap it out for real reasoning.
* ``OpenAICompatLLM`` — talks to any OpenAI-compatible endpoint (vLLM, Ollama,
  DeepSeek, ...). Imported lazily so the stub path needs no `openai` package.

Both expose: ``generate(prompt: str, stop: list[str] | None = None) -> str``.
"""
from __future__ import annotations

import re
from typing import List, Optional

from config import CONFIG

_WORD = re.compile(r"[a-zA-Z0-9]+")
_STOP_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "is", "was", "were", "are", "what",
    "who", "where", "when", "which", "how", "did", "does", "do", "for", "and",
    "by", "with", "from", "that", "this", "it", "its", "be", "as", "at", "city",
}
# Leading capitalised tokens we never want to return as an "entity".
_STOP_CAPS = {"The", "A", "An", "He", "She", "It", "This", "That", "In", "On",
              "Its", "His", "Her", "They", "There", "When", "Where", "Final"}


def _tokens(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _capitalized_phrases(text: str) -> List[str]:
    """Heuristic named-entity grab: runs of Capitalised words, minus leading stopcaps."""
    phrases = re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*", text)
    out: List[str] = []
    for p in phrases:
        words = p.split()
        while words and words[0] in _STOP_CAPS:
            words = words[1:]
        if words:
            out.append(" ".join(words))
    return out


class LLM:
    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:  # pragma: no cover
        raise NotImplementedError


class StubLLM(LLM):
    """Deterministic offline mock — see module docstring."""

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if "Task: REASON_IN_DOCUMENTS" in prompt:
            return self._distill(prompt)
        if "Task: AGENT_STEP" in prompt:
            return self._agent_step(prompt)
        return self._direct_answer(prompt)

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _field(prompt: str, label: str) -> str:
        m = re.search(rf"{label}:\s*(.+)", prompt)
        return m.group(1).strip() if m else ""

    def _content_query(self, question: str) -> str:
        toks = [t for t in _tokens(question) if t not in _STOP_WORDS]
        return " ".join(toks) or question

    def _extract_answer(self, question: str, context: str) -> str:
        if not context.strip():
            return "I don't know"
        qtok = set(_tokens(question))
        sents = _split_sentences(context)
        best = max(sents, key=lambda s: len(set(_tokens(s)) & qtok), default="")
        for ent in _capitalized_phrases(best):
            if not (set(_tokens(ent)) & qtok):     # an entity not already named in the question
                return ent
        return " ".join(best.split()[:6]) if best else "I don't know"

    # -- behaviours -----------------------------------------------------------
    def _agent_step(self, prompt: str) -> str:
        question = self._field(prompt, "Question")
        from agent.prompts import BEGIN_SEARCH_RESULT, END_SEARCH_RESULT, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY
        # Real evidence only ever appears under this header (see build_agent_prompt),
        # which keeps us from matching the example tokens in the instructions.
        marker = "Evidence gathered so far:"
        idx = prompt.find(marker)
        if idx == -1:
            q = self._content_query(question)
            return f"I need more information about this.\n{BEGIN_SEARCH_QUERY} {q} {END_SEARCH_QUERY}"
        results = re.findall(
            re.escape(BEGIN_SEARCH_RESULT) + r"(.*?)" + re.escape(END_SEARCH_RESULT),
            prompt[idx:], flags=re.DOTALL,
        )
        context = " ".join(results)
        return f"Based on the evidence, I can answer.\nFinal Answer: {self._extract_answer(question, context)}"

    def _direct_answer(self, prompt: str) -> str:
        question = self._field(prompt, "Question")
        context = self._field(prompt, "Context")
        return f"Final Answer: {self._extract_answer(question, context)}"

    def _distill(self, prompt: str) -> str:
        question = self._field(prompt, "Question")
        doc_lines = re.findall(r"^- \[.*?\]\s*(.+)$", prompt, flags=re.MULTILINE)
        sents = [s for line in doc_lines for s in _split_sentences(line)]
        if not sents:
            return "No helpful information found."
        qtok = set(_tokens(question))
        ranked = sorted(sents, key=lambda s: len(set(_tokens(s)) & qtok), reverse=True)
        top = [s for s in ranked[:2] if set(_tokens(s)) & qtok]
        return " ".join(top) if top else "No helpful information found."


class OpenAICompatLLM(LLM):
    """Any OpenAI-compatible chat endpoint (vLLM / Ollama / DeepSeek / ...)."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai  (needed for LLM_BACKEND=openai)") from e
        self._client = OpenAI(base_url=CONFIG.openai_base_url, api_key=CONFIG.openai_api_key)

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        import time
        last_err = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=CONFIG.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=CONFIG.temperature,
                    max_tokens=CONFIG.max_tokens,
                    stop=stop,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # transient API/network error -> back off and retry
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise last_err


def build_llm() -> LLM:
    if CONFIG.llm_backend == "openai":
        return OpenAICompatLLM()
    return StubLLM()
