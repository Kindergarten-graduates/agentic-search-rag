"""Reason-in-Documents — Search-o1's key idea.

Raw retrieved passages are long and noisy; dumping them straight into the
reasoning chain pollutes context and derails the model. Instead we run a focused
LLM call that distills each retrieval into 1-3 sentences of clean, on-topic
evidence *before* injecting it. This is the component most worth ablating: its
delta on the comparison table is a concrete talking point (context engineering /
avoiding context pollution).
"""
from __future__ import annotations

from typing import List

from agent.llm import LLM
from agent.prompts import build_rid_prompt
from agent.retriever import Document


def reason_in_documents(question: str, query: str, docs: List[Document], llm: LLM) -> str:
    if not docs:
        return "No helpful information found."
    doc_dicts = [{"title": d.title, "text": d.text} for d in docs]
    prompt = build_rid_prompt(question, query, doc_dicts)
    return llm.generate(prompt).strip()


def concat_documents(docs: List[Document]) -> str:
    """Fallback evidence when Reason-in-Documents is disabled (for ablation)."""
    return "\n".join(f"[{d.title}] {d.text}" for d in docs)
