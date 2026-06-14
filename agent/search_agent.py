"""The Search-o1-style agentic loop.

reason -> (emit search query) -> retrieve -> Reason-in-Documents -> inject ->
reason -> ... -> Final Answer.

This is the "agent" the project is built around. Everything else (baselines,
metrics) exists to measure it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from config import Config
from agent.llm import LLM
from agent.prompts import (
    BEGIN_SEARCH_QUERY,
    END_SEARCH_QUERY,
    build_agent_prompt,
    build_direct_prompt,
)
from agent.reason_in_documents import concat_documents, reason_in_documents
from agent.retriever import Document


@dataclass
class AgentResult:
    answer: str
    retrieved_ids: List[str] = field(default_factory=list)
    num_searches: int = 0
    transcript: List[dict] = field(default_factory=list)


def extract_search_query(text: str) -> Optional[str]:
    start = text.find(BEGIN_SEARCH_QUERY)
    if start == -1:
        return None
    rest = text[start + len(BEGIN_SEARCH_QUERY):]
    end = rest.find(END_SEARCH_QUERY)
    query = (rest if end == -1 else rest[:end]).strip()
    return query or None


def extract_final_answer(text: str) -> str:
    m = re.search(r"Final Answer:\s*(.+)", text)
    if m:
        return m.group(1).splitlines()[0].strip()
    # Fallback: strip any leftover search markers and take the last non-empty line.
    cleaned = text.replace(BEGIN_SEARCH_QUERY, "").replace(END_SEARCH_QUERY, "").strip()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


class SearchAgent:
    def __init__(self, llm: LLM, retriever, cfg: Config):
        self.llm = llm
        self.retriever = retriever
        self.cfg = cfg

    def run(self, question: str) -> AgentResult:
        evidence_blocks: List[dict] = []
        retrieved_ids: List[str] = []

        for round_idx in range(self.cfg.max_search_rounds + 1):
            prompt = build_agent_prompt(question, evidence_blocks)
            out = self.llm.generate(prompt, stop=[END_SEARCH_QUERY])
            query = extract_search_query(out)

            if query is not None and round_idx < self.cfg.max_search_rounds:
                docs: List[Document] = self.retriever.retrieve(query, self.cfg.top_k)
                for d in docs:
                    if d.id not in retrieved_ids:
                        retrieved_ids.append(d.id)
                if self.cfg.use_reason_in_documents:
                    evidence = reason_in_documents(question, query, docs, self.llm)
                else:
                    evidence = concat_documents(docs)
                evidence_blocks.append({
                    "query": query,
                    "evidence": evidence,
                    "docs": [{"id": d.id, "title": d.title} for d in docs],
                })
            else:
                return AgentResult(
                    answer=extract_final_answer(out),
                    retrieved_ids=retrieved_ids,
                    num_searches=len(evidence_blocks),
                    transcript=evidence_blocks,
                )

        # Exhausted the round budget: force an answer from gathered evidence.
        ctx = "\n".join(b["evidence"] for b in evidence_blocks)
        out = self.llm.generate(build_direct_prompt(question, ctx))
        return AgentResult(
            answer=extract_final_answer(out),
            retrieved_ids=retrieved_ids,
            num_searches=len(evidence_blocks),
            transcript=evidence_blocks,
        )
