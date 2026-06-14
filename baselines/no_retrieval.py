"""Baseline 1: no retrieval — the LLM answers from parametric knowledge only.

Lower bound. The gap between this and the retrieval systems is the value of
search; the gap between the two retrieval systems is the value of *agentic*
search.
"""
from __future__ import annotations

from config import Config
from agent.llm import LLM
from agent.prompts import build_direct_prompt
from agent.search_agent import AgentResult, extract_final_answer


class NoRetrievalBaseline:
    name = "no_retrieval"

    def __init__(self, llm: LLM, retriever, cfg: Config):
        self.llm = llm

    def run(self, question: str) -> AgentResult:
        out = self.llm.generate(build_direct_prompt(question, context=None))
        return AgentResult(answer=extract_final_answer(out), retrieved_ids=[], num_searches=0)
