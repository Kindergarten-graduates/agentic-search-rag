"""Baseline 2: naive RAG — retrieve top-k once on the original question, stuff
the passages into the prompt, answer in a single shot.

This is the "everybody's RAG" baseline. Beating it on multi-hop questions is
exactly what the agentic loop is supposed to demonstrate.
"""
from __future__ import annotations

from typing import List

from config import Config
from agent.llm import LLM
from agent.prompts import build_direct_prompt
from agent.reason_in_documents import concat_documents
from agent.retriever import Document
from agent.search_agent import AgentResult, extract_final_answer


class NaiveRAGBaseline:
    name = "naive_rag"

    def __init__(self, llm: LLM, retriever, cfg: Config):
        self.llm = llm
        self.retriever = retriever
        self.cfg = cfg

    def run(self, question: str) -> AgentResult:
        docs: List[Document] = self.retriever.retrieve(question, self.cfg.top_k)
        context = concat_documents(docs)
        out = self.llm.generate(build_direct_prompt(question, context))
        return AgentResult(
            answer=extract_final_answer(out),
            retrieved_ids=[d.id for d in docs],
            num_searches=1,
        )
