"""Prompt templates for the agentic search loop.

Format follows Search-o1 (arXiv 2501.05366): the model reasons freely and,
when it hits a knowledge gap, emits a search query delimited by special
tokens. Retrieved + distilled evidence is injected back wrapped in result
tokens, and the model continues until it produces a final answer.

The `Task: XXX` label on the first line is a machine-readable hint that the
offline StubLLM keys on to pick its behaviour. Real models simply read it as
part of the instruction, so it is harmless.
"""
from __future__ import annotations

from typing import List

# --- Search-o1 style special tokens -------------------------------------------
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

_AGENT_INSTRUCTIONS = f"""You are a reasoning agent that can call a search engine.
When you are uncertain about a fact, issue ONE search query like this:
{BEGIN_SEARCH_QUERY} your query here {END_SEARCH_QUERY}
Then stop. You will be given results wrapped in {BEGIN_SEARCH_RESULT} ... {END_SEARCH_RESULT}.
Use them to keep reasoning, and search again if you still lack information.
When you are confident, output exactly one line:
Final Answer: <concise answer>"""


def build_agent_prompt(question: str, evidence_blocks: List[dict]) -> str:
    """Prompt for one step of the agentic loop."""
    parts = ["Task: AGENT_STEP", _AGENT_INSTRUCTIONS, "", f"Question: {question}", ""]
    if evidence_blocks:
        parts.append("Evidence gathered so far:")
        for b in evidence_blocks:
            parts.append(f"(searched: {b['query']})")
            parts.append(f"{BEGIN_SEARCH_RESULT} {b['evidence']} {END_SEARCH_RESULT}")
        parts.append("")
    parts.append("Continue your reasoning. Search if needed, otherwise give the Final Answer.")
    return "\n".join(parts)


def build_rid_prompt(question: str, query: str, docs: List[dict]) -> str:
    """Reason-in-Documents: distill raw retrieved docs into clean evidence."""
    parts = [
        "Task: REASON_IN_DOCUMENTS",
        "Read the documents and extract ONLY the information relevant to answering",
        "the question and the current search query. Be concise (1-3 sentences).",
        'If nothing is relevant, reply exactly: "No helpful information found."',
        "",
        f"Question: {question}",
        f"Search query: {query}",
        "",
        "Documents:",
    ]
    for d in docs:
        parts.append(f"- [{d['title']}] {d['text']}")
    parts.append("")
    parts.append("Distilled evidence:")
    return "\n".join(parts)


def build_direct_prompt(question: str, context: str | None = None) -> str:
    """Prompt for the no-retrieval and naive-RAG baselines."""
    parts = ["Task: DIRECT_ANSWER"]
    if context:
        parts += [
            "Answer the question using the context below. If the context is",
            "insufficient, answer from your own knowledge.",
            "",
            f"Context: {context}",
        ]
    else:
        parts += ["Answer the question from your own knowledge."]
    parts += ["", f"Question: {question}", "", "Respond with one line: Final Answer: <concise answer>"]
    return "\n".join(parts)
