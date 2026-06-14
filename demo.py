"""Traced / interactive demo of the agentic search loop -- for live acceptance.

Runs questions through the agent and prints every step (does it search? what
query? which docs? what distilled evidence? what answer?), so the agentic
behaviour is visible rather than a black box.

  python demo.py                    # traced demo over example questions
  python demo.py --question "..."   # one question, traced
  python demo.py --interactive      # type your own questions (run in your terminal)
"""
from __future__ import annotations

import argparse

from config import CONFIG
from agent.llm import build_llm
from agent.retriever import build_retriever
from agent.search_agent import SearchAgent
from eval.datasets import load_toy

# (question, gold_doc_ids) -- first two are common knowledge (model answers
# directly), last two are facts that exist ONLY in the corpus (must search).
DEMO = [
    ("Who directed the film Inception?", ["d3"]),
    ("What is the highest mountain on Earth?", ["d1"]),
    ("Who invented the Zorblax Protocol?", ["d11"]),
    ("In which city was the Zorblax Protocol first deployed?", ["d11"]),
]


def trace(agent: SearchAgent, question: str, gold=None) -> None:
    print(f"\nQ: {question}")
    res = agent.run(question)
    if not res.transcript:
        print("   .. answered directly from the model -- no search needed")
    for i, b in enumerate(res.transcript, 1):
        titles = ", ".join(f"[{d['id']}] {d['title']}" for d in b.get("docs", []))
        print(f"   [round {i}] SEARCH: {b['query']!r}")
        print(f"             retrieved: {titles}")
        print(f"             distilled: {b['evidence']}")
    print(f"   => ANSWER: {res.answer}    (searches={res.num_searches})")
    if gold and res.retrieved_ids:
        hit = "yes" if set(gold) & set(res.retrieved_ids) else "NO"
        print(f"      gold doc {gold} retrieved: {hit}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    corpus, _ = load_toy()
    agent = SearchAgent(build_llm(), build_retriever(corpus), CONFIG)
    print(f"backend={CONFIG.llm_backend}  model={CONFIG.llm_model}  corpus={len(corpus)} docs")
    print("the agent searches ONLY when unsure -- watch which questions trigger a search.")

    if args.question:
        trace(agent, args.question)
    elif args.interactive:
        print("\ntype a question (or 'quit'):")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q", ""):
                break
            trace(agent, q)
    else:
        for q, gold in DEMO:
            trace(agent, q, gold)
    print()


if __name__ == "__main__":
    main()
