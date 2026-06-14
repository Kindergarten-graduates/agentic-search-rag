"""Comparison / ablation runner — the heart of the project's evidence.

Runs several systems over a dataset and prints one table:

  System          EM     F1     Recall@k   Searches   Lat(s/q)
  no_retrieval    ...
  naive_rag       ...
  agent           ...        <- agentic search (Search-o1 style)
  agent_no_rid    ...        <- ablation: Reason-in-Documents OFF

Usage:
  python run_eval.py --dataset toy
  python run_eval.py --dataset toy --systems no_retrieval,naive_rag,agent,agent_no_rid --verbose
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from config import CONFIG
from agent.llm import build_llm
from agent.retriever import build_retriever
from agent.search_agent import SearchAgent, AgentResult
from baselines.no_retrieval import NoRetrievalBaseline
from baselines.naive_rag import NaiveRAGBaseline
from eval.datasets import load_dataset
from eval.metrics import best_over_golds, exact_match, f1_score, retrieval_recall

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_system(name, llm, retriever, cfg):
    if name == "no_retrieval":
        return NoRetrievalBaseline(llm, retriever, cfg)
    if name == "naive_rag":
        return NaiveRAGBaseline(llm, retriever, cfg)
    if name == "agent":
        return SearchAgent(llm, retriever, cfg)
    if name == "agent_no_rid":
        return SearchAgent(llm, retriever, dataclasses.replace(cfg, use_reason_in_documents=False))
    raise ValueError(f"unknown system '{name}'")


def evaluate(system, examples):
    em = f1 = rec = 0.0
    rec_n = 0
    latency = 0.0
    preds = []
    for idx, ex in enumerate(examples):
        t0 = time.perf_counter()
        try:
            res = system.run(ex["question"])
        except Exception as e:
            print(f"   [warn] {ex['id']} failed: {repr(e)[:100]}")
            res = AgentResult(answer="")
        latency += time.perf_counter() - t0
        if (idx + 1) % 25 == 0:
            print(f"   ...{idx + 1}/{len(examples)}")
        em += best_over_golds(exact_match, res.answer, ex["answers"])
        f1 += best_over_golds(f1_score, res.answer, ex["answers"])
        if ex["gold_doc_ids"] and res.retrieved_ids:
            rec += retrieval_recall(res.retrieved_ids, ex["gold_doc_ids"])
            rec_n += 1
        preds.append({"id": ex["id"], "pred": res.answer,
                      "retrieved": res.retrieved_ids, "searches": res.num_searches})
    n = len(examples)
    return {
        "em": em / n, "f1": f1 / n,
        "recall": (rec / rec_n) if rec_n else None,
        "searches": sum(p["searches"] for p in preds) / n,
        "latency": latency / n,
        "preds": preds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="toy")
    ap.add_argument("--systems", default="no_retrieval,naive_rag,agent,agent_no_rid")
    ap.add_argument("--limit", type=int, default=0, help="cap #examples (0 = all)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    corpus, examples = load_dataset(args.dataset)
    if args.limit:
        examples = examples[: args.limit]
    retriever = build_retriever(corpus)
    llm = build_llm()

    print(f"\nconfig: {CONFIG.describe()}")
    print(f"dataset: {args.dataset}  |  corpus={len(corpus)} docs  |  examples={len(examples)}\n")

    rows = {}
    for name in [s.strip() for s in args.systems.split(",") if s.strip()]:
        system = build_system(name, llm, retriever, CONFIG)
        print(f"[run] {name} over {len(examples)} examples ...")
        rows[name] = evaluate(system, examples)
        if args.verbose:
            print(f"--- {name} ---")
            for p in rows[name]["preds"]:
                print(f"  {p['id']}: {p['pred']!r}  (searches={p['searches']})")
            print()

    k = CONFIG.top_k
    print(f"{'System':<16}{'EM':>7}{'F1':>7}{f'Recall@{k}':>11}{'Searches':>10}{'Lat(s/q)':>10}")
    print("-" * 61)
    for name, r in rows.items():
        rec = f"{r['recall']:.2f}" if r["recall"] is not None else "—"
        print(f"{name:<16}{r['em']:>7.2f}{r['f1']:>7.2f}{rec:>11}{r['searches']:>10.1f}{r['latency']:>10.3f}")
    print()

    if CONFIG.llm_backend == "stub":
        print("note: StubLLM is a plumbing mock (crude extractive QA), not a real model.\n"
              "      The table validates the harness end-to-end: no_retrieval collapses while\n"
              "      retrieval recovers the answer, and recall@k is reported SEPARATELY so you\n"
              "      can attribute failures to retrieval vs generation. On this toy single-hop\n"
              "      set naive_rag and agent tie — the agent's edge shows up on real multi-hop\n"
              "      QA with a real model. Set LLM_BACKEND=openai (.env.example) for that.\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.dataset}_results.json"
    payload = {"config": CONFIG.describe(), "dataset": args.dataset,
               "results": {k2: {kk: vv for kk, vv in v.items() if kk != "preds"}
                           for k2, v in rows.items()},
               "details": {k2: v["preds"] for k2, v in rows.items()}}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out}\n")


if __name__ == "__main__":
    main()
