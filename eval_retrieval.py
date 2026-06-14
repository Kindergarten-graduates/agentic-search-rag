"""Retrieval-only comparison: BM25 vs dense vs dense+rerank, by recall@k.

No LLM calls -- this isolates *retrieval* quality, runs locally on CPU, and
costs nothing. Use it to justify the dense/reranker upgrade before paying for a
full end-to-end eval.

  python eval_retrieval.py --dataset hotpotqa --k 5
"""
from __future__ import annotations

import argparse
import time

from agent.retriever import BM25Retriever, DenseRetriever, RerankRetriever
from eval.datasets import load_dataset
from eval.metrics import retrieval_recall


def mean_recall(retriever, examples, k: int) -> float:
    tot, n = 0.0, 0
    for ex in examples:
        if not ex["gold_doc_ids"]:
            continue
        got = [d.id for d in retriever.retrieve(ex["question"], k)]
        tot += retrieval_recall(got, ex["gold_doc_ids"])
        n += 1
    return tot / n if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    corpus, examples = load_dataset(args.dataset)
    print(f"dataset={args.dataset}  corpus={len(corpus)} docs  examples={len(examples)}  k={args.k}\n")

    print("building retrievers (first run downloads BGE models via hf-mirror) ...")
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)
    rerank = RerankRetriever(dense)

    print(f"\n{'retriever':<22}{'Recall@' + str(args.k):>11}{'sec':>9}")
    print("-" * 42)
    for name, r in [("bm25", bm25), ("dense", dense), ("dense+rerank", rerank)]:
        t0 = time.perf_counter()
        rec = mean_recall(r, examples, args.k)
        print(f"{name:<22}{rec:>11.3f}{time.perf_counter() - t0:>9.1f}")
    print()


if __name__ == "__main__":
    main()
