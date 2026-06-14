"""Metrics.

Answer quality: SQuAD-style Exact Match and token-F1.
Retrieval quality: recall@k, measured INDEPENDENTLY of the answer so you can
attribute failures to retrieval vs generation.
"""
from __future__ import annotations

import re
import string
from typing import List


def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lowercase, strip punctuation / articles / extra ws."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = 0
    gold_pool = list(gold_toks)
    for t in pred_toks:
        if t in gold_pool:
            common += 1
            gold_pool.remove(t)
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def best_over_golds(metric, pred: str, golds: List[str]) -> float:
    """Take the max metric over a list of acceptable gold answers."""
    return max((metric(pred, g) for g in golds), default=0.0)


def retrieval_recall(retrieved_ids: List[str], gold_doc_ids: List[str]) -> float:
    """Fraction of gold documents present in the retrieved set."""
    if not gold_doc_ids:
        return float("nan")
    hit = sum(1 for g in gold_doc_ids if g in set(retrieved_ids))
    return hit / len(gold_doc_ids)
