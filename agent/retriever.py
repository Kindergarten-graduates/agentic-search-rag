"""Retrieval.

``BM25Retriever`` is a dependency-free, pure-stdlib BM25 — fine for the toy
corpus and fast iteration. It is the *foundation* you will improve: the obvious
upgrades (and strong resume material) are a dense retriever (E5 / BGE) and a
cross-encoder reranker. Hooks for those are marked TODO below.

Separating retrieval from generation is deliberate: the eval harness measures
retrieval recall@k independently, so you can tell whether a failure is "didn't
retrieve the right doc" vs "retrieved it but answered wrong".
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

_WORD = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


@dataclass
class Document:
    id: str
    title: str
    text: str
    score: float = 0.0


class _BM25:
    """Okapi BM25 over a list of pre-tokenised documents."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in corpus_tokens]
        df: Counter = Counter()
        for d in corpus_tokens:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for t, freq in df.items()
        }

    def score(self, query_tokens: List[str], idx: int) -> float:
        tf, dl = self.tf[idx], self.doc_len[idx]
        s = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
        return s


class BM25Retriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self._bm25 = _BM25([_tokenize(f"{d.title} {d.text}") for d in documents])

    def retrieve(self, query: str, k: int) -> List[Document]:
        qt = _tokenize(query)
        scored = sorted(
            ((self._bm25.score(qt, i), i) for i in range(len(self.documents))),
            key=lambda x: x[0],
            reverse=True,
        )
        out: List[Document] = []
        for score, i in scored[:k]:
            d = self.documents[i]
            out.append(Document(id=d.id, title=d.title, text=d.text, score=score))
        return out


class DenseRetriever:
    """Dense bi-encoder retrieval (sentence-transformers, E5/BGE). Brute-force
    cosine over the corpus — at HotpotQA-distractor scale (~2k docs) no FAISS is
    needed. For full Wikipedia, swap in a FAISS index built on the GPU box.
    Heavy deps are imported lazily so the BM25 / stub paths stay zero-install.
    """

    def __init__(self, documents: List[Document], batch_size: int = 64):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.documents = documents
        self.model_name = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
        self.model = SentenceTransformer(self.model_name)
        passages = [self._passage(f"{d.title}. {d.text}") for d in documents]
        self.emb = self.model.encode(
            passages, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )

    def _passage(self, text: str) -> str:
        return ("passage: " + text) if "e5" in self.model_name.lower() else text

    def _query(self, q: str) -> str:
        name = self.model_name.lower()
        if "e5" in name:
            return "query: " + q
        if "bge" in name and "-en" in name:
            return "Represent this sentence for searching relevant passages: " + q
        return q

    def retrieve(self, query: str, k: int) -> List[Document]:
        qv = self.model.encode([self._query(query)], normalize_embeddings=True)[0]
        scores = self.emb @ qv
        order = self.np.argsort(-scores)[:k]
        return [
            Document(id=self.documents[i].id, title=self.documents[i].title,
                     text=self.documents[i].text, score=float(scores[i]))
            for i in order
        ]


class RerankRetriever:
    """Wrap a base retriever: pull a wider candidate pool, then re-score with a
    cross-encoder (bge-reranker). Usually the cheapest single accuracy win —
    measure its delta in the ablation table."""

    def __init__(self, base, pool: int = 20):
        from sentence_transformers import CrossEncoder

        self.base = base
        self.pool = int(os.getenv("RERANK_POOL", str(pool)))
        self.model = CrossEncoder(os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"))

    def retrieve(self, query: str, k: int) -> List[Document]:
        cands = self.base.retrieve(query, max(self.pool, k))
        if not cands:
            return []
        scores = self.model.predict([(query, f"{d.title}. {d.text}") for d in cands])
        ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)[:k]
        return [Document(id=d.id, title=d.title, text=d.text, score=float(s)) for d, s in ranked]


def build_retriever(documents: List[Document]):
    from config import CONFIG
    base = DenseRetriever(documents) if CONFIG.retriever == "dense" else BM25Retriever(documents)
    return RerankRetriever(base) if CONFIG.use_reranker else base
