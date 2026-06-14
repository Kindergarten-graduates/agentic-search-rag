"""Dataset loaders.

`toy` is a tiny offline fixture so everything runs with zero downloads. The real
benchmarks (multi-hop QA, comparable to Search-o1 / Search-R1) are stubbed with
TODOs — wire them up via HuggingFace `datasets` once the pipeline is solid.
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import List, Tuple

from agent.retriever import Document

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _download_cached(url: str, dest: Path) -> Path:
    """Download `url` to `dest` once; reuse the local copy afterwards."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[datasets] downloading {url} ...")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    print(f"[datasets] cached -> {dest}")
    return dest


def load_toy() -> Tuple[List[Document], List[dict]]:
    """Returns (corpus, examples). Each example: id, question, answers, gold_doc_ids."""
    root = DATA_DIR / "toy"
    corpus = [Document(id=d["id"], title=d["title"], text=d["text"])
              for d in _read_jsonl(root / "corpus.jsonl")]
    examples = []
    for ex in _read_jsonl(root / "questions.jsonl"):
        answers = ex.get("answers") or ([ex["answer"]] if "answer" in ex else [])
        examples.append({
            "id": ex["id"],
            "question": ex["question"],
            "answers": answers,
            "gold_doc_ids": ex.get("gold_doc_ids", []),
        })
    return corpus, examples


def load_hotpotqa() -> Tuple[List[Document], List[dict]]:
    """HotpotQA distractor: each question ships 10 candidate paragraphs (2 gold,
    8 distractors). We build ONE global corpus from the union of all sampled
    questions' paragraphs (dedup by title), so retrieval recall@k is meaningful
    (gold must outrank other questions' distractors too).

    Tunables (env): HOTPOT_SAMPLE = #questions to load (default 100).
    Uses the hf-mirror.com endpoint by default for reliable access in CN; set
    HF_ENDPOINT to override (e.g. https://huggingface.co if you have a proxy).
    """
    import pyarrow.parquet as pq
    # Use a DEDICATED mirror var for the dataset's big LFS file (the mirror serves
    # LFS reliably). Do NOT touch HF_ENDPOINT — model downloads need the real hub.
    endpoint = os.getenv("HF_MIRROR", "https://hf-mirror.com").rstrip("/")
    sample = int(os.getenv("HOTPOT_SAMPLE", "100"))

    # datasets 5.x can't resolve the legacy script dataset, and hf_hub_download's
    # HEAD probe is flaky against the CN mirror, so fetch the parquet shard
    # directly (cached locally after the first download).
    url = (f"{endpoint}/datasets/hotpotqa/hotpot_qa/resolve/main/"
           "distractor/validation-00000-of-00001.parquet")
    path = _download_cached(url, DATA_DIR / "cache" / "hotpotqa_distractor_validation.parquet")
    rows = pq.read_table(path).slice(0, sample).to_pylist()

    docs_by_title: dict = {}
    examples = []
    for i, ex in enumerate(rows):
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title not in docs_by_title:
                docs_by_title[title] = " ".join(s.strip() for s in sents).strip()
        gold = list(dict.fromkeys(ex["supporting_facts"]["title"]))
        examples.append({
            "id": ex.get("id", f"hp{i}"),
            "question": ex["question"],
            "answers": [ex["answer"]],
            "gold_doc_ids": gold,
        })
    corpus = [Document(id=t, title=t, text=txt) for t, txt in docs_by_title.items()]
    return corpus, examples


def load_bamboogle() -> Tuple[List[Document], List[dict]]:
    # TODO: Bamboogle (~125 2-hop questions). Great as a FAST dev-loop eval.
    #   Needs a web-search retriever or a Wikipedia index for the corpus.
    raise NotImplementedError("Bamboogle loader not wired yet — see TODO in eval/datasets.py")


LOADERS = {
    "toy": load_toy,
    "hotpotqa": load_hotpotqa,
    "bamboogle": load_bamboogle,
}


def load_dataset(name: str) -> Tuple[List[Document], List[dict]]:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset '{name}'. options: {list(LOADERS)}")
    return LOADERS[name]()
