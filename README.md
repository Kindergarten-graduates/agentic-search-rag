# Agentic Search-Augmented RAG

A reasoning agent that **searches when it's unsure**, distills what it finds, and
answers — built in the style of [Search-o1](https://arxiv.org/abs/2501.05366) and
measured against baselines with a real eval harness.

The point of this project is **not** "I built a RAG bot." It's: *here is an
agentic search system, here is the harness that proves it works, and here is an
honest breakdown of where it helps and where it breaks* — including the parts that
**didn't** work, and why.

## The loop

```
question
  ↓
reason ──► uncertain? ──► <|begin_search_query|> q <|end_search_query|>
  ▲                              ↓
  │                          retrieve top-k
  │                              ↓
  │                    Reason-in-Documents  (distill noisy passages → clean evidence)
  │                              ↓
  └───────────── inject <|begin_search_result|> … <|end_search_result|>
                                 ↓
                          Final Answer: …
```

`agent/` is the system; `baselines/` and `eval/` exist to measure it.

## Quickstart

**Offline, zero install** — StubLLM + stdlib BM25 over a toy corpus; validates the
whole pipeline end to end:

```bash
python run_eval.py --dataset toy
```

**Real LLM** — DeepSeek / vLLM / Ollama (any OpenAI-compatible endpoint):

```bash
cp .env.example .env          # set OPENAI_BASE_URL / OPENAI_API_KEY / LLM_MODEL
pip install openai
python demo.py                # watch the agent search on-demand (traced)
python run_eval.py --dataset toy --verbose
```

**Real benchmark — HotpotQA (multi-hop):**

```bash
pip install datasets sentence-transformers torch   # CPU torch is fine
# BM25 reader, 200 questions, <=2 search rounds:
HOTPOT_SAMPLE=200 MAX_SEARCH_ROUNDS=2 python run_eval.py --dataset hotpotqa --limit 200
# dense retriever instead of BM25:
RETRIEVER=dense HOTPOT_SAMPLE=200 MAX_SEARCH_ROUNDS=2 python run_eval.py --dataset hotpotqa --limit 200
# retrieval-only recall comparison (no LLM, no API cost):
python eval_retrieval.py --dataset hotpotqa --k 5
```

> **CN networks:** the HotpotQA shard downloads from `hf-mirror.com` (override with
> `HF_MIRROR`); models download from the real `huggingface.co`. Do **not** set
> `HF_ENDPOINT` — the mirror breaks `huggingface_hub`'s metadata calls.

## Results

**Setup.** HotpotQA (distractor), 200 validation questions, reader =
`deepseek-chat`, top-k = 5, ≤ 2 search rounds. Corpus = the union of the
questions' candidate paragraphs. `agent` = Search-o1-style multi-step search;
`agent_no_rid` = the same with Reason-in-Documents turned off.

**Answer accuracy (BM25 vs dense retriever):**

| System              | EM (BM25) | EM (dense) | F1 (dense) | Recall@5 (BM25 → dense) |
|---------------------|:---------:|:----------:|:----------:|:-----------------------:|
| no_retrieval        |   0.20    |     —      |    0.35    |            —            |
| naive_rag           |   0.33    |    0.34    |    0.55    |       0.72 → 0.84       |
| agent               |   0.47    |    0.48    |    0.60    |       0.74 → 0.83       |
| **agent (RID off)** |   0.49    |  **0.51**  |  **0.64**  |       0.80 → 0.81       |

**Retrieval quality in isolation — recall@5 (100 questions, no LLM):**

| Retriever          | Recall@5 |
|--------------------|:--------:|
| BM25               |  0.745   |
| dense (BGE-small)  |  0.840   |
| dense + reranker   |  0.830   |

Two separate tables on purpose: **retrieval quality (recall@k) is measured apart
from answer quality (EM/F1)**, so a failure can be pinned on *retrieval* ("never
fetched the right doc") vs *generation* ("had the doc, still answered wrong"). That
separation is what makes Finding 4 below provable instead of a guess.

### Findings

1. **Agentic multi-step search ≫ naive RAG.** EM 0.33 → 0.49–0.51 (≈ +50%), and
   ≈ 2.5× the no-retrieval reader (0.20). Multi-hop questions need more than one
   retrieval, and the agent decides *when* to search (≈ 0.7–0.8 searches/question
   vs naive RAG's fixed 1).
2. **Reason-in-Documents *hurt* here** (agent 0.47 < agent_no_rid 0.49; recall
   0.74 < 0.80). Distractor passages are already short and clean, so the
   distillation step removes detail useful for the 2nd-hop query instead of
   removing noise. Expect it to pay off only on long/noisy (full-Wikipedia)
   passages.
3. **A cross-encoder reranker didn't help** on multi-hop (recall 0.840 → 0.830,
   13× slower). A generic single-hop relevance reranker mis-ranks the "bridge"
   paragraph a multi-hop answer depends on.
4. **Dense lifted recall but not EM** (recall 0.745 → 0.840; EM ≈ flat). Because
   the two axes are measured apart, this *localizes the bottleneck*: it is no
   longer retrieval — the reader has the right documents and still can't produce
   the exact answer. The EM ≪ F1 gap (0.51 vs 0.64) confirms many answers are
   right-but-not-exact-match.

**Takeaway.** The remaining bottleneck is multi-hop *reasoning*, not retrieval — so
the next levers are a stronger reasoning reader and answer normalization, not more
retrieval tuning.

## Roadmap

- [x] Agent loop + Reason-in-Documents + eval harness (EM/F1 + separate recall@k)
- [x] Baselines: no-retrieval, naive RAG, agent, agent-no-RID
- [x] Real multi-hop benchmark: HotpotQA (distractor)
- [x] Dense retriever (BGE) + cross-encoder reranker, with ablations
- [ ] Stronger reader (deepseek-reasoner / R1) to attack the reasoning bottleneck
- [ ] Answer normalization to close the EM ≪ F1 gap
- [ ] Answer-level citation grounding + hallucination check
- [ ] Full-Wikipedia retrieval (where RID should finally help); Bamboogle

## References

- Search-o1: Agentic Search-Enhanced Large Reasoning Models — arXiv
  [2501.05366](https://arxiv.org/abs/2501.05366) ·
  [code](https://github.com/RUC-NLPIR/Search-o1)
- Search-R1: Training LLMs to Reason and Leverage Search Engines with RL — arXiv
  [2503.09516](https://arxiv.org/abs/2503.09516) ·
  [code](https://github.com/PeterGriffinJin/Search-R1) (RL; we borrow its eval
  setup, not the training)

> **StubLLM is a mock**, not a model — it does crude extractive QA so the pipeline
> runs with zero setup. Swap in a real endpoint for real numbers.
