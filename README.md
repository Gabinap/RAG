*This project has been created as part of the 42 curriculum by [gagulhon](https://github.com/Gabinap).*

# 🧗 RAG against the machine

> *Will you answer my questions?*

A Retrieval-Augmented Generation system that ingests the **vLLM** codebase,
retrieves the most relevant snippets for a question, and lets a small LLM
(`Qwen/Qwen3-0.6B`) send the answer — grounded in real sources, no hallucinated
beta. 🪢

---

## 📖 Description

The pipeline climbs in four pitches:

1. **Index** 🪓 — read every `.py` / `.md` / `.txt` of the repo, chunk it, and
   build a searchable index (BM25 and/or embeddings).
2. **Search** 🧭 — for a question, retrieve the top-k chunks with their exact
   character positions in the source files.
3. **Answer** 🤖 — feed the retrieved context to the LLM and generate a concise,
   source-grounded answer.
4. **Evaluate** 📊 — score retrieval quality with `recall@k` against ground truth.

Everything is driven by a single CLI (Python Fire) and a `Makefile`.

---

## ⚙️ Instructions

**Requirements:** Python 3.10, [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
make install

# 2. Place the vLLM repo at the expected approach (anchor point)
#    data/raw/vllm-0.10.1   ← provided as attachment
mkdir -p data/raw && ln -s <path>/vllm-0.10.1 data/raw/vllm-0.10.1

# 3. Send the full route (index → search → answer → evaluate)
make run
```

Useful knobs (override any Make variable on the fly):

```bash
make run RETRIEVER=bm25          # lexical only (fast on CPU)
make run RETRIEVER=hybrid        # BM25 + embeddings (best, needs a GPU to be quick)
make search_docs K=10 EXPAND=True
make index CHUNK_SIZE=1500
make debug ARGS="search 'how to load a LoRA adapter?'"
make fclean                      # wipe generated indexes & outputs
```

---

## 🗺️ System architecture

```
src/
├── main.py            CLI (Python Fire) — index/search/answer/evaluate
├── models.py          Pydantic models (sources, results, configs, cache meta)
├── constants.py       Shared colour palette & display helpers
├── indexing/
│   ├── __init__.py    Orchestrator: build_index, retrieve, search_dataset
│   ├── chunking.py    Language-aware chunking (.py / .md / .txt)
│   ├── bm25.py        BM25 build / load / search
│   ├── embedding.py   Embedding build / load / search + RRF fusion
│   ├── query_expansion.py   WordNet synonym expansion
│   └── store.py       Chunk persistence, artifact detection, index id
├── generation/
│   ├── inference.py   answer_query / answer_dataset (+ batching, caching)
│   ├── model.py       Lazy LLM singleton
│   └── prompt.py      Context assembly & chat template
└── evaluation/
    └── metrics.py     recall@k against ground truth
```

The four CLI verbs each spawn a short process; heavy ML libraries are imported
**lazily** so commands that don't need the LLM start in under a second.

---

## 🪓 Chunking strategy

Different rock, different gear — chunking adapts to the file type via
`langchain`'s `RecursiveCharacterTextSplitter.from_language`:

- **Python** → splits on classes, functions, then blocks/lines.
- **Markdown / text** → splits on headers, paragraphs, sentences.

Max chunk size is **2000 characters** (configurable, `--max_chunk_size`) with a
200-char overlap so no hold is lost between two chunks. Each chunk keeps its
exact `first/last_character_index` in the original file for evaluation.

`.py` and `.md`+`.txt` are indexed **separately** (different stopword / IDF
profiles, and questions route to the matching index).

---

## 🧭 Retrieval method

Three routes, chosen with `--retriever`:

| Method | How it climbs |
|---|---|
| `bm25` | Lexical scoring (`bm25s`) — exact token matching |
| `embedding` | Semantic cosine search (`BAAI/bge-small-en-v1.5`, normalised) |
| `hybrid` | Both branches fused with **Reciprocal Rank Fusion** |

**Query expansion** 🪢 (WordNet synonyms) is layered onto the BM25 branch to
catch paraphrases. It's auto-enabled for `bm25` only — embeddings already handle
meaning, so it would be redundant for `embedding`/`hybrid`. The original query is
always kept as the anchor, so exact matches are never lost.

---

## 🎒 Example usage

```bash
# Single question, full answer
uv run python src/main.py answer "How do I configure the OpenAI server?" --k 10

# Search a dataset, then generate answers from the results
uv run python src/main.py search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
uv run python src/main.py answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json

# Evaluate retrieval quality
uv run python src/main.py evaluate \
  --student_answer_path  data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

---

## 📊 Performance analysis

> _Numbers to be filled once measured — see the test conditions below._

### 🧪 Test conditions

| | |
|---|---|
| **Machine** | _CPU model / RAM / GPU (none or model)_ |
| **OS / Python** | _Linux_ / 3.10 |
| **Corpus** | vLLM 0.10.1 — ~14.6k chunks (13.7k `.py`, 0.9k `.md`/`.txt`) |
| **Datasets** | 100 docs + 100 code questions (public) |
| **Defaults** | `k=5`, `chunk_size=2000`, embedding `bge-small-en-v1.5`, LLM `Qwen3-0.6B` |
| **Cold start** | first query *including* model load · **Warm** = index + model already loaded |
| **Retriever measured** | _hybrid_ (the submitted default) |

### ✅ Mandatory thresholds (subject)

Measured with the default configuration above.

| Metric | Target | Result | Status |
|---|---|---|---|
| Recall@5 — docs | ≥ 80 % | _TBD_ | ⬜ |
| Recall@5 — code | ≥ 50 % | _TBD_ | ⬜ |
| Indexing time | ≤ 5 min | _TBD_ | ⬜ |
| Cold-start latency | ≤ 60 s | _TBD_ | ⬜ |
| Throughput — 1000 questions (warm) | ≤ 90 s | _TBD_ | ⬜ |

### 🧭 Retriever comparison (exploratory, not mandatory)

Same conditions, varying only `--retriever` (and query expansion for BM25).
Highlights the lexical / semantic / fused trade-off.

| Retriever | R@1 docs | R@5 docs | R@5 code | Index time | 1000 q (warm) |
|---|---|---|---|---|---|
| `bm25` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `bm25` + expansion | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `embedding` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `hybrid` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> Caching note: the figures above are **cold** (first run). A fully cached
> `make run` re-uses the stamped `index_id` and skips index/search/answer
> entirely — _TBD_ s end to end.

---

## 🧠 Design decisions

- **Two-level caching** ⛏️ — search and answer outputs embed a `meta` block; a
  content-fingerprint `index_id` is stamped at build time. Re-running with the
  same inputs skips the work entirely; changing the repo, params or model
  invalidates the right link of the chain automatically.
- **`text` stored in each source** — avoids re-reading files at answer time.
- **Lazy heavy imports** — `torch`/`transformers`/`langchain` load only when
  actually needed, keeping cached runs near-instant.
- **Deterministic everything** — chunk ids, synonym choice and ranking are
  reproducible, so `recall@k` is stable between runs.

---

## ⛰️ Challenges faced

- **The BGE query prefix** — passages and queries must be embedded
  asymmetrically; missing the query instruction silently hurts recall.
- **`langchain` pulling in `torch`** — a hidden 10 s import on every command,
  fixed by deferring it to the chunking step (the crux of the perf work).
- **Left padding for batched generation** — decoder models generate from the
  last token, so the batch must be left-padded.
- **Embedding indexing on CPU** — encoding ~15k chunks is slow without a GPU;
  use `RETRIEVER=bm25` for a quick send on a laptop.

---

## 📚 Resources & AI usage

- BM25 — [`bm25s`](https://github.com/xhluca/bm25s)
- Embeddings — [BGE](https://huggingface.co/BAAI/bge-small-en-v1.5),
  [sentence-transformers](https://www.sbert.net/)
- Fusion — [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- LLM — [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B),
  [vLLM](https://docs.vllm.ai/)

**AI usage:** an AI assistant was used as a belay partner 🧗 — to explore
library trade-offs (BM25 vs embeddings, chunking, fusion), draft boilerplate,
and review design decisions. Every choice was understood, tested and validated
before being committed.
