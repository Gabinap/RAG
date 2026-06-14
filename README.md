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
make run                         # bm25, the default — best recall, ~4 s index
make run RETRIEVER=hybrid        # BM25 + embeddings (slower on CPU, lower recall)
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

**Header breadcrumb** 🧭 — each markdown chunk carries the trail of the last
three headers above it (e.g. *Quantization › FP8 › Accuracy*) in a separate
`context` field. This breadcrumb is prepended to the text **only when tokenizing
for BM25** — it never touches the stored text or character indices — so
topic-level questions match the right section. It lifts docs recall@5 from
93 % to 94 %.

---

## 🧭 Retrieval method

Four routes, chosen with `--retriever`:

| Method | How it climbs |
|---|---|
| `bm25` *(default)* | Lexical scoring (`bm25s`) with identifier-aware tokenization |
| `embedding` | Semantic cosine search (`TaylorAI/gte-tiny`, normalised) |
| `hybrid` | BM25 + embedding fused with **Reciprocal Rank Fusion** |
| `auto` | Resolves to `bm25` for both docs and code |

**Identifier-aware tokenization** 🔑 is the key recall lever. Questions
paraphrase code symbols in plain English (`trust_remote_code`,
`tie_word_embeddings`, `BaseProcessingInfo`), but a default tokenizer keeps each
identifier as one opaque token, so the lexical match is lost. We split compound
identifiers into their component words — on both the corpus and the query,
keeping the originals — with four rules, each kept only after it was measured to
help on a held-out set:

| Rule | Example |
|---|---|
| `snake_case` / `UPPER_CASE` | `trust_remote_code` → `trust remote code` |
| `camelCase` / `PascalCase` | `BaseModelLoader` → `Base Model Loader` |
| acronym boundary | `HTTPServer` → `HTTP Server` |
| `kebab-case` | `multi-step` → `multi step` |

This lifts code recall@5 from 50 % to 70 % and docs from 89 % to 93 % (a
digit/letter split, `FP8` → `FP 8`, was tried and dropped — it only helped the
public code set). See `_augment` in [`src/indexing/bm25.py`](src/indexing/bm25.py).

**Query expansion** 🪢 (WordNet synonyms) is available via `--expand True` but
**off by default**: it was measured to *reduce* recall on this corpus.

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

All figures below are **measured** with the default configuration
(`make run`): BM25 with identifier-aware tokenization, `k=5`,
`chunk_size=2000`, on CPU.

### 🧪 Test conditions

| | |
|---|---|
| **Machine** | Intel Core Ultra 7 265 (20 threads) · 16 GB RAM · CPU-only inference (no CUDA; the discrete GPU is unused) |
| **OS / Python** | Ubuntu 22.04.5 LTS · kernel 6.8 / Python 3.10.20 |
| **Corpus** | vLLM 0.10.1 — ~14.6k chunks (13.7k `.py`, 0.9k `.md`/`.txt`) |
| **Datasets** | 100 docs + 100 code questions (public) |
| **Defaults** | `k=5`, `chunk_size=2000`, retriever `bm25`, LLM `Qwen3-0.6B` |
| **Cold start** | a fresh process answering a single query, index load included |
| **Warm throughput** | index loaded once, then 1000 queries searched in-process |
| **Retriever measured** | `bm25` with identifier-aware tokenization (both indexes) |

### ✅ Mandatory thresholds (subject)

Measured with the default configuration above. Recall@k uses the same metric
as the official moulinette: a source counts as found at **IoU ≥ 5 %** (character
ranges) — `make evaluate` reproduces the grader's numbers exactly.

| Metric | Target | Result | Status |
|---|---|---|---|
| Recall@5 — docs | ≥ 80 % | **94 %** | ✅ |
| Recall@5 — code | ≥ 50 % | **70 %** | ✅ |
| Indexing time | ≤ 5 min | **4.3 s** | ✅ |
| Cold-start latency | ≤ 60 s | **0.4 s** | ✅ |
| Throughput — 1000 questions (warm) | ≤ 90 s | **74.6 s** | ✅ |

Full recall breakdown for the default `bm25` retriever (100 questions each):

| | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| docs | 69 % | 89 % | 94 % | 96 % |
| code | 44 % | 62 % | 70 % | 79 % |

### 🧭 How the recall was built up

The default retriever is `bm25`; each lever below was added on top and kept
only after it was measured to help (numbers are recall@5, IoU metric, public
set).

| Step | R@5 docs | R@5 code |
|---|---|---|
| **`bm25`** lexical (baseline) | 89 % | 50 % |
| + identifier splitting (snake / camelCase) | 92 % | 69 % |
| + acronym & kebab-case splitting | 93 % | 70 % |
| + header breadcrumb on markdown chunks **(default)** | **94 %** | **70 %** |

Identifier splitting is the big lever for **code** (+19), the extra splitting
rules and the header breadcrumb close the gap on **docs** (+5 over plain BM25).
`embedding` (gte-tiny) and `hybrid` were tried as bonus alternatives and both
**underperform** lexical retrieval here (docs ~73 / ~86, code ~47 / ~58).

> Why no embeddings by default: encoding the 13.7k code chunks with a stronger
> model (e.g. `bge-small`) takes ~15 min on CPU — over the 5-min indexing cap —
> and still does not beat BM25 here. On this corpus, lexical retrieval with
> identifier splitting wins outright while keeping indexing near-instant.

---

## 🧠 Design decisions

- **Two-level caching** ⛏️ — search and answer outputs embed a `meta` block; a
  content-fingerprint `index_id` is stamped at build time. Re-running with the
  same inputs skips the work entirely; changing the repo, params or model
  invalidates the right link of the chain automatically.

  | When | What is cached | Where |
  |---|---|---|
  | `index` | chunks + BM25/embedding indexes + `index_id` fingerprint | `data/processed/{py,md}/` |
  | `search_dataset` | search results (`meta` + sources) | `data/output/search_results/` |
  | `answer_dataset` | results + generated answers (`meta`) | `data/output/search_results_and_answer/` |

  Each step skips itself on the next run if its `meta` and question ids still
  match; nothing is written outside `data/` (it is git-ignored).
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
- **Embedding indexing on CPU** — encoding ~15k chunks is slow without a GPU
  (a stronger model like `bge-small` needs ~15 min, over the 5-min cap). This,
  plus the recall results, is why `bm25` is the default and embeddings are an
  optional bonus.
- **Lexical match on code symbols** — the decisive insight: natural-language
  questions rarely contain a code identifier verbatim, so splitting
  `snake_case`/`camelCase` on both sides was worth far more than any semantic
  model on this corpus (+19 pts code recall@5).

---

## 📚 Resources & AI usage

### Papers & articles

- **RAG** — Lewis et al., 2020. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — the foundational paper introducing RAG.
- **BM25** — Robertson & Zaragoza, 2009. [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.nowpublishers.com/article/Details/INR-019) — the ranking function behind the lexical retrieval branch.
- **RRF** — Cormack et al., 2009. [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — the fusion strategy used for hybrid retrieval.
- **Dense retrieval** — Karpukhin et al., 2020. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — motivation for embedding-based retrieval alongside BM25.
- **RAG survey** — Gao et al., 2023. [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — comprehensive overview of RAG techniques and architectures.

### Libraries & tools

- BM25 — [`bm25s`](https://github.com/xhluca/bm25s)
- Embeddings — [GTE-tiny](https://huggingface.co/TaylorAI/gte-tiny),
  [sentence-transformers](https://www.sbert.net/)
- Chunking — [langchain-text-splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- LLM — [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B),
  [vLLM](https://docs.vllm.ai/)

### AI usage

An AI assistant was used as a belay partner 🧗 — to explore library trade-offs
(BM25 vs embeddings, chunking strategies, RRF fusion), draft boilerplate, and
review design decisions. Every choice was understood, tested and validated before
being committed.
