"""Indexing orchestrator: build_index, search_index, search_dataset."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from tqdm import tqdm

from constants import (
    BLUE, BOLD, COLOR_ACCENT, COLOR_SUCCESS, MAGENTA, colorize, divider
)
from models import (
    Chunk,
    IndexConfig,
    MinimalSearchResults,
    MinimalSource,
    RetrieverMethod,
    SearchConfig,
    SearchMeta,
    StudentSearchResults,
)
from indexing.chunking import chunk_repository, collect_files
from indexing.bm25 import build_bm25, load_bm25, search_bm25
from indexing.embedding import (
    build_embeddings,
    load_embeddings,
    rrf,
    search_embeddings,
)
from indexing.query_expansion import expand_query
from indexing.store import (
    has_bm25,
    has_embeddings,
    load_chunks,
    load_index_id,
    save_chunks,
    save_index_id,
)

logger = logging.getLogger(__name__)

_PY_SUBDIR = "py"
_MD_SUBDIR = "md"
_CODE_STOPWORDS = None
_DOCS_STOPWORDS = "english"
_FUSION_POOL = 20

# A loaded index: (method, chunks, bm25_retriever_or_none, embeddings_or_none).
LoadedIndex = Tuple[RetrieverMethod, List[Chunk], Any, Any]


def _split_chunks(
    chunks: List[Chunk],
) -> Tuple[List[Chunk], List[Chunk]]:
    py = [c for c in chunks if c.file_path.endswith(".py")]
    md = [c for c in chunks if c.file_path.endswith((".md", ".txt"))]
    return py, md


def _infer_index_subdir(dataset_path: str) -> str:
    """Return 'py' for code datasets, 'md' for docs datasets."""
    return _PY_SUBDIR if "code" in Path(dataset_path).stem else _MD_SUBDIR


def _chunks_to_sources(chunks: List[Chunk]) -> List[MinimalSource]:
    return [
        MinimalSource(
            file_path=c.file_path,
            first_character_index=c.first_character_index,
            last_character_index=c.last_character_index,
            text=c.text,
        )
        for c in chunks
    ]


def _should_expand(cfg: SearchConfig) -> bool:
    """Decide whether to apply query expansion for this search.

    Rule: explicit cfg.expand wins; otherwise auto — on for BM25 only,
    off for embedding and hybrid (the embedding handles semantics natively).
    """
    if cfg.expand is not None:
        return cfg.expand
    return cfg.retriever == RetrieverMethod.BM25


def _resolve_method(
    requested: RetrieverMethod,
    index_dir: str,
) -> RetrieverMethod:
    """Resolve the retriever to use given the artifacts actually present.

    Degrades gracefully to whatever index exists, logging a warning, and
    only fails when no usable index is found.

    Args:
        requested: The retriever method asked for.
        index_dir: Directory whose artifacts are inspected.

    Returns:
        The retriever method that can actually be served.

    Raises:
        FileNotFoundError: If neither a BM25 nor an embedding index exists.
    """
    bm = has_bm25(index_dir)
    emb = has_embeddings(index_dir)

    if not bm and not emb:
        raise FileNotFoundError(
            f"No index found in '{index_dir}'. Run 'make index' first."
        )

    if requested == RetrieverMethod.HYBRID:
        if bm and emb:
            return RetrieverMethod.HYBRID
        fallback = RetrieverMethod.BM25 if bm else RetrieverMethod.EMBEDDING
        logger.warning(
            "hybrid requested but only one index present in '%s'; "
            "using %s", index_dir, fallback.value,
        )
        return fallback

    if requested == RetrieverMethod.EMBEDDING and not emb:
        logger.warning(
            "embedding requested but absent in '%s'; using bm25", index_dir
        )
        return RetrieverMethod.BM25

    if requested == RetrieverMethod.BM25 and not bm:
        logger.warning(
            "bm25 requested but absent in '%s'; using embedding", index_dir
        )
        return RetrieverMethod.EMBEDDING

    return requested


def _load_for_search(index_dir: str, cfg: SearchConfig) -> LoadedIndex:
    """Resolve the method and load only the artifacts it needs."""
    method = _resolve_method(cfg.retriever, index_dir)
    chunks = load_chunks(index_dir)
    retriever = (
        load_bm25(index_dir)
        if method in (RetrieverMethod.BM25, RetrieverMethod.HYBRID)
        else None
    )
    matrix = (
        load_embeddings(index_dir)
        if method in (RetrieverMethod.EMBEDDING, RetrieverMethod.HYBRID)
        else None
    )
    return method, chunks, retriever, matrix


def _bm25_search(
    query: str,
    cfg: SearchConfig,
    retriever: Any,
    chunks: List[Chunk],
    stopwords: Optional[str],
    top_n: Optional[int] = None,
) -> List[Chunk]:
    """Run BM25 on one index, optionally with synonym query expansion.

    With expansion enabled, the query and its synonym variants are each
    searched and fused with RRF. The original query is always the anchor,
    so exact-term matches are never lost.

    Args:
        query: The search query.
        cfg: Search configuration.
        retriever: Loaded BM25 retriever.
        chunks: Chunks in corpus order.
        stopwords: Stopword set matching the index.
        top_n: Number of results to return (defaults to cfg.k).

    Returns:
        Top-n chunks for this index.
    """
    n = top_n if top_n is not None else cfg.k

    if not _should_expand(cfg):
        return search_bm25(query, retriever, chunks, n, stopwords)

    queries = expand_query(query)
    if len(queries) == 1:
        return search_bm25(query, retriever, chunks, n, stopwords)

    ranked_lists = [
        search_bm25(q, retriever, chunks, n, stopwords) for q in queries
    ]
    id_to_chunk = {c.id: c for c in chunks}
    return [id_to_chunk[cid] for cid in rrf(*ranked_lists)[:n]]


def _embedding_search(
    query: str,
    cfg: SearchConfig,
    matrix: Any,
    chunks: List[Chunk],
    top_n: Optional[int] = None,
) -> List[Chunk]:
    """Run embedding similarity search on one index."""
    n = top_n if top_n is not None else cfg.k
    return search_embeddings(query, matrix, chunks, n, cfg.embedding_model)


def _search_loaded(
    query: str,
    cfg: SearchConfig,
    loaded: LoadedIndex,
    stopwords: Optional[str],
) -> List[Chunk]:
    """Run the resolved retrieval method on a pre-loaded index."""
    method, chunks, retriever, matrix = loaded

    if method == RetrieverMethod.BM25:
        return _bm25_search(query, cfg, retriever, chunks, stopwords)
    if method == RetrieverMethod.EMBEDDING:
        return _embedding_search(query, cfg, matrix, chunks)

    # Hybrid: pull a deeper pool per branch so RRF can surface mid-rank
    # consensus, then cut the fused list to k.
    pool = max(cfg.k, _FUSION_POOL)
    bm = _bm25_search(query, cfg, retriever, chunks, stopwords, top_n=pool)
    em = _embedding_search(query, cfg, matrix, chunks, top_n=pool)
    id_to_chunk = {c.id: c for c in chunks}
    return [id_to_chunk[cid] for cid in rrf(bm, em)[: cfg.k]]


def _compute_index_id(cfg: IndexConfig) -> str:
    """Compute a content fingerprint of the would-be index.

    Hashes the indexing parameters plus, for every file to be indexed, its
    path, size and mtime. Identical inputs yield the same id, so an unchanged
    rebuild keeps downstream caches valid.

    Args:
        cfg: Indexing configuration.

    Returns:
        A 16-char hexadecimal fingerprint.
    """
    h = hashlib.sha256()
    h.update(
        f"{cfg.max_chunk_size}|{cfg.retriever.value}|"
        f"{cfg.embedding_model}".encode()
    )
    for path in collect_files(cfg.repo_path):
        st = path.stat()
        h.update(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def _index_artifacts_present(cfg: IndexConfig) -> bool:
    """Return True if every artifact required by cfg.retriever exists."""
    need_bm25 = cfg.retriever in (
        RetrieverMethod.BM25, RetrieverMethod.HYBRID
    )
    need_emb = cfg.retriever in (
        RetrieverMethod.EMBEDDING, RetrieverMethod.HYBRID
    )
    for subdir in (_PY_SUBDIR, _MD_SUBDIR):
        target = str(Path(cfg.output_path) / subdir)
        if need_bm25 and not has_bm25(target):
            return False
        if need_emb and not has_embeddings(target):
            return False
    return True


# ── Public API ───────────────────────────────────────────────────────────────

def build_index(cfg: IndexConfig) -> None:
    """Chunk the repository and build the requested indexes per file type.

    BM25 and/or embedding indexes are built depending on cfg.retriever.
    If an index with the same content fingerprint already exists, the
    rebuild is skipped entirely.

    Args:
        cfg: Indexing configuration (repo path, output, chunk size, method).
    """
    index_id = _compute_index_id(cfg)
    if (
        load_index_id(cfg.output_path) == index_id
        and _index_artifacts_present(cfg)
    ):
        print(colorize(
            f"  ✓  Index up to date ({index_id}) — skipping rebuild",
            COLOR_SUCCESS, BOLD,
        ))
        return

    chunks = chunk_repository(cfg.repo_path, cfg.max_chunk_size)
    py_chunks, md_chunks = _split_chunks(chunks)

    need_bm25 = cfg.retriever in (
        RetrieverMethod.BM25, RetrieverMethod.HYBRID
    )
    need_emb = cfg.retriever in (
        RetrieverMethod.EMBEDDING, RetrieverMethod.HYBRID
    )

    print()
    print(divider())
    print(colorize("  📚  INDEXING", MAGENTA, BOLD))
    print(divider(thin=True))
    print(f"  {'Repository':<26}" + colorize(cfg.repo_path, BOLD))
    print(
        f"  {'Max chunk size':<26}"
        + colorize(f"{cfg.max_chunk_size} chars", BOLD)
    )
    print(f"  {'Retriever':<26}" + colorize(cfg.retriever.value, BOLD))
    if need_emb:
        print(
            f"  {'Embedding model':<26}"
            + colorize(cfg.embedding_model, BOLD)
        )
    print(
        f"  {'Python chunks':<26}"
        + colorize(str(len(py_chunks)), BOLD, COLOR_SUCCESS)
    )
    print(
        f"  {'Markdown / txt chunks':<26}"
        + colorize(str(len(md_chunks)), BOLD, COLOR_SUCCESS)
    )
    print(divider(thin=True))
    print()
    print(colorize("  Building indexes…", BLUE, BOLD))
    print()

    for subdir, sub_chunks, stopwords in [
        (_PY_SUBDIR, py_chunks, _CODE_STOPWORDS),
        (_MD_SUBDIR, md_chunks, _DOCS_STOPWORDS),
    ]:
        target = str(Path(cfg.output_path) / subdir)
        save_chunks(sub_chunks, target)
        if need_bm25:
            build_bm25(sub_chunks, target, stopwords)
        if need_emb:
            build_embeddings(sub_chunks, target, cfg.embedding_model)

    # Stamp the content fingerprint so unchanged rebuilds keep caches valid.
    save_index_id(cfg.output_path, index_id)

    total_chunks = len(py_chunks) + len(md_chunks)
    print()
    print(divider(thin=True))
    print(
        colorize("  ✓  Ingestion complete!", COLOR_SUCCESS, BOLD)
        + f"  {colorize(str(total_chunks), BOLD)} chunks"
        + f"  →  {colorize(cfg.output_path + '/', COLOR_ACCENT)}"
    )
    print(divider())
    print()


def retrieve(query: str, cfg: SearchConfig) -> List[MinimalSource]:
    """Search both indexes and return top-k sources for a query.

    Args:
        query: The search query string.
        cfg: Search configuration (index path, k, retriever).

    Returns:
        Top-k MinimalSource objects (with text), best-ranked first.
    """
    base = Path(cfg.index_path)
    seen: set[str] = set()
    merged: List[Chunk] = []

    for subdir, stopwords in [
        (_PY_SUBDIR, _CODE_STOPWORDS),
        (_MD_SUBDIR, _DOCS_STOPWORDS),
    ]:
        loaded = _load_for_search(str(base / subdir), cfg)
        for chunk in _search_loaded(query, cfg, loaded, stopwords):
            if chunk.id not in seen:
                seen.add(chunk.id)
                merged.append(chunk)

    return _chunks_to_sources(merged[: cfg.k])


def search_index(query: str, cfg: SearchConfig) -> None:
    """Search both indexes for a single query and print JSON results.

    Args:
        query: The search query string.
        cfg: Search configuration (index path, k, retriever).
    """
    sources = retrieve(query, cfg)
    result = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="",
                question=query,
                retrieved_sources=sources,
            )
        ],
        k=cfg.k,
    )
    print(result.model_dump_json(indent=2))


def _build_meta(cfg: SearchConfig) -> SearchMeta:
    """Build the cache metadata from the current search configuration."""
    return SearchMeta(
        retriever=cfg.retriever.value,
        k=cfg.k,
        expand=cfg.expand,
        embedding_model=cfg.embedding_model,
        max_chunk_size=cfg.max_chunk_size,
        index_id=load_index_id(cfg.index_path),
    )


def _cache_is_valid(
    out_file: Path,
    meta: SearchMeta,
    question_ids: List[str],
) -> bool:
    """Return True if an existing output can be reused as-is.

    Reuse requires the stored meta to match the current config exactly and
    the question ids to be identical (same dataset, same order).
    """
    if not out_file.exists():
        return False
    try:
        with out_file.open(encoding="utf-8") as f:
            existing = StudentSearchResults(**json.load(f))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return False
    if existing.meta != meta:
        return False
    existing_ids = [r.question_id for r in existing.search_results]
    return existing_ids == question_ids


def search_dataset(
    dataset_path: str,
    save_directory: str,
    cfg: SearchConfig,
    limit: int = 0,
) -> None:
    """Search for every question in a dataset file and save results as JSON.

    Result caching: if an output already exists whose meta matches the
    current config and whose question ids are identical, it is reused and
    no search is run. Rebuilding the index changes its id and invalidates
    the cache automatically.

    Args:
        dataset_path: Path to the UnansweredQuestions JSON dataset.
        save_directory: Directory where the output JSON will be written.
        cfg: Search configuration (index path, k, retriever).
        limit: Maximum number of questions to process (0 = all).
    """
    with open(dataset_path, encoding="utf-8") as f:
        questions = json.load(f)["rag_questions"]
    if limit > 0:
        questions = questions[:limit]
    question_ids = [q["question_id"] for q in questions]

    meta = _build_meta(cfg)
    out_file = Path(save_directory) / Path(dataset_path).name

    if _cache_is_valid(out_file, meta, question_ids):
        print(colorize(
            f"  ✓  Cache hit — reusing {out_file}", COLOR_SUCCESS, BOLD
        ))
        return

    subdir = _infer_index_subdir(dataset_path)
    stopwords = _CODE_STOPWORDS if subdir == _PY_SUBDIR else _DOCS_STOPWORDS
    loaded = _load_for_search(str(Path(cfg.index_path) / subdir), cfg)

    results: List[MinimalSearchResults] = []
    for q in tqdm(questions, desc="Searching", unit="q"):
        top_k = _search_loaded(q["question"], cfg, loaded, stopwords)
        results.append(
            MinimalSearchResults(
                question_id=q["question_id"],
                question=q["question"],
                retrieved_sources=_chunks_to_sources(top_k),
            )
        )

    output = StudentSearchResults(meta=meta, search_results=results, k=cfg.k)

    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    print(f"Saved {len(results)} results -> {out_file}")
