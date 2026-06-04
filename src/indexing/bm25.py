"""BM25 index build, persist, load, and search."""

from pathlib import Path
from typing import List, Optional

import bm25s

from indexing.store import BM25_SUBDIR
from models import Chunk


def build_bm25(
    chunks: List[Chunk],
    output_dir: str,
    stopwords: Optional[str] = None,
) -> None:
    """Build and persist a BM25 index from a list of chunks.

    Chunks themselves are persisted separately via store.save_chunks.

    Args:
        chunks: Ordered list of Chunk objects to index.
        output_dir: Directory where the BM25 index will be saved.
        stopwords: Stopword set ('english') or None to keep all tokens.

    Raises:
        RuntimeError: If the index cannot be written.
    """
    texts = [chunk.text for chunk in chunks]
    corpus_tokens = bm25s.tokenize(texts, stopwords=stopwords)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    try:
        retriever.save(str(Path(output_dir) / BM25_SUBDIR))
    except OSError as e:
        raise RuntimeError(
            f"Cannot save BM25 index in '{output_dir}': {e}"
        ) from e


def load_bm25(index_dir: str) -> bm25s.BM25:
    """Load a BM25 retriever from disk.

    Args:
        index_dir: Directory containing the bm25_index/ subdirectory.

    Returns:
        The loaded BM25 retriever.

    Raises:
        FileNotFoundError: If the BM25 index is absent.
        RuntimeError: If the index is corrupted.
    """
    bm25_path = Path(index_dir) / BM25_SUBDIR
    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index missing at '{bm25_path}'. Re-run 'make index'."
        )
    try:
        return bm25s.BM25.load(str(bm25_path), load_corpus=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load BM25 index from '{bm25_path}': {e}"
        ) from e


def search_bm25(
    query: str,
    retriever: bm25s.BM25,
    chunks: List[Chunk],
    k: int,
    stopwords: Optional[str] = None,
) -> List[Chunk]:
    """Retrieve the top-k chunks for a query using BM25.

    Args:
        query: Search query string.
        retriever: Loaded BM25 retriever.
        chunks: Chunks in the same order as the index corpus.
        k: Number of results to return.
        stopwords: Must match the stopwords used during indexing.

    Returns:
        Top-k Chunk objects ranked by BM25 score.
    """
    query_tokens = bm25s.tokenize([query], stopwords=stopwords)
    results, _ = retriever.retrieve(query_tokens, k=min(k, len(chunks)))
    return [chunks[int(i)] for i in results[0]]
