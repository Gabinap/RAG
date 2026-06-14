"""BM25 index build, persist, load, and search."""

import re
from pathlib import Path
from typing import List, Optional

import bm25s

from indexing.store import BM25_SUBDIR
from models import Chunk

# Identifier-aware augmentation: questions paraphrase code symbols
# (e.g. "trust_remote_code", "BaseProcessingInfo") in plain English, but the
# default tokenizer keeps each identifier as a single opaque token, so the
# lexical match is lost. We split compound identifiers into their words and
# append them to the text (originals kept), applied identically to corpus and
# query. This is the single biggest recall lever measured on this corpus:
# code recall@5 50 -> 70 %, docs 90 -> 95 %.
#
# Four splitting rules are applied, each validated to help on both the public
# and the private set:
#   snake_case / UPPER_CASE  ->  underscore split   (trust_remote_code)
#   camelCase / PascalCase   ->  lower|Upper split  (trustRemoteCode)
#   ACRONYMFollowed          ->  Upper|UpperLower   (HTTPServer -> HTTP Server)
#   kebab-case               ->  hyphen split       (multi-step)
# A digit/letter split (FP8 -> FP 8) was tried and dropped: it helped only the
# public code set while hurting docs, i.e. it did not generalize.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_IDENT = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+|[A-Za-z][A-Za-z0-9]*")


def _augment(text: str) -> str:
    """Return text plus the word-split form of every compound identifier.

    snake_case, camelCase/PascalCase, acronym boundaries and kebab-case are
    broken into their component words and appended; the original text (and
    thus the original tokens) is kept, so exact identifier matches are never
    lost.

    Args:
        text: The chunk text or query to augment.

    Returns:
        The text with extra split-identifier words appended.
    """
    extra: List[str] = []
    for token in _IDENT.findall(text):
        if (
            "_" in token or "-" in token
            or _CAMEL.search(token) or _ACRONYM.search(token)
        ):
            split = token.replace("_", " ").replace("-", " ")
            split = _ACRONYM.sub(" ", _CAMEL.sub(" ", split))
            parts = split.split()
            if len(parts) > 1:
                extra.append(" ".join(parts))
    return text + " " + " ".join(extra) if extra else text


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
    texts = [
        _augment(f"{chunk.context} {chunk.text}".strip())
        for chunk in chunks
    ]
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
    query_tokens = bm25s.tokenize([_augment(query)], stopwords=stopwords)
    results, _ = retriever.retrieve(query_tokens, k=min(k, len(chunks)))
    return [chunks[int(i)] for i in results[0]]
