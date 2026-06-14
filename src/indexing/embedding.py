"""Embedding retrieval: model singleton, build, load, search, fusion."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from constants import BLUE, BOLD, COLOR_SUCCESS, colorize
from indexing.store import EMBEDDING_CHUNKS_FILE, EMBEDDINGS_FILE
from models import Chunk

# BGE models expect an instruction on the query side only (not on passages).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Batch sizes tuned for CPU to avoid L3-cache thrashing with long chunks.
_BATCH_SIZE_CPU = {
    "tiny":  32, "micro":  32, "small":  32, "base": 16, "large":  8,
}


def _batch_size_for(model_name: str) -> int:
    """Return a sensible encode batch size for the model."""
    lower = model_name.lower()
    for key, size in _BATCH_SIZE_CPU.items():
        if key in lower:
            return size
    return 32


_model: Optional[Any] = None
_loaded_model_name: Optional[str] = None


def get_embedding_model(model_name: str) -> Any:
    """Load and cache the embedding model (lazy singleton).

    Args:
        model_name: SentenceTransformer model identifier.

    Returns:
        The cached SentenceTransformer instance.
    """
    global _model, _loaded_model_name
    if _model is not None and _loaded_model_name == model_name:
        return _model

    import torch
    from sentence_transformers import SentenceTransformer

    print(colorize(f"  🧠  Loading embedding model: {model_name}", BLUE, BOLD))

    _model = SentenceTransformer(model_name, device="cpu")
    _model.float()
    torch.quantization.quantize_dynamic(
        _model,
        {torch.nn.Linear},
        dtype=torch.qint8,
        inplace=True,
    )
    print(colorize(
        "  ⚡  INT8 dynamic quantisation applied", COLOR_SUCCESS, BOLD
    ))

    _loaded_model_name = model_name
    print(colorize("  ✓  Embedding model ready", COLOR_SUCCESS, BOLD))
    return _model


def _query_prefix(model_name: str) -> str:
    """Return the query-side instruction prefix for the model, if any."""
    return _BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""


def build_embeddings(
    chunks: List[Chunk],
    output_dir: str,
    model_name: str,
) -> None:
    """Encode chunk texts and persist normalised embeddings as .npy.

    Row i of the saved matrix corresponds to chunks[i] (corpus order).

    Args:
        chunks: Ordered list of Chunk objects to encode.
        output_dir: Directory where embeddings.npy will be saved.
        model_name: SentenceTransformer model identifier.

    Raises:
        RuntimeError: If the embeddings file cannot be written.
    """
    model = get_embedding_model(model_name)
    texts = [chunk.text for chunk in chunks]
    bs = _batch_size_for(model_name)
    print(colorize(f"  ⚡  Encoding batch_size={bs}", BLUE, BOLD))
    embeddings = model.encode(
        texts,
        batch_size=bs,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    out = Path(output_dir)
    try:
        np.save(str(out / EMBEDDINGS_FILE), embeddings)
        with (out / EMBEDDING_CHUNKS_FILE).open("w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f)
    except OSError as e:
        raise RuntimeError(
            f"Cannot save embeddings in '{output_dir}': {e}"
        ) from e


def load_embeddings(index_dir: str) -> Tuple[np.ndarray, List[Chunk]]:
    """Load the embedding matrix and its aligned chunk list from disk.

    Args:
        index_dir: Directory containing embeddings.npy and
            embedding_chunks.json.

    Returns:
        Tuple of (matrix, chunks) where matrix row i corresponds to chunks[i].

    Raises:
        FileNotFoundError: If either file is absent.
    """
    d = Path(index_dir)
    mat_path = d / EMBEDDINGS_FILE
    chunks_path = d / EMBEDDING_CHUNKS_FILE
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Embeddings missing at '{mat_path}'. Re-run 'make index'."
        )
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Embedding chunks missing at '{chunks_path}'."
            " Re-run 'make index'."
        )
    matrix: np.ndarray = np.load(str(mat_path))
    with chunks_path.open(encoding="utf-8") as f:
        emb_chunks = [Chunk(**c) for c in json.load(f)]
    return matrix, emb_chunks


def search_embeddings(
    query: str,
    matrix: np.ndarray,
    chunks: List[Chunk],
    k: int,
    model_name: str,
) -> List[Chunk]:
    """Retrieve the top-k chunks for a query by cosine similarity.

    Args:
        query: Search query string.
        matrix: Normalised embedding matrix aligned with chunks.
        chunks: Chunks in the same order as the matrix rows.
        k: Number of results to return.
        model_name: Model used to encode the query (must match the index).

    Returns:
        Top-k Chunk objects ranked by cosine similarity.
    """
    model = get_embedding_model(model_name)
    prefixed = _query_prefix(model_name) + query
    query_vec = model.encode(
        [prefixed],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    # Both sides are unit-normalised, so the dot product is the cosine.
    similarities = matrix @ query_vec
    n = min(k, len(chunks))
    top_unsorted = np.argpartition(-similarities, n - 1)[:n]
    top_sorted = top_unsorted[np.argsort(-similarities[top_unsorted])]
    return [chunks[int(i)] for i in top_sorted]


def rrf(*ranked_lists: List[Any], k: int = 60) -> List[str]:
    """Combine several ranked result lists with Reciprocal Rank Fusion.

    Works for any number of ranked lists: query-expansion variants (many
    BM25 lists) or hybrid retrieval (BM25 list + embedding list).

    Args:
        *ranked_lists: One or more ranked lists of chunks (each chunk must
            expose an ``id`` attribute).
        k: RRF constant controlling rank influence (default 60).

    Returns:
        List of chunk IDs sorted by combined RRF score descending.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)
