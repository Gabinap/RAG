"""Shared on-disk persistence for chunks and index artifact detection."""

import json
from pathlib import Path
from typing import List, Optional

from models import Chunk

CHUNKS_FILE = "chunks.json"
BM25_SUBDIR = "bm25_index"
EMBEDDINGS_FILE = "embeddings.npy"
INDEX_ID_FILE = "index_id.txt"


def save_chunks(chunks: List[Chunk], output_dir: str) -> None:
    """Persist chunks as JSON in output_dir (corpus order preserved).

    Args:
        chunks: Ordered list of Chunk objects.
        output_dir: Directory where chunks.json will be written.

    Raises:
        RuntimeError: If the directory or file cannot be written.
    """
    out = Path(output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        with (out / CHUNKS_FILE).open("w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in chunks], f)
    except OSError as e:
        raise RuntimeError(
            f"Cannot write chunks file in '{output_dir}': {e}"
        ) from e


def load_chunks(index_dir: str) -> List[Chunk]:
    """Load chunks from chunks.json in index_dir.

    Args:
        index_dir: Directory containing chunks.json.

    Returns:
        List of Chunk objects in corpus order.

    Raises:
        FileNotFoundError: If chunks.json is absent.
        RuntimeError: If chunks.json is corrupted.
    """
    path = Path(index_dir) / CHUNKS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file missing at '{path}'. Re-run 'make index'."
        )
    try:
        with path.open(encoding="utf-8") as f:
            return [Chunk(**c) for c in json.load(f)]
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"Failed to parse chunks file '{path}': {e}"
        ) from e


def save_index_id(output_path: str, index_id: str) -> None:
    """Persist the index identifier.

    The id is a content fingerprint (see indexing.build_index): identical
    inputs yield the same id, so unchanged rebuilds keep caches valid.

    Args:
        output_path: Root index directory (where index_id.txt is written).
        index_id: The fingerprint to persist.
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / INDEX_ID_FILE).write_text(index_id, encoding="utf-8")


def load_index_id(index_dir: str) -> Optional[str]:
    """Return the persisted index id, or None if absent."""
    path = Path(index_dir) / INDEX_ID_FILE
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def has_bm25(index_dir: str) -> bool:
    """Return True if a BM25 index exists in index_dir."""
    return (Path(index_dir) / BM25_SUBDIR).exists()


def has_embeddings(index_dir: str) -> bool:
    """Return True if an embedding index exists in index_dir."""
    return (Path(index_dir) / EMBEDDINGS_FILE).exists()
