"""File collection and chunking for .py and .md files."""

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from models import Chunk

_CHUNK_OVERLAP = 200
# Suffix -> langchain Language enum NAME. The enum (and langchain, which
# transitively pulls torch) is imported lazily in chunk_file so that search,
# answer and evaluate — which never chunk — start fast.
_EXTENSIONS: Dict[str, str] = {
    ".py": "PYTHON",
    ".md": "MARKDOWN",
    ".txt": "MARKDOWN",
}


def collect_files(repo_path: str) -> List[Path]:
    """Return all .py and .md files under repo_path, sorted.

    Args:
        repo_path: Root directory of the repository to scan.

    Returns:
        Sorted list of matching file paths.
    """
    root = Path(repo_path)
    if not root.exists():
        raise FileNotFoundError(
            f"Repository path '{repo_path}' does not exist."
        )
    if not root.is_dir():
        raise NotADirectoryError(
            f"'{repo_path}' is not a directory."
        )
    files: List[Path] = []
    for ext in _EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def _chunk_id(file_path: str, start: int) -> str:
    return hashlib.md5(f"{file_path}:{start}".encode()).hexdigest()


def _find_positions(
    content: str, chunks: List[str]
) -> List[Tuple[int, int]]:
    """Locate each chunk's start/end character index inside content.

    Scans sequentially to handle overlapping chunks correctly.

    Args:
        content: Original file content.
        chunks: Ordered list of text chunks produced by the splitter.

    Returns:
        List of (first_character_index, last_character_index) tuples.
    """
    positions: List[Tuple[int, int]] = []
    search_from = 0
    for chunk in chunks:
        start = content.find(chunk, search_from)
        if start == -1:
            start = content.find(chunk)
        end = start + len(chunk)
        positions.append((start, end))
        search_from = start + 1
    return positions


def chunk_file(
    path: Path,
    content: str,
    max_chunk_size: int,
) -> List[Chunk]:
    """Split a single file into Chunk objects with character positions.

    Args:
        path: Path to the source file (used as file_path in chunks).
        content: Full text content of the file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        List of Chunk objects with correct character indices.
    """
    language_name = _EXTENSIONS.get(path.suffix)
    if language_name is None:
        return []

    from langchain_text_splitters import (
        Language,
        RecursiveCharacterTextSplitter,
    )

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language[language_name],
        chunk_size=max_chunk_size,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    raw_chunks = splitter.split_text(content)
    if not raw_chunks:
        return []

    file_path_str = str(path)
    positions = _find_positions(content, raw_chunks)

    return [
        Chunk(
            id=_chunk_id(file_path_str, start),
            file_path=file_path_str,
            first_character_index=start,
            last_character_index=end,
            text=text,
        )
        for text, (start, end) in zip(raw_chunks, positions)
        if text.strip()
    ]


def chunk_repository(repo_path: str, max_chunk_size: int) -> List[Chunk]:
    """Collect and chunk all .py and .md files in the repository.

    Args:
        repo_path: Root directory of the repository to index.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Flat list of all Chunk objects across all files.
    """
    files = collect_files(repo_path)
    chunks: List[Chunk] = []

    for path in tqdm(files, desc="Chunking", unit="file"):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chunks.extend(chunk_file(path, content, max_chunk_size))

    return chunks
