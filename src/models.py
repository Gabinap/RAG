"""Pydantic data models and CLI configuration definitions."""

import uuid
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, computed_field, field_validator


class MinimalSource(BaseModel):
    """A source location inside a repository file."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: Optional[str] = None


class UnansweredQuestion(BaseModel):
    """A question without an answer or sources."""

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth answer and sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions (answered or unanswered)."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Search results for a single question.

    The subject's model names the field ``question``; the moulinette validator
    requires ``question_str``. We emit both: ``question`` is the real field,
    ``question_str`` is a computed mirror so the output validates against the
    moulinette without changing internal code.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def question_str(self) -> str:
        """Alias of ``question`` required by the moulinette validator."""
        return self.question


class MinimalAnswer(MinimalSearchResults):
    """Search results with a generated answer for a single question."""

    answer: str


class SearchMeta(BaseModel):
    """Metadata describing how a search output was produced (for caching).

    A cached output is reused only if its meta matches the current config
    exactly and the question ids are identical.
    """

    retriever: str
    k: int
    expand: bool = False
    embedding_model: str
    max_chunk_size: int
    index_id: Optional[str] = None


class StudentSearchResults(BaseModel):
    """Full search results output for a dataset."""

    meta: Optional[SearchMeta] = None
    search_results: List[MinimalSearchResults]
    k: int


class AnswerMeta(BaseModel):
    """Metadata describing how answers were generated (for caching).

    Cached answers are reused only if these generation parameters and the
    metadata of the source search results match exactly.
    """

    model_name: str
    max_new_tokens: int
    max_chunk_size: int
    source_meta: Optional[SearchMeta] = None


class StudentSearchResultsAndAnswer(BaseModel):
    """Full search results with answers output for a dataset."""

    meta: Optional[AnswerMeta] = None
    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """A chunk of text extracted from a source file.

    ``context`` holds retrieval-only metadata (e.g. the markdown header
    breadcrumb above the chunk). It is prepended to the text *only* when
    tokenizing for BM25 — it never alters ``text`` or the character indices,
    so evaluation positions and answer context stay exact.
    """

    id: str
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    context: str = ""


class RetrieverMethod(str, Enum):
    """Available retrieval methods."""

    BM25 = "bm25"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"
    AUTO = "auto"


class IndexSubset(str, Enum):
    """Which file types to index."""

    ALL = "all"
    CODE = "code"
    DOCS = "docs"


class IndexConfig(BaseModel):
    """Configuration for the index command."""

    repo_path: str = Field(
        default="data/raw/vllm-0.10.1",
        description="Path to the vLLM repository to index.",
    )
    output_path: str = Field(
        default="data/processed",
        description="Directory where the index will be saved.",
    )
    max_chunk_size: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Maximum number of characters per chunk.",
    )
    retriever: RetrieverMethod = Field(
        default=RetrieverMethod.HYBRID,
        description="Retrieval method: bm25, embedding, or hybrid.",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="SentenceTransformer model used for embeddings.",
    )
    index_subset: IndexSubset = Field(
        default=IndexSubset.ALL,
        description="File types to index: all, code (.py), or docs (.md).",
    )


class SearchConfig(BaseModel):
    """Configuration for search and answer commands (retrieval phase)."""

    index_path: str = Field(
        default="data/processed",
        description="Path to the saved index directory.",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of top results to retrieve.",
    )
    retriever: RetrieverMethod = Field(
        default=RetrieverMethod.HYBRID,
        description="Retrieval method: bm25, embedding, or hybrid.",
    )
    max_chunk_size: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Maximum context length in characters passed to the LLM.",
    )
    expand: bool = Field(
        default=False,
        description="Query expansion via WordNet synonyms (default off).",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="SentenceTransformer model used for embeddings.",
    )


class GenerationConfig(BaseModel):
    """Configuration for the generation phase (answer and answer_dataset)."""

    model_name: str = Field(
        default="Qwen/Qwen3-0.6B",
        description="HuggingFace model identifier to use for generation.",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Maximum number of tokens to generate.",
    )


class EvaluateConfig(BaseModel):
    """Configuration for the evaluate command."""

    k_values: List[int] = Field(
        default=[1, 3, 5, 10],
        description="List of k values for recall@k computation.",
    )
    min_overlap: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum character overlap ratio to count a source as found."
        ),
    )

    @field_validator("k_values")
    @classmethod
    def sort_and_deduplicate(cls, v: List[int]) -> List[int]:
        """Sort and deduplicate k values, reject non-positive entries."""
        if not v or any(k <= 0 for k in v):
            raise ValueError(
                "k_values must be a non-empty list of positive integers."
            )
        return sorted(set(v))
