"""RAG against the machine — CLI entry point."""

# 1. Ingest the vLLM repository (provided as attachment) and create
#    a searchable knowledge base
# Is there a lib fr that ?
# 2. Search this knowledge base to find relevant code snippets and
#    documentation for given questions
# BM25 | embedding model
# 3. Answer questions using an LLM (Qwen/Qwen3-0.6B) with the retrieved context
# Retrieved context + question + contexte ?
# 4. Evaluate your retrieval system's quality using recall@k metrics
# maybe ask to co and cle

import logging
import sys
from pathlib import Path
from typing import List, Optional

import fire

from evaluation.metrics import evaluate_results
from generation.inference import answer_dataset, answer_query
from indexing import build_index, search_dataset, search_index
from models import (
    EvaluateConfig,
    GenerationConfig,
    IndexConfig,
    RetrieverMethod,
    SearchConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers
for _noisy in ("httpx", "huggingface_hub", "bm25s", "transformers"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


class RAG:
    """Retrieval-Augmented Generation system over the vLLM repository."""

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        output_path: str = "data/processed",
        max_chunk_size: int = 2000,
        retriever: str = "hybrid",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        """Index the vLLM repository and persist the search index.

        Args:
            repo_path: Path to the vLLM repository to index.
            output_path: Directory where the index will be saved.
            max_chunk_size: Maximum number of characters per chunk (<=10000).
            retriever: Retrieval method — bm25, embedding, or hybrid.
            embedding_model: SentenceTransformer model for embeddings.
        """
        try:
            cfg = IndexConfig(
                repo_path=repo_path,
                output_path=output_path,
                max_chunk_size=max_chunk_size,
                retriever=RetrieverMethod(retriever),
                embedding_model=embedding_model,
            )
            build_index(cfg)
        except Exception as e:
            logger.error("index failed: %s", e)
            sys.exit(1)

    def search(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed",
        retriever: str = "hybrid",
        max_chunk_size: int = 2000,
        expand: Optional[bool] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        """Search the index for a single query and print results as JSON.

        Args:
            query: The question or search query.
            k: Number of top results to retrieve.
            index_path: Path to the saved index directory.
            retriever: Retrieval method — bm25, embedding, or hybrid.
            max_chunk_size: Maximum context length in characters.
            expand: Query expansion — None=auto (bm25 only), True/False=force.
            embedding_model: SentenceTransformer model for embeddings.
        """
        try:
            cfg = SearchConfig(
                index_path=index_path,
                k=k,
                retriever=RetrieverMethod(retriever),
                max_chunk_size=max_chunk_size,
                expand=expand,
                embedding_model=embedding_model,
            )
            search_index(query=query, cfg=cfg)
        except Exception as e:
            logger.error("search failed: %s", e)
            sys.exit(1)

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str = "data/output/search_results",
        k: int = 5,
        index_path: str = "data/processed",
        retriever: str = "hybrid",
        max_chunk_size: int = 2000,
        limit: int = 0,
        expand: Optional[bool] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        """Process all questions in a dataset and save search results.

        Args:
            dataset_path: Path to the JSON dataset (UnansweredQuestions).
            save_directory: Directory where search results JSON will be saved.
            k: Number of top results per question.
            index_path: Path to the saved index directory.
            retriever: Retrieval method — bm25, embedding, or hybrid.
            max_chunk_size: Maximum context length in characters.
            limit: Max questions to process (0 = all).
            expand: Query expansion — None=auto (bm25 only), True/False=force.
            embedding_model: SentenceTransformer model for embeddings.
        """
        try:
            cfg = SearchConfig(
                index_path=index_path,
                k=k,
                retriever=RetrieverMethod(retriever),
                max_chunk_size=max_chunk_size,
                expand=expand,
                embedding_model=embedding_model,
            )
            search_dataset(
                dataset_path=dataset_path,
                save_directory=save_directory,
                cfg=cfg,
                limit=limit,
            )
        except Exception as e:
            logger.error("search_dataset failed: %s", e)
            sys.exit(1)

    def answer(
        self,
        query: str,
        k: int = 5,
        index_path: str = "data/processed",
        retriever: str = "hybrid",
        max_chunk_size: int = 2000,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 512,
        expand: Optional[bool] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        """Answer a single question using retrieved context and the LLM.

        Args:
            query: The question to answer.
            k: Number of retrieved sources to include in context.
            index_path: Path to the saved index directory.
            retriever: Retrieval method — bm25, embedding, or hybrid.
            max_chunk_size: Maximum context length in characters.
            model_name: HuggingFace model identifier for generation.
            max_new_tokens: Maximum tokens to generate.
            expand: Query expansion — None=auto (bm25 only), True/False=force.
            embedding_model: SentenceTransformer model for embeddings.
        """
        try:
            search_cfg = SearchConfig(
                index_path=index_path,
                k=k,
                retriever=RetrieverMethod(retriever),
                max_chunk_size=max_chunk_size,
                expand=expand,
                embedding_model=embedding_model,
            )
            gen_cfg = GenerationConfig(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )
            answer_query(query=query, search_cfg=search_cfg, gen_cfg=gen_cfg)
        except Exception as e:
            logger.error("answer failed: %s", e)
            sys.exit(1)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 128,
        max_chunk_size: int = 2000,
        batch_size: int = 4,
    ) -> None:
        """Generate answers for all entries in a search results file.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            save_directory: Directory where results with answers will be saved.
            model_name: HuggingFace model identifier for generation.
            max_new_tokens: Maximum tokens to generate.
            max_chunk_size: Maximum context length in characters.
            batch_size: Number of questions per GPU batch (opti 1.).
        """
        try:
            gen_cfg = GenerationConfig(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )
            answer_dataset(
                student_search_results_path=student_search_results_path,
                save_directory=save_directory,
                gen_cfg=gen_cfg,
                max_chunk_size=max_chunk_size,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.error("answer_dataset failed: %s", e)
            sys.exit(1)

    def answer_all(
        self,
        search_results_dir: str = "data/output/search_results",
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 128,
        max_chunk_size: int = 2000,
        batch_size: int = 4,
    ) -> None:
        """Answer every search results file in a directory, loading once.

        Convenience command for full local runs: the model singleton is
        loaded on the first dataset and reused for the rest, avoiding a
        second cold start.

        Args:
            search_results_dir: Directory of StudentSearchResults JSON files.
            save_directory: Directory where results with answers will be saved.
            model_name: HuggingFace model identifier for generation.
            max_new_tokens: Maximum tokens to generate.
            max_chunk_size: Maximum context length in characters.
            batch_size: Number of questions per GPU batch.
        """
        try:
            gen_cfg = GenerationConfig(
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )
            json_files = sorted(Path(search_results_dir).glob("*.json"))
            if not json_files:
                raise FileNotFoundError(
                    f"No JSON files found in '{search_results_dir}'."
                )
            for json_file in json_files:
                answer_dataset(
                    student_search_results_path=str(json_file),
                    save_directory=save_directory,
                    gen_cfg=gen_cfg,
                    max_chunk_size=max_chunk_size,
                    batch_size=batch_size,
                )
        except Exception as e:
            logger.error("answer_all failed: %s", e)
            sys.exit(1)

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 5,
        k_values: Optional[List[int]] = None,
        max_context_length: int = 2000,
        min_overlap: float = 0.05,
    ) -> None:
        """Evaluate search results against ground truth using recall@k.

        Args:
            student_answer_path: Path to the student search results JSON.
            dataset_path: Path to the AnsweredQuestions ground truth JSON.
            k: Primary k used during the search.
            k_values: k values for recall@k (default [1,3,5,10]).
            max_context_length: Maximum context length used during search.
            min_overlap: Minimum overlap ratio to count a source as found.
        """
        try:
            eval_cfg = EvaluateConfig(
                k_values=k_values if k_values is not None else [1, 3, 5],
                min_overlap=min_overlap,
            )
            evaluate_results(
                student_answer_path=student_answer_path,
                dataset_path=dataset_path,
                k=k,
                max_context_length=max_context_length,
                cfg=eval_cfg,
            )
        except Exception as e:
            logger.error("evaluate failed: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    fire.Fire(RAG)
