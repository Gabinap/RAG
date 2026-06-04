"""Recall@k evaluation metrics for RAG retrieval quality."""

import json
from pathlib import Path
from typing import Dict, List

from constants import (
    BAR_WIDTH,
    BOLD,
    CODE_PALETTE,
    COLOR_ERROR,
    COLOR_SUCCESS,
    DIM,
    DOCS_PALETTE,
    GREEN,
    Palette,
    RED,
    YELLOW,
    colorize,
    divider,
)
from models import EvaluateConfig, MinimalSource


def _progress_bar(score: float) -> str:
    filled_blocks = round(score * BAR_WIDTH)
    empty_blocks = BAR_WIDTH - filled_blocks
    bar_color = GREEN if score >= 0.8 else YELLOW if score >= 0.5 else RED
    return (
        colorize("█" * filled_blocks, bar_color)
        + colorize("░" * empty_blocks, DIM)
    )


def _score_label(score: float) -> str:
    if score >= 0.80:
        return colorize("★  EXCELLENT", GREEN, BOLD)
    if score >= 0.60:
        return colorize("✓  GOOD     ", GREEN)
    if score >= 0.40:
        return colorize("~  OK       ", YELLOW)
    return colorize("✗  LOW      ", RED, BOLD)


def _display(
    total_questions: int,
    questions_with_correct_sources: int,
    questions_with_retrieved_sources: int,
    recall_by_k: Dict[int, float],
    search_k: int,
    palette: Palette,
) -> None:
    """Print a colourful evaluation report using the given palette."""
    b = palette.border
    t = palette.title
    s = palette.section
    v = palette.value

    div = divider(color=b)
    thin = divider(thin=True, color=s)

    print()
    print(div)
    print(colorize(
        f"  {palette.icon}  RAG EVALUATION — {palette.name}  {palette.icon}",
        t, BOLD,
    ))
    print(div)
    print()
    print(colorize("  Dataset", s, BOLD))
    print(thin)
    print(
        f"  {'Questions evaluated':<30}"
        + colorize(str(total_questions), v, BOLD)
    )
    print(
        f"  {'GT questions with sources':<30}"
        + colorize(str(questions_with_correct_sources), v, BOLD)
    )
    incomplete = questions_with_retrieved_sources < total_questions
    count_color = COLOR_ERROR if incomplete else COLOR_SUCCESS
    print(
        f"  {'Student questions searched':<30}"
        + colorize(str(questions_with_retrieved_sources), count_color, BOLD)
    )
    print(
        f"  {'Search k':<30}"
        + colorize(str(search_k), v, BOLD)
    )
    print()
    print(colorize("  Recall@k", s, BOLD))
    print(thin)
    for k_value, score in sorted(recall_by_k.items()):
        recall_label = f"Recall@{k_value}"
        score_pct = f"{score * 100:5.1f}%"
        print(
            f"  {colorize(recall_label, v, BOLD):<20}  "
            f"{_progress_bar(score)}  "
            f"{colorize(score_pct, v, BOLD)}  "
            f"{_score_label(score)}"
        )
    print()
    print(div)
    print()


def _overlap_chars(
    retrieved: MinimalSource,
    correct: MinimalSource,
) -> int:
    """Return the character intersection size between two sources.

    Args:
        retrieved: A retrieved source with file path and char range.
        correct: A ground-truth source with file path and char range.

    Returns:
        Number of overlapping characters, 0 if different files.
    """
    if retrieved.file_path != correct.file_path:
        return 0
    overlap_start = max(
        retrieved.first_character_index,
        correct.first_character_index,
    )
    overlap_end = min(
        retrieved.last_character_index,
        correct.last_character_index,
    )
    return max(0, overlap_end - overlap_start)


def _is_found(
    retrieved_sources: List[MinimalSource],
    correct_source: MinimalSource,
    min_overlap: float,
) -> bool:
    """Return True if any retrieved source covers enough of correct_source.

    Args:
        retrieved_sources: Top-k retrieved sources for the question.
        correct_source: One ground-truth source to check against.
        min_overlap: Minimum ratio of correct source length that must
            be covered by a retrieved source to count as found.

    Returns:
        True if at least one retrieved source meets the overlap threshold.
    """
    correct_source_length = (
        correct_source.last_character_index
        - correct_source.first_character_index
    )
    if correct_source_length <= 0:
        return False
    return any(
        _overlap_chars(candidate, correct_source) / correct_source_length
        >= min_overlap
        for candidate in retrieved_sources
    )


def _recall_at_k(
    retrieved_sources: List[MinimalSource],
    correct_sources: List[MinimalSource],
    k: int,
    min_overlap: float,
) -> float:
    """Compute recall@k for a single question.

    Args:
        retrieved_sources: All retrieved sources (ordered by rank).
        correct_sources: Ground-truth sources for the question.
        k: Number of top retrieved sources to consider.
        min_overlap: Minimum overlap ratio to count a source as found.

    Returns:
        Fraction of correct sources found in the top-k retrieved sources.
    """
    if not correct_sources:
        return 0.0
    top_k_sources = retrieved_sources[:k]
    found_count = sum(
        1 for correct_source in correct_sources
        if _is_found(top_k_sources, correct_source, min_overlap)
    )
    return found_count / len(correct_sources)


def evaluate_results(
    student_answer_path: str,
    dataset_path: str,
    k: int,
    max_context_length: int,
    cfg: EvaluateConfig,
) -> None:
    """Evaluate search results against ground truth using recall@k.

    Args:
        student_answer_path: Path to the StudentSearchResults JSON.
        dataset_path: Path to the AnsweredQuestions ground truth JSON.
        k: k used when running search_dataset (for validation display).
        max_context_length: Context length used during search (display only).
        cfg: Evaluation configuration (k_values, min_overlap).

    Raises:
        FileNotFoundError: If either input file does not exist.
        ValueError: If any requested k_value exceeds the search k.
    """
    student_results_path = Path(student_answer_path)
    ground_truth_path = Path(dataset_path)

    if not student_results_path.exists():
        raise FileNotFoundError(
            f"Student results not found: '{student_answer_path}'"
        )
    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth dataset not found: '{dataset_path}'"
        )

    invalid_k_values = [kv for kv in cfg.k_values if kv > k]
    if invalid_k_values:
        raise ValueError(
            f"k_values {invalid_k_values} exceed the search k={k}. "
            "Re-run search_dataset with a higher --k."
        )

    with student_results_path.open(encoding="utf-8") as f:
        student_results = json.load(f)
    with ground_truth_path.open(encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    correct_sources_by_id: Dict[str, List[MinimalSource]] = {
        question["question_id"]: [
            MinimalSource(**source)
            for source in question.get("sources", [])
        ]
        for question in ground_truth_data["rag_questions"]
    }
    retrieved_sources_by_id: Dict[str, List[MinimalSource]] = {
        result["question_id"]: [
            MinimalSource(**source)
            for source in result["retrieved_sources"]
        ]
        for result in student_results["search_results"]
    }

    evaluated_ids = set(retrieved_sources_by_id.keys())

    total_questions = len(evaluated_ids)
    questions_with_correct_sources = sum(
        1 for qid in evaluated_ids
        if correct_sources_by_id.get(qid)
    )
    questions_with_retrieved_sources = sum(
        1 for sources in retrieved_sources_by_id.values() if sources
    )

    per_question_scores: Dict[int, List[float]] = {
        k_value: [] for k_value in cfg.k_values
    }
    for question_id in evaluated_ids:
        correct_sources = correct_sources_by_id.get(question_id, [])
        retrieved_sources = retrieved_sources_by_id[question_id]
        for k_value in cfg.k_values:
            per_question_scores[k_value].append(
                _recall_at_k(
                    retrieved_sources,
                    correct_sources,
                    k_value,
                    cfg.min_overlap,
                )
            )

    recall_by_k: Dict[int, float] = {
        k_value: sum(scores) / len(scores) if scores else 0.0
        for k_value, scores in per_question_scores.items()
    }

    palette = (
        CODE_PALETTE if "code" in Path(dataset_path).stem else DOCS_PALETTE
    )
    _display(
        total_questions,
        questions_with_correct_sources,
        questions_with_retrieved_sources,
        recall_by_k,
        k,
        palette,
    )
