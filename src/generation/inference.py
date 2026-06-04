"""Answer generation using the LLM."""

import json
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from constants import (
    BOLD, COLOR_ACCENT, COLOR_SUCCESS, colorize, divider
)
from generation.model import get_model_and_tokenizer
from generation.prompt import build_context, build_messages
from indexing import retrieve
from models import (
    AnswerMeta,
    GenerationConfig,
    MinimalAnswer,
    MinimalSource,
    SearchConfig,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


def _build_prompt(
    question: str,
    sources: List[MinimalSource],
    tokenizer: Any,
    max_chunk_size: int,
) -> str:
    """Assemble context + question into a chat-formatted prompt string."""
    context = build_context(sources, max_chunk_size)
    messages = build_messages(question, context)
    try:
        return str(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
    except TypeError:
        return str(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ))


def _generate(
    question: str,
    sources: List[MinimalSource],
    model: Any,
    tokenizer: Any,
    gen_cfg: GenerationConfig,
    max_chunk_size: int,
) -> str:
    """Run one forward pass and return the answer as a plain string.

    Args:
        question: The question to answer.
        sources: Retrieved sources with text.
        model: Loaded causal LM.
        tokenizer: Matching tokenizer.
        gen_cfg: Generation hyper-parameters.
        max_chunk_size: Total character budget for the context.

    Returns:
        The model's answer stripped of special tokens and whitespace.
    """
    import torch

    prompt = _build_prompt(question, sources, tokenizer, max_chunk_size)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


def _generate_batch(
    questions: List[str],
    sources_list: List[List[MinimalSource]],
    model: Any,
    tokenizer: Any,
    gen_cfg: GenerationConfig,
    max_chunk_size: int,
) -> List[str]:
    """Run one forward pass for a batch of questions and return answers.

    # opti 1. batch: N questions tokenised and generated in a single GPU call.
    #   The GPU processes the whole batch in parallel instead of one at a time,
    #   maximising CUDA core utilisation. Expected speedup: ×3 to ×5 on GPU.
    #   Left padding required for decoder models: generation starts from the
    #   last token, so padding must not appear there.

    Args:
        questions: List of questions to answer.
        sources_list: Corresponding retrieved sources for each question.
        model: Loaded causal LM.
        tokenizer: Matching tokenizer.
        gen_cfg: Generation hyper-parameters.
        max_chunk_size: Total character budget for the context.

    Returns:
        List of answer strings in the same order as questions.
    """
    import torch

    prompts = [
        _build_prompt(q, s, tokenizer, max_chunk_size)
        for q, s in zip(questions, sources_list)
    ]

    # Left padding: decoder models generate from the last token position.
    # Right padding would cause generation to start from a padding position,
    # corrupting the output.
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # All input sequences share the same padded length.
    # Slice from that length to extract only the newly generated tokens.
    input_len = inputs["input_ids"].shape[1]
    return [
        str(tokenizer.decode(
            output_ids[i][input_len:], skip_special_tokens=True
        )).strip()
        for i in range(len(prompts))
    ]


def answer_query(
    query: str,
    search_cfg: SearchConfig,
    gen_cfg: GenerationConfig,
) -> None:
    """Answer a single question using retrieved context and the LLM.

    Args:
        query: The question to answer.
        search_cfg: Search configuration (index path, k, retriever).
        gen_cfg: Generation configuration (model name, max new tokens).
    """
    sources = retrieve(query, search_cfg)
    model, tokenizer = get_model_and_tokenizer(gen_cfg.model_name)

    answer_text = _generate(
        query, sources, model, tokenizer, gen_cfg, search_cfg.max_chunk_size
    )
    result = MinimalAnswer(
        question_id="",
        question=query,
        retrieved_sources=sources,
        answer=answer_text,
    )
    print(result.model_dump_json(indent=2))


def _answer_cache_is_valid(
    out_file: Path,
    meta: AnswerMeta,
    question_ids: List[str],
) -> bool:
    """Return True if an existing answer output can be reused as-is.

    Reuse requires identical generation params, identical source search
    metadata, and identical question ids.
    """
    if not out_file.exists():
        return False
    try:
        with out_file.open(encoding="utf-8") as f:
            existing = StudentSearchResultsAndAnswer(**json.load(f))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return False
    if existing.meta != meta:
        return False
    existing_ids = [r.question_id for r in existing.search_results]
    return existing_ids == question_ids


def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
    gen_cfg: GenerationConfig,
    max_chunk_size: int,
    batch_size: int = 4,
) -> None:
    """Generate answers for all questions in a search results file.

    The model is loaded once before the loop. Questions are processed in
    batches of batch_size for efficient GPU utilisation (opti 1.).

    Args:
        student_search_results_path: Path to StudentSearchResults JSON.
        save_directory: Directory where the output JSON will be written.
        gen_cfg: Generation configuration (model name, max new tokens).
        max_chunk_size: Total character budget per context.
        batch_size: Number of questions per GPU batch (default 4).

    Raises:
        FileNotFoundError: If the search results file does not exist.
    """
    results_path = Path(student_search_results_path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Search results not found: '{student_search_results_path}'"
        )

    with results_path.open(encoding="utf-8") as f:
        search_results = StudentSearchResults(**json.load(f))

    meta = AnswerMeta(
        model_name=gen_cfg.model_name,
        max_new_tokens=gen_cfg.max_new_tokens,
        max_chunk_size=max_chunk_size,
        source_meta=search_results.meta,
    )
    question_ids = [r.question_id for r in search_results.search_results]
    out_file = Path(save_directory) / results_path.name

    # Cache: skip generation (and model loading) if a matching output exists.
    if _answer_cache_is_valid(out_file, meta, question_ids):
        print(colorize(
            f"  ✓  Cache hit — reusing {out_file}", COLOR_SUCCESS, BOLD
        ))
        return

    model, tokenizer = get_model_and_tokenizer(gen_cfg.model_name)

    all_results = search_results.search_results
    answers: List[MinimalAnswer] = []

    for i in tqdm(
        range(0, len(all_results), batch_size),
        desc="Generating",
        unit="batch",
    ):
        batch = all_results[i: i + batch_size]
        questions = [r.question for r in batch]
        sources_list = [r.retrieved_sources for r in batch]

        batch_answers = _generate_batch(
            questions, sources_list, model, tokenizer, gen_cfg, max_chunk_size
        )

        for result, answer_text in zip(batch, batch_answers):
            answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=answer_text,
            ))

    output = StudentSearchResultsAndAnswer(
        meta=meta,
        search_results=answers,
        k=search_results.k,
    )

    Path(save_directory).mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    total = len(answers)
    print()
    print(divider(thin=True))
    print(
        colorize("  ✓  Generation complete!", COLOR_SUCCESS, BOLD)
        + f"  {colorize(str(total), BOLD)} answers"
        + f"  →  {colorize(str(out_file), COLOR_ACCENT)}"
    )
    print(divider())
    print()
