"""Prompt construction and context assembly for generation."""

from typing import Dict, List

from models import MinimalSource

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the vLLM "
    "codebase. Answer based only on the provided context. "
    "Be concise, accurate, and self-contained."
)


def build_context(
    sources: List[MinimalSource],
    max_chars: int,
) -> str:
    """Concatenate source texts into a context string capped at max_chars.

    Sources are included in order until max_chars would be exceeded. The last
    fitting source is truncated rather than dropped entirely.

    Args:
        sources: Retrieved sources, each with an optional text field.
        max_chars: Maximum total character budget for the context.

    Returns:
        A formatted context string ready to embed in a prompt.
    """
    parts: List[str] = []
    chars_used = 0

    for i, source in enumerate(sources, 1):
        if not source.text:
            continue
        header = f"[Source {i}: {source.file_path}]\n"
        body = source.text.strip()
        block = f"{header}{body}\n\n"

        remaining_budget = max_chars - chars_used
        if len(block) <= remaining_budget:
            parts.append(block)
            chars_used += len(block)
        else:
            truncated_body_len = remaining_budget - len(header) - 2
            if truncated_body_len > 0:
                parts.append(header + body[:truncated_body_len] + "\n\n")
            break

    return "".join(parts).strip()


def build_messages(
    question: str,
    context: str,
) -> List[Dict[str, str]]:
    """Build the chat message list for the model's chat template.

    Args:
        question: The question to answer.
        context: The assembled context string from retrieved sources.

    Returns:
        List of role/content dicts compatible with apply_chat_template.
    """
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
