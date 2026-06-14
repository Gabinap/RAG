"""Lazy singleton loader for the LLM model and tokenizer."""

from typing import Any, Optional, Tuple

from constants import BLUE, BOLD, COLOR_SUCCESS, colorize, divider

_model: Optional[Any] = None
_tokenizer: Optional[Any] = None
_loaded_model_name: Optional[str] = None


def get_model_and_tokenizer(model_name: str) -> Tuple[Any, Any]:
    """Load and cache the model and tokenizer (lazy singleton).

    On first call the model is downloaded/loaded from disk. Subsequent calls
    with the same model_name return the cached instance immediately.

    Args:
        model_name: HuggingFace model identifier (e.g. 'Qwen/Qwen3-0.6B').

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    global _model, _tokenizer, _loaded_model_name

    if _model is not None and _loaded_model_name == model_name:
        return _model, _tokenizer

    # Heavy imports are deferred here so commands that never load the LLM
    # (index, search, evaluate) don't pay the torch/transformers import cost.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print()
    print(divider())
    print(colorize(f"  🤖  Loading model: {model_name}", BLUE, BOLD))
    print(divider(thin=True))

    print(f"  {'Device':<20}" + colorize("cpu", BOLD))
    print(f"  {'Dtype':<20}" + colorize(str(torch.float32), BOLD))

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="cpu",
    )
    _model.eval()
    _model = torch.compile(_model)
    _loaded_model_name = model_name

    print(colorize("  ✓  Model ready", COLOR_SUCCESS, BOLD))
    print(divider())
    print()

    return _model, _tokenizer
