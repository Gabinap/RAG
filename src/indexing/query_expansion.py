"""Query expansion via WordNet synonyms (lexical, BM25-only)."""

import re
from math import ceil
from typing import List, Optional

_MIN_WORD_LEN = 4
_wordnet_ready = False


def _ensure_wordnet() -> None:
    """Lazily download the WordNet corpus on first use."""
    global _wordnet_ready
    if _wordnet_ready:
        return
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    _wordnet_ready = True


def _first_synonym(word: str) -> Optional[str]:
    """Return the first WordNet lemma distinct from word, or None.

    Args:
        word: A lowercase alphabetic word to look up.

    Returns:
        The first synonym (spaces instead of underscores), or None if the
        word is unknown to WordNet or has no distinct synonym.
    """
    from nltk.corpus import wordnet

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            candidate = str(lemma.name()).replace("_", " ")
            if candidate.lower() != word.lower():
                return candidate
    return None


def expand_query(query: str, max_ratio: float = 0.25) -> List[str]:
    """Generate synonym variants of a query for BM25 retrieval.

    The original query is always returned first as the anchor. Variants are
    built deterministically: the longest words are tried first, and each
    eligible word (known to WordNet) yields one variant where that single
    word is replaced by its first synonym.

    Args:
        query: The original search query.
        max_ratio: Maximum fraction of words to turn into variants.

    Returns:
        A list starting with the original query, followed by up to
        ceil(n_words * max_ratio) single-word synonym variants.
    """
    words = query.split()
    if not words:
        return [query]

    _ensure_wordnet()

    # Longest words first: a cheap proxy for content words over stopwords.
    order = sorted(
        range(len(words)), key=lambda i: len(words[i]), reverse=True
    )
    max_variants = max(1, ceil(len(words) * max_ratio))

    variants = [query]
    for i in order:
        if len(variants) - 1 >= max_variants:
            break
        clean = re.sub(r"[^A-Za-z]", "", words[i])
        if len(clean) < _MIN_WORD_LEN:
            continue
        synonym = _first_synonym(clean.lower())
        if synonym is None:
            continue
        variant_words = list(words)
        variant_words[i] = synonym
        variants.append(" ".join(variant_words))

    return variants
