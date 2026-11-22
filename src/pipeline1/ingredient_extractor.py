"""Pipeline 1 — ingredient mention extraction over the closed vocabulary."""

from __future__ import annotations

import re

from src.preprocessing.ingredient_normalizer import Vocabulary, normalize_ingredient, singularize

TOKEN = re.compile(r"[a-z0-9]+")


def match_ingredients(text: str, vocab: Vocabulary) -> list[tuple[int, int, str]]:
    """Longest-match scan; returns ``(start, end, canonical)`` per hit in text order."""
    tokens = [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]
    hits: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        remaining = len(tokens) - index
        for size in range(min(vocab.max_tokens, remaining), 0, -1):
            window = [token for token, _, _ in tokens[index : index + size]]
            window[-1] = singularize(window[-1])
            canonical = vocab.lookup(" ".join(window))
            if canonical:
                hits.append((tokens[index][1], tokens[index + size - 1][2], canonical))
                index += size
                break
        else:
            token, start, end = tokens[index]
            joined = _split_compound(token, vocab)
            if joined:
                hits.append((start, end, joined))
            index += 1
    return hits


def _split_compound(token: str, vocab: Vocabulary) -> str | None:
    """``sweetpotato`` -> ``sweet potato``: a joined spelling the alias table
    does not spell out. Split only when both halves are vocabulary words."""
    folded = singularize(token)
    for cut in range(1, len(folded)):
        canonical = vocab.lookup(f"{folded[:cut]} {singularize(folded[cut:])}")
        if canonical:
            return canonical
    return None


def extract_ingredients(text: str, vocab: Vocabulary) -> list[str]:
    """Canonical ingredients mentioned in ``text``, first mention order, unique."""
    seen: dict[str, None] = {}
    for _, _, canonical in match_ingredients(text, vocab):
        seen.setdefault(canonical, None)
    return list(seen)


def canonicalize(name: str, vocab: Vocabulary) -> str | None:
    return vocab.canonical(normalize_ingredient(name))
