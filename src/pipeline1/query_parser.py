"""Pipeline 1 — Query & Constraint Understanding (PRD §12).

    raw query -> normalize -> exclusion/calorie constraints -> ingredient scan
              -> Structured Query

The output matches the PRD contract exactly:

    {
      "available_ingredients": ["chicken", ...],
      "excluded_ingredients": ["cheese", ...],
      "calorie_target": 500 | None,
      "calorie_min": None,
      "calorie_max": None
    }

An ingredient named in an exclusion clause is never reported as available,
even when it is also listed earlier ("I have butter... without butter").
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.pipeline1.constraint_extractor import (
    exclusion_spans,
    extract_calorie_constraints,
    normalize_query_text,
)
from src.pipeline1.ingredient_extractor import extract_ingredients
from src.preprocessing.ingredient_normalizer import Vocabulary


@lru_cache(maxsize=1)
def load_vocabulary() -> Vocabulary:
    return Vocabulary.load()


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


@dataclass
class ConstrainedQuery:
    """Normalized query with calorie/exclusion spans masked out of the contexts.

    The single composition shared by ``parse_query`` (production) and
    ``runner.trace_parse_query`` (UI evidence), so the two cannot drift.
    """

    text: str
    calories: dict[str, int | None]
    calorie_spans: list[tuple[int, int]]
    exclusion_spans: list[tuple[int, int]]
    available_context: str  # exclusion clauses blanked; input to the ingredient scan
    exclusion_context: str  # only the exclusion clause text


def constrain_query(query: str, vocab: Vocabulary | None = None) -> ConstrainedQuery:
    """Normalize a query and split it into masked contexts for ingredient matching."""
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    text, boundaries = normalize_query_text(query)
    calories, calorie_spans = extract_calorie_constraints(text, vocab)
    masked = _blank(text, calorie_spans)
    masked_boundaries = _blank(boundaries, calorie_spans)

    spans = exclusion_spans(masked, masked_boundaries, vocab)
    return ConstrainedQuery(
        text=text,
        calories=calories,
        calorie_spans=calorie_spans,
        exclusion_spans=spans,
        available_context=_blank(masked, spans),
        exclusion_context=" ".join(masked[start:end] for start, end in spans),
    )


def parse_query(
    query: str, vocab: Vocabulary | None = None, matcher=extract_ingredients
) -> dict:
    """Turn a natural-language pantry request into the structured query.

    ``matcher`` is the ingredient scan, ``(text, vocab) -> [canonical, ...]``;
    experiments swap it in without touching the constraint logic.
    """
    vocab = vocab or load_vocabulary()
    constrained = constrain_query(query, vocab)

    available = matcher(constrained.available_context, vocab)
    excluded = matcher(constrained.exclusion_context, vocab)

    return {
        "available_ingredients": [item for item in available if item not in excluded],
        "excluded_ingredients": excluded,
        **constrained.calories,
    }
