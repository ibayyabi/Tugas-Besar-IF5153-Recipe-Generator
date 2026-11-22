"""Trace-aware Pipeline 1 runner: same composition as ``query_parser.parse_query``,
recording every intermediate step for the UI."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline1.approaches.base import ApproachTrace, IngredientApproach
from src.pipeline1.query_parser import constrain_query, load_vocabulary
from src.preprocessing.ingredient_normalizer import Vocabulary


@dataclass
class Pipeline1Trace:
    """Everything the UI needs to replay one query through Pipeline 1."""

    approach: str
    raw_query: str
    normalized_text: str
    calorie_constraints: dict[str, int | None]
    calorie_spans: list[tuple[int, int]]
    exclusion_spans: list[tuple[int, int]]
    available_context: str  # masked text handed to the ingredient matcher
    exclusion_context: str
    available_trace: ApproachTrace
    excluded_trace: ApproachTrace
    structured_query: dict[str, object] = field(default_factory=dict)


def trace_parse_query(
    query: str,
    approach: IngredientApproach,
    vocab: Vocabulary | None = None,
) -> Pipeline1Trace:
    """Run Pipeline 1 with the selected approach and capture per-step evidence.

    Uses the same ``constrain_query`` composition as ``query_parser.parse_query``;
    only the ingredient scan differs (approach traces instead of bare results).
    """
    vocab = vocab or load_vocabulary()
    constrained = constrain_query(query)

    available_trace = approach.trace(constrained.available_context, vocab)
    excluded_trace = approach.trace(constrained.exclusion_context, vocab)

    available = available_trace.ingredients
    excluded = excluded_trace.ingredients
    structured = {
        "available_ingredients": [item for item in available if item not in excluded],
        "excluded_ingredients": excluded,
        **constrained.calories,
    }
    return Pipeline1Trace(
        approach=approach.name,
        raw_query=query,
        normalized_text=constrained.text,
        calorie_constraints=constrained.calories,
        calorie_spans=constrained.calorie_spans,
        exclusion_spans=constrained.exclusion_spans,
        available_context=constrained.available_context,
        exclusion_context=constrained.exclusion_context,
        available_trace=available_trace,
        excluded_trace=excluded_trace,
        structured_query=structured,
    )
