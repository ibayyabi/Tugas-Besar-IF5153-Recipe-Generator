"""Pipeline 1 — pluggable ingredient-matching approaches.

Every approach answers the same contract so the query parser, the experiment
harness, and the Streamlit UI can swap them freely:

* ``extract(text, vocab) -> list[str]`` — canonical ingredients, first-mention
  order (drop-in for ``parse_query``'s ``matcher`` argument);
* ``trace(text, vocab) -> ApproachTrace`` — the same result plus the
  intermediate evidence (matched spans, provenance, scores) the UI renders.

The matching context is the *masked* query text produced by
``query_parser.parse_query``: calorie and exclusion spans are already blanked,
so approaches never see constraint clauses.
"""

from __future__ import annotations

from pathlib import Path

from .base import ApproachTrace, IngredientApproach, IngredientMatch
from .fuzzy import FuzzyApproach
from .ner import NerApproach
from .rule import RuleApproach, SurfaceExactApproach
from .runner import trace_parse_query

__all__ = [
    "ApproachTrace",
    "FuzzyApproach",
    "IngredientApproach",
    "IngredientMatch",
    "NerApproach",
    "RuleApproach",
    "SurfaceExactApproach",
    "available_approaches",
    "get_approach",
    "trace_parse_query",
]

_NER_DEFAULT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "ner_model"


def _approach_factories() -> dict[str, type[IngredientApproach]]:
    return {
        RuleApproach.name: RuleApproach,
        SurfaceExactApproach.name: SurfaceExactApproach,
        FuzzyApproach.difflib_name: FuzzyApproach,
        FuzzyApproach.ngram_name: FuzzyApproach,
        NerApproach.name: NerApproach,
    }


def available_approaches() -> list[dict[str, object]]:
    """Registry metadata for UI pickers; ``available`` flags artifact-dependent models."""
    ner_ready = (_NER_DEFAULT_DIR / "config.json").exists()
    return [
        {"name": RuleApproach.name, "description": RuleApproach.description, "available": True},
        {"name": SurfaceExactApproach.name, "description": SurfaceExactApproach.description, "available": True},
        {"name": FuzzyApproach.difflib_name, "description": FuzzyApproach.difflib_description, "available": True},
        {"name": FuzzyApproach.ngram_name, "description": FuzzyApproach.ngram_description, "available": True},
        {"name": NerApproach.name, "description": NerApproach.description, "available": ner_ready},
    ]


def get_approach(name: str) -> IngredientApproach:
    factories = _approach_factories()
    if name not in factories:
        raise KeyError(f"unknown Pipeline 1 approach: {name!r}; known: {sorted(factories)}")
    if name == NerApproach.name:
        return NerApproach()
    if name in (FuzzyApproach.difflib_name, FuzzyApproach.ngram_name):
        return FuzzyApproach.for_name(name)
    return factories[name]()
