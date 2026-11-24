"""End-to-end queries through Pipeline 1 -> Pipeline 2 -> Pipeline 3.

These tests need the offline artifacts (corpus parquet, TF-IDF vectorizer and
matrix); they skip on a fresh checkout until the quickstart build has run.
Assertions stay on observable contract behaviour, not on a specific ranking.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.pipeline1.approaches import get_approach, trace_parse_query
from src.pipeline1.query_parser import parse_query
from src.pipeline2.approaches import TfidfApproach
from src.pipeline3.reranker import MultiCriteriaReranker

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    ROOT / "data" / "processed" / "recipes.parquet",
    ROOT / "artifacts" / "tfidf_vectorizer.pkl",
    ROOT / "artifacts" / "recipe_vectors.pkl",
    ROOT / "data" / "processed" / "pantry_staples.csv",
)
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED),
    reason="E2E artifacts missing; run the README quickstart build first",
)

EMPTY_QUERY = {
    "available_ingredients": [],
    "excluded_ingredients": [],
    "calorie_target": None,
    "calorie_min": None,
    "calorie_max": None,
}


@pytest.fixture(scope="module")
def pipeline():
    return TfidfApproach(), MultiCriteriaReranker()


def _run(pipeline, query: str, top_n: int = 50, top_k: int = 3):
    retriever, reranker = pipeline
    structured = parse_query(query)
    candidates = retriever.retrieve(structured, top_n=top_n)
    result = reranker.rerank(candidates, structured, top_k=top_k)
    return structured, candidates, result


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def test_normal_query_end_to_end(pipeline):
    structured, _, result = _run(
        pipeline,
        "I have chicken, potato and garlic. Around 500 calories, no cheese.",
    )
    assert structured["available_ingredients"] == ["chicken", "potato", "garlic"]
    assert structured["excluded_ingredients"] == ["cheese"]
    assert structured["calorie_target"] == 500
    assert len(result.recommendations) == 3
    assert not result.is_fallback
    assert all(not item.exclusion.violated for item in result.recommendations)
    assert any(
        "chicken" in item.coverage.available_matched for item in result.recommendations
    )


def test_no_cheese_query_never_returns_cheese(pipeline):
    _, _, result = _run(
        pipeline,
        "I have chicken, potato and garlic. Around 500 calories, no cheese.",
    )
    for item in result.recommendations:
        recipe = item.recipe
        assert "cheese" not in recipe.title.lower()
        assert not any("cheese" in ingredient.lower() for ingredient in recipe.ingredients)
        assert not any("cheese" in ingredient.lower() for ingredient in recipe.canonical_ingredients)


def test_top_k_has_no_duplicate_titles(pipeline):
    _, _, result = _run(
        pipeline,
        "I have chicken, potato and garlic. Around 500 calories, no cheese.",
    )
    titles = [_norm_title(item.recipe.title) for item in result.recommendations]
    assert len(titles) == len(set(titles))


def test_clause_boundary_keeps_available_ingredient(pipeline):
    structured, _, result = _run(
        pipeline,
        "I have chicken and rice. No cheese. I have potato.",
    )
    assert structured["available_ingredients"] == ["chicken", "rice", "potato"]
    assert structured["excluded_ingredients"] == ["cheese"]
    assert len(result.recommendations) == 3
    assert not result.is_fallback


def test_time_phrase_is_not_a_calorie_bound(pipeline):
    structured, _, result = _run(
        pipeline,
        "I have chicken and rice, cook under 20 minutes.",
    )
    assert structured["calorie_max"] is None
    assert result.filtered_out_calorie == 0
    assert not result.is_fallback


def test_count_bound_phrase_is_not_a_calorie_bound(pipeline):
    structured, _, result = _run(
        pipeline,
        "I have eggs, no more than 3 eggs",
    )
    assert structured["calorie_target"] is None
    assert structured["calorie_min"] is None
    assert structured["calorie_max"] is None
    assert result.filtered_out_calorie == 0
    assert not result.is_fallback


def test_unitless_count_e2e(pipeline):
    structured, _, result = _run(pipeline, "I have potatoes, under 2 potatoes")
    assert structured["calorie_max"] is None
    assert result.filtered_out_calorie == 0
    assert not result.is_fallback
    assert result.recommendations


def test_comma_list_exclusion_e2e(pipeline):
    structured, _, result = _run(pipeline, "I have chicken, no cheese, milk")
    assert structured["excluded_ingredients"] == ["cheese", "milk"]
    assert result.recommendations
    assert all(not item.exclusion.violated for item in result.recommendations)


def test_calorie_max_is_enforced_for_strict_results(pipeline):
    structured, _, result = _run(pipeline, "I have chicken, 500 calories max")
    assert structured["calorie_max"] == 500
    assert not result.is_fallback
    assert result.strict_match_count >= 3
    for item in result.recommendations:
        assert item.calorie.estimated_calories is not None
        assert item.calorie.estimated_calories <= 500
        assert item.calorie.passed_hard_filter


def test_exclusion_filter_holds_for_strict_results(pipeline):
    structured, _, result = _run(
        pipeline,
        "I have salmon and rice. No butter and no cream.",
    )
    assert structured["excluded_ingredients"] == ["butter", "cream"]
    assert not result.is_fallback
    assert result.filtered_out_exclusion > 0
    assert all(not item.exclusion.violated for item in result.recommendations)


def test_pure_calorie_query_without_pantry_is_no_match(pipeline):
    structured, _, result = _run(pipeline, "under 300 calories")
    assert structured["calorie_max"] == 300
    assert result.recommendations == []
    assert result.no_match_reason
    assert not result.is_fallback


def test_unknown_ingredients_return_no_match(pipeline):
    structured, candidates, result = _run(
        pipeline,
        "I have dragonfruit and starfruit. Around 500 calories.",
    )
    assert structured["available_ingredients"] == []
    assert candidates == []
    assert result.recommendations == []
    assert result.no_match_reason


def test_empty_query_returns_no_match(pipeline):
    structured, candidates, result = _run(pipeline, "")
    assert structured == EMPTY_QUERY
    assert candidates == []
    assert result.recommendations == []
    assert result.no_match_reason


def test_difflib_typo_recovery_end_to_end(pipeline):
    retriever, reranker = pipeline
    trace = trace_parse_query(
        "I have chiken and potatos.",
        get_approach("difflib-0.82"),
    )
    structured = trace.structured_query
    assert {"chicken", "potato"} <= set(structured["available_ingredients"])

    candidates = retriever.retrieve(structured, top_n=50)
    result = reranker.rerank(candidates, structured, top_k=3)
    assert len(result.recommendations) == 3
