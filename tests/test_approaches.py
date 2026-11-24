"""Approach registry, trace runner, and retrieval approach guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline1.approaches import available_approaches, get_approach, trace_parse_query
from src.pipeline1.query_parser import load_vocabulary, parse_query

ROOT = Path(__file__).resolve().parents[1]
P2_ARTIFACTS_READY = all(
    (ROOT / path).exists()
    for path in (
        "data/processed/recipes.parquet",
        "artifacts/tfidf_vectorizer.pkl",
        "artifacts/recipe_vectors.pkl",
    )
)


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


def test_registry_lists_all_lexical_approaches():
    names = [entry["name"] for entry in available_approaches()]
    assert names[:4] == ["rule", "surface-exact", "difflib-0.82", "char-ngram-0.60"]
    # ner is registry-listed regardless of artifact presence (flagged via `available`)
    assert "ner" in names


def test_rule_approach_matches_parse_query_default(vocab):
    approach = get_approach("rule")
    for query in (
        "I have chicken, potato, carrot, onion and garlic. Around 500 calories and no cheese.",
        "no milk or butter, just beef",
        "sweetpotato and green onions",
    ):
        assert parse_query(query, vocab, approach.extract) == parse_query(query, vocab)


def test_trace_parse_query_records_spans(vocab):
    trace = trace_parse_query("I have eggs. Around 500 calories, no cheese.", get_approach("rule"), vocab)
    assert trace.structured_query == parse_query(
        "I have eggs. Around 500 calories, no cheese.", vocab
    )
    assert trace.calorie_spans, "calorie span must be recorded"
    assert trace.exclusion_spans, "exclusion span must be recorded"
    assert [m.canonical for m in trace.available_trace.matches] == ["egg"]
    assert trace.excluded_trace.ingredients == ["cheese"]


def test_fuzzy_approach_rescues_typos(vocab):
    approach = get_approach("difflib-0.82")
    trace = trace_parse_query("I have chiken and tommato.", approach, vocab)
    assert trace.structured_query["available_ingredients"] == ["chicken", "tomato"]
    methods = {m.method for m in trace.available_trace.matches}
    assert "fuzzy-difflib" in methods


def test_ner_approach_requires_artifacts(tmp_path):
    from src.pipeline1.approaches import NerApproach

    missing = NerApproach(model_dir=tmp_path / "nope")
    if not missing.available:
        with pytest.raises(FileNotFoundError):
            missing._load()
    else:  # artifacts present in this checkout — loading must succeed
        missing._load()


@pytest.mark.skipif(
    not P2_ARTIFACTS_READY,
    reason="P2 artifacts missing; run the README quickstart build first",
)
def test_p2_approaches_share_candidate_schema():
    from src.pipeline2.approaches import TfidfApproach

    approach = TfidfApproach()
    candidates = approach.retrieve(
        {"available_ingredients": ["chicken", "potato"], "excluded_ingredients": []}, top_n=5
    )
    required = {
        "recipe_id",
        "title",
        "canonical_ingredients",
        "tfidf_score",
        "total_calories",
        "calories_per_serving",
        "calorie_coverage",
    }
    assert candidates and required.issubset(candidates[0].keys())
    # harness compatibility (evaluate_retriever reads these directly)
    assert approach.recipe_ids is not None
    indices, scores = approach.retrieve_indices(
        {"available_ingredients": ["chicken"], "excluded_ingredients": []}, top_n=3
    )
    assert len(indices) == len(scores) == 3
