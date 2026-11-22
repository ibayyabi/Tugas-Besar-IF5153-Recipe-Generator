from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.evaluate_retrieval import evaluate_retriever, generate_in_memory_queries
from src.pipeline2.retriever import CandidateRetriever
from src.pipeline2.vectorizer import RecipeVectorizer, fit_corpus_tfidf
from src.preprocessing.build_corpus import build_processed_corpus

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "samples" / "recipes_sample.csv"

DUMMY_STRUCTURED_QUERIES = [
    {
        "available_ingredients": ["chicken", "potato", "carrot", "onion", "garlic"],
        "excluded_ingredients": ["cheese"],
        "calorie_target": 500,
        "calorie_min": None,
        "calorie_max": None,
    },
    {
        "available_ingredients": ["egg", "milk", "butter", "flour", "sugar"],
        "excluded_ingredients": [],
        "calorie_target": None,
        "calorie_min": None,
        "calorie_max": 600,
    },
    {
        "available_ingredients": ["ground beef", "tomato", "onion", "garlic", "pasta"],
        "excluded_ingredients": ["milk"],
        "calorie_target": None,
        "calorie_min": 400,
        "calorie_max": 700,
    },
]


def _make_dummy_corpus() -> pd.DataFrame:
    rows = [
        {
            "recipe_id": 0,
            "title": "Chicken Potato Soup",
            "ingredients": ["200 g chicken", "2 potatoes", "1 carrot", "1 onion", "2 cloves garlic", "1 stalk celery"],
            "canonical_ingredients": ["chicken", "potato", "carrot", "onion", "garlic", "celery"],
            "canonical_text": "chicken potato carrot onion garlic celery",
            "directions": ["Boil all ingredients until tender."],
            "total_calories": 482.0,
            "calorie_coverage": 1.0,
        },
        {
            "recipe_id": 1,
            "title": "Basic Pancakes",
            "ingredients": ["1 cup flour", "1 cup milk", "1 egg", "2 tbsp butter", "2 tbsp sugar", "1 tsp salt"],
            "canonical_ingredients": ["flour", "milk", "egg", "butter", "sugar", "salt"],
            "canonical_text": "flour milk egg butter sugar salt",
            "directions": ["Mix batter and cook on griddle."],
            "total_calories": 520.0,
            "calorie_coverage": 1.0,
        },
        {
            "recipe_id": 2,
            "title": "Beef Tomato Pasta",
            "ingredients": ["250 g ground beef", "2 tomatoes", "1 onion", "2 cloves garlic", "200 g pasta"],
            "canonical_ingredients": ["ground beef", "tomato", "onion", "garlic", "pasta"],
            "canonical_text": "ground beef tomato onion garlic pasta",
            "directions": ["Brown beef, simmer with tomato, toss with pasta."],
            "total_calories": 640.0,
            "calorie_coverage": 1.0,
        },
        {
            "recipe_id": 3,
            "title": "Salted Water Broth",
            "ingredients": ["4 cups water", "2 tsp salt", "1 tsp pepper"],
            "canonical_ingredients": ["water", "salt", "pepper"],
            "canonical_text": "water salt pepper",
            "directions": ["Boil water with salt and pepper."],
            "total_calories": 0.0,
            "calorie_coverage": 1.0,
        },
    ]
    return pd.DataFrame(rows)


def test_dummy_query_vectorization_and_retrieval() -> None:
    corpus = _make_dummy_corpus()
    vectorizer = RecipeVectorizer(max_features=1000, ngram_range=(1, 2), use_staple_downweight=True)
    recipe_vectors = vectorizer.fit_transform(corpus["canonical_text"].tolist())
    retriever = CandidateRetriever(vectorizer=vectorizer, recipe_vectors=recipe_vectors, corpus=corpus)

    q_vec = vectorizer.transform_query(DUMMY_STRUCTURED_QUERIES[0])
    assert q_vec.shape == (1, recipe_vectors.shape[1])
    assert q_vec.nnz > 0

    candidates_soup = retriever.retrieve(DUMMY_STRUCTURED_QUERIES[0], top_n=3)
    assert len(candidates_soup) == 3
    assert candidates_soup[0]["title"] == "Chicken Potato Soup"
    assert candidates_soup[0]["tfidf_score"] >= candidates_soup[1]["tfidf_score"]

    candidates_pancake = retriever.retrieve(DUMMY_STRUCTURED_QUERIES[1], top_n=3)
    assert candidates_pancake[0]["title"] == "Basic Pancakes"

    candidates_pasta = retriever.retrieve(DUMMY_STRUCTURED_QUERIES[2], top_n=3)
    assert candidates_pasta[0]["title"] == "Beef Tomato Pasta"


def test_staple_downweight_reduces_staple_dominance() -> None:
    corpus = _make_dummy_corpus()
    vec_weighted = RecipeVectorizer(max_features=1000, ngram_range=(1, 2), use_staple_downweight=True)
    mat_weighted = vec_weighted.fit_transform(corpus["canonical_text"].tolist())
    retriever = CandidateRetriever(vectorizer=vec_weighted, recipe_vectors=mat_weighted, corpus=corpus)

    dummy_query = {
        "available_ingredients": ["flour", "egg", "salt", "water", "pepper"],
        "excluded_ingredients": [],
        "calorie_target": None,
        "calorie_min": None,
        "calorie_max": None,
    }
    candidates = retriever.retrieve(dummy_query, top_n=2)
    assert candidates[0]["title"] == "Basic Pancakes"


def test_pipeline2_end_to_end_with_artifacts_and_evaluation(tmp_path: Path) -> None:
    corpus_parquet = tmp_path / "recipes.parquet"
    vocab_pkl = tmp_path / "ingredient_vocabulary.pkl"
    vec_pkl = tmp_path / "tfidf_vectorizer.pkl"
    mat_pkl = tmp_path / "recipe_vectors.pkl"

    build_processed_corpus(
        corpus_path=SAMPLE_CSV,
        output_path=corpus_parquet,
        vocab_path=vocab_pkl,
        limit=100,
    )
    fit_corpus_tfidf(
        corpus_parquet=corpus_parquet,
        vectorizer_path=vec_pkl,
        vectors_path=mat_pkl,
        max_features=1000,
        use_staple_downweight=True,
    )

    retriever = CandidateRetriever.from_artifacts(
        corpus_path=corpus_parquet,
        vectorizer_path=vec_pkl,
        vectors_path=mat_pkl,
    )
    candidates = retriever.retrieve(DUMMY_STRUCTURED_QUERIES[1], top_n=10)
    assert len(candidates) == 10
    assert all("tfidf_score" in c and "canonical_ingredients" in c for c in candidates)

    queries = generate_in_memory_queries(retriever.corpus, num_queries=20, seed=42)
    metrics = evaluate_retriever(retriever, queries, ks=(1, 5, 10, 50))
    assert metrics["evaluated_queries"] == 20.0
    assert 0.0 <= metrics["recall@50"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0
