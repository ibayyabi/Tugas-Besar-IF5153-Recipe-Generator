from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from src.pipeline2.retriever import CandidateRetriever

DUMMY_QUERIES = [
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


def main() -> None:
    retriever = CandidateRetriever.from_artifacts()
    results = []
    for query in DUMMY_QUERIES:
        candidates = retriever.retrieve(query, top_n=5)
        results.append(
            {
                "dummy_structured_query": query,
                "top_candidates": [
                    {
                        "recipe_id": c["recipe_id"],
                        "title": c["title"],
                        "tfidf_score": c["tfidf_score"],
                        "total_calories": c["total_calories"],
                        "canonical_ingredients": c["canonical_ingredients"],
                    }
                    for c in candidates
                ],
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
