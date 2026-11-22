from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from src.pipeline2.retriever import CandidateRetriever
from src.pipeline2.vectorizer import (
    DEFAULT_CORPUS_PARQUET,
    DEFAULT_VECTORIZER_PATH,
    DEFAULT_VECTORS_PATH,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES_PATH = ROOT / "data" / "test" / "synthetic_retrieval_queries.json"


def generate_in_memory_queries(
    corpus: pd.DataFrame,
    num_queries: int = 200,
    pantry_fraction: float = 0.6,
    min_ingredients: int = 4,
    seed: int = 42,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    eligible = [
        (int(row["recipe_id"]), str(row["title"]), list(row["canonical_ingredients"]))
        for _, row in corpus.iterrows()
        if len(row["canonical_ingredients"]) >= min_ingredients
    ]
    if not eligible:
        return []
    sampled = rng.sample(eligible, min(num_queries, len(eligible)))
    queries: list[dict[str, object]] = []
    for idx, (recipe_id, title, ingredients) in enumerate(sampled, start=1):
        count = max(3, round(len(ingredients) * pantry_fraction))
        pantry = rng.sample(ingredients, min(count, len(ingredients) - 1))
        held_out = [item for item in ingredients if item not in pantry]
        queries.append(
            {
                "id": f"sim{idx:04d}",
                "recipe_id": recipe_id,
                "title": title,
                "pantry": pantry,
                "held_out": held_out,
                "required": ingredients,
            }
        )
    return queries


def rank_target(retriever: CandidateRetriever, item: dict[str, object], max_k: int) -> int | None:
    """1-based rank of the target recipe in the retriever's top-``max_k``, or ``None``.

    Shared by ``evaluate_retriever`` and the Pipeline 2 experiment harness so the
    retrieval loop is written once.
    """
    structured_query = {
        "available_ingredients": list(item["pantry"]),
        "excluded_ingredients": [],
        "calorie_target": None,
        "calorie_min": None,
        "calorie_max": None,
    }
    indices, _ = retriever.retrieve_indices(structured_query, top_n=max_k)
    for pos, idx in enumerate(indices, start=1):
        if int(retriever.recipe_ids[idx]) == int(item["recipe_id"]):
            return pos
    return None


def evaluate_retriever(
    retriever: CandidateRetriever,
    queries: list[dict[str, object]],
    ks: tuple[int, ...] = (1, 5, 10, 20, 50),
) -> dict[str, float]:
    max_k = max(ks)
    hits_at_k = {k: 0 for k in ks}
    precision_at_k = {k: 0.0 for k in ks}
    reciprocal_ranks: list[float] = []

    valid_ids = set(int(rid) for rid in retriever.recipe_ids)
    evaluated = 0

    for item in queries:
        if int(item["recipe_id"]) not in valid_ids:
            continue
        evaluated += 1
        rank = rank_target(retriever, item, max_k)
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            for k in ks:
                if rank <= k:
                    hits_at_k[k] += 1
                    precision_at_k[k] += 1.0 / k
        else:
            reciprocal_ranks.append(0.0)

    if evaluated == 0:
        return {"evaluated_queries": 0.0, "mrr": 0.0}

    metrics: dict[str, float] = {"evaluated_queries": float(evaluated)}
    for k in ks:
        metrics[f"recall@{k}"] = round(hits_at_k[k] / evaluated, 4)
        metrics[f"precision@{k}"] = round(precision_at_k[k] / evaluated, 4)
    metrics["mrr"] = round(sum(reciprocal_ranks) / evaluated, 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PARQUET)
    parser.add_argument("--vectorizer", type=Path, default=DEFAULT_VECTORIZER_PATH)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS_PATH)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--sample-queries", type=int, default=200)
    args = parser.parse_args()

    retriever = CandidateRetriever.from_artifacts(
        corpus_path=args.corpus,
        vectorizer_path=args.vectorizer,
        vectors_path=args.vectors,
    )

    queries: list[dict[str, object]] = []
    if args.queries.exists():
        payload = json.loads(args.queries.read_text(encoding="utf-8"))
        queries = payload.get("queries", [])

    metrics = evaluate_retriever(retriever, queries)
    if metrics.get("evaluated_queries", 0.0) == 0.0:
        queries = generate_in_memory_queries(retriever.corpus, num_queries=args.sample_queries)
        metrics = evaluate_retriever(retriever, queries)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
