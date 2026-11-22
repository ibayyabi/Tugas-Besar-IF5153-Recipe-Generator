from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline2.vectorizer import (
    DEFAULT_CORPUS_PARQUET,
    DEFAULT_VECTORIZER_PATH,
    DEFAULT_VECTORS_PATH,
    RecipeVectorizer,
)


class CandidateRetriever:
    def __init__(
        self,
        vectorizer: RecipeVectorizer,
        recipe_vectors: csr_matrix,
        corpus: pd.DataFrame,
    ) -> None:
        self.vectorizer = vectorizer
        self.recipe_vectors = recipe_vectors
        self.corpus = corpus.reset_index(drop=True)
        self.recipe_ids = self.corpus["recipe_id"].to_numpy()

    @classmethod
    def from_artifacts(
        cls,
        corpus_path: Path | str = DEFAULT_CORPUS_PARQUET,
        vectorizer_path: Path | str = DEFAULT_VECTORIZER_PATH,
        vectors_path: Path | str = DEFAULT_VECTORS_PATH,
    ) -> CandidateRetriever:
        vectorizer, recipe_vectors = RecipeVectorizer.load(
            vectorizer_path=vectorizer_path,
            vectors_path=vectors_path,
        )
        corpus = pd.read_parquet(corpus_path)
        return cls(vectorizer=vectorizer, recipe_vectors=recipe_vectors, corpus=corpus)

    def score_all(self, structured_query: dict | Iterable[str] | str) -> np.ndarray:
        query_vector = self.vectorizer.transform_query(structured_query)
        if query_vector.nnz == 0:
            return np.zeros(self.recipe_vectors.shape[0], dtype=np.float32)
        scores = cosine_similarity(query_vector, self.recipe_vectors).ravel()
        return scores

    def retrieve_indices(
        self,
        structured_query: dict | Iterable[str] | str,
        top_n: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score_all(structured_query)
        total = len(scores)
        if total == 0 or top_n <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(top_n, total)
        if k < total:
            top_idx = np.argpartition(scores, -k)[-k:]
            ordered = top_idx[np.argsort(scores[top_idx])[::-1]]
        else:
            ordered = np.argsort(scores)[::-1]
        return ordered, scores[ordered]

    def retrieve(
        self,
        structured_query: dict | Iterable[str] | str,
        top_n: int = 50,
    ) -> list[dict[str, object]]:
        indices, scores = self.retrieve_indices(structured_query, top_n=top_n)
        candidates: list[dict[str, object]] = []
        for idx, score in zip(indices, scores):
            row = self.corpus.iloc[int(idx)]
            candidates.append(
                {
                    "recipe_id": int(row["recipe_id"]),
                    "title": str(row["title"]),
                    "ingredients": list(row["ingredients"]),
                    "canonical_ingredients": list(row["canonical_ingredients"]),
                    "directions": list(row["directions"]),
                    "total_calories": float(row["total_calories"]),
                    "calorie_coverage": float(row["calorie_coverage"]),
                    "tfidf_score": round(float(score), 6),
                }
            )
        return candidates
