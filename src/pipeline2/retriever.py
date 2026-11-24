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
    corpus_signature,
)


REQUIRED_CORPUS_COLUMNS = (
    "recipe_id",
    "title",
    "ingredients",
    "canonical_ingredients",
    "directions",
    "total_calories",
    "calories_per_serving",
    "calorie_coverage",
)


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _row_calories_per_serving(row: pd.Series) -> float | None:
    """Missing/non-finite/zero per-serving values must not override total calories."""
    try:
        number = float(row.get("calories_per_serving"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


class CandidateRetriever:
    def __init__(
        self,
        vectorizer: RecipeVectorizer,
        recipe_vectors: csr_matrix,
        corpus: pd.DataFrame,
    ) -> None:
        corpus = corpus.reset_index(drop=True)
        missing = [column for column in REQUIRED_CORPUS_COLUMNS if column not in corpus.columns]
        if missing:
            raise ValueError(
                f"corpus is missing required columns {missing}; "
                "rebuild it with src.preprocessing.build_corpus.build_processed_corpus"
            )
        if recipe_vectors.shape[0] != len(corpus):
            raise ValueError(
                f"recipe_vectors has {recipe_vectors.shape[0]} rows but corpus has {len(corpus)} recipes; "
                "artifacts are stale — refit them with src.pipeline2.vectorizer.fit_corpus_tfidf"
            )
        expected_signature = getattr(vectorizer, "corpus_signature", None)
        if expected_signature and expected_signature != corpus_signature(corpus):
            raise ValueError(
                "TF-IDF artifacts were fitted on a different corpus snapshot "
                f"(artifact {expected_signature}, corpus {corpus_signature(corpus)}); "
                "refit with src.pipeline2.vectorizer.fit_corpus_tfidf"
            )
        self.vectorizer = vectorizer
        self.recipe_vectors = recipe_vectors
        self.corpus = corpus
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

    def _empty_result(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    def retrieve_indices(
        self,
        structured_query: dict | Iterable[str] | str,
        top_n: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_vector = self.vectorizer.transform_query(structured_query)
        if query_vector.nnz == 0:
            return self._empty_result()
        scores = cosine_similarity(query_vector, self.recipe_vectors).ravel()
        total = len(scores)
        if total == 0 or top_n <= 0 or float(scores.max(initial=0.0)) <= 0.0:
            return self._empty_result()
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
                    "calories_per_serving": _row_calories_per_serving(row),
                    "calorie_coverage": float(row["calorie_coverage"]),
                    "tfidf_score": round(float(score), 6),
                }
            )
        return candidates
