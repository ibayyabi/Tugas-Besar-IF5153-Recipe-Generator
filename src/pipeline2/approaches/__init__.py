"""Pipeline 2 — pluggable retrieval approaches.

Both approaches answer one contract so the query parser output, the reranker,
and the UI can swap them freely:

* ``retrieve(structured_query, top_n) -> list[dict]`` — candidate dicts with the
  exact schema ``pipeline3.models.CandidateRecipe.from_dict`` consumes
  (``recipe_id``, ``title``, ``ingredients``, ``canonical_ingredients``,
  ``directions``, ``total_calories``, ``calorie_coverage``, ``tfidf_score``);
* ``trace(structured_query, top_n) -> RetrievalTrace`` — query terms, weights,
  and per-candidate scores for the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.pipeline2.retriever import CandidateRetriever
from src.pipeline2.vectorizer import (
    DEFAULT_VECTORS_PATH,
    DEFAULT_VECTORIZER_PATH,
    RecipeVectorizer,
    query_to_text,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_DENSE_VECTORS_PATH = DEFAULT_ARTIFACTS_DIR / "dense_vectors.npy"
DEFAULT_DENSE_META_PATH = DEFAULT_ARTIFACTS_DIR / "dense_meta.json"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class RetrievalTrace:
    """Intermediate evidence for one retrieval call; rendered by the UI."""

    approach: str
    query_text: str
    score_type: str  # tfidf-cosine | dense-cosine
    query_terms: list[dict[str, float]] = field(default_factory=list)  # [{"term", "weight"}]
    candidates: list[dict[str, object]] = field(default_factory=list)
    extras: dict[str, object] = field(default_factory=dict)


class RetrievalApproach:
    """Strategy interface wrapping a concrete retriever."""

    name: str = ""
    description: str = ""

    def score_all(self, structured_query: dict | list[str] | str) -> np.ndarray:
        raise NotImplementedError

    def retrieve(self, structured_query: dict | list[str] | str, top_n: int = 50) -> list[dict]:
        raise NotImplementedError

    def trace(self, structured_query: dict | list[str] | str, top_n: int = 50) -> RetrievalTrace:
        raise NotImplementedError


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def available_approaches() -> list[dict[str, object]]:
    """Registry metadata for UI pickers; each ``available`` flags artifact readiness."""
    tfidf_ready = DEFAULT_VECTORIZER_PATH.exists() and DEFAULT_VECTORS_PATH.exists()
    dense_ready = (
        _sentence_transformers_available()
        and DEFAULT_DENSE_VECTORS_PATH.exists()
        and DEFAULT_DENSE_META_PATH.exists()
    )
    return [
        {"name": TfidfApproach.name, "description": TfidfApproach.description, "available": tfidf_ready},
        {"name": DenseApproach.name, "description": DenseApproach.description, "available": dense_ready},
    ]


def get_approach(name: str) -> RetrievalApproach:
    if name == TfidfApproach.name:
        return TfidfApproach()
    if name == DenseApproach.name:
        return DenseApproach()
    raise KeyError(f"unknown Pipeline 2 approach: {name!r}")


class TfidfApproach(RetrievalApproach):
    """TF-IDF + cosine similarity over canonical ingredient text (production baseline)."""

    name = "tfidf"
    description = "TF-IDF unigram (full vocab, sublinear_tf) with pantry-staple down-weighting, cosine similarity."

    def __init__(self, retriever: CandidateRetriever | None = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> CandidateRetriever:
        if self._retriever is None:
            self._retriever = CandidateRetriever.from_artifacts()
        return self._retriever

    def score_all(self, structured_query: dict | list[str] | str) -> np.ndarray:
        return self.retriever.score_all(structured_query)

    @property
    def recipe_ids(self) -> np.ndarray:
        """Harness compatibility: evaluate_retriever reads recipe_ids directly."""
        return self.retriever.recipe_ids

    def retrieve_indices(self, structured_query: dict | list[str] | str, top_n: int = 50) -> tuple[np.ndarray, np.ndarray]:
        return self.retriever.retrieve_indices(structured_query, top_n=top_n)

    def retrieve(self, structured_query: dict | list[str] | str, top_n: int = 50) -> list[dict]:
        return self.retriever.retrieve(structured_query, top_n=top_n)

    def trace(self, structured_query: dict | list[str] | str, top_n: int = 50) -> RetrievalTrace:
        vectorizer: RecipeVectorizer = self.retriever.vectorizer
        query_vector = vectorizer.transform_query(structured_query)
        feature_names = vectorizer.vectorizer.get_feature_names_out()

        terms: list[dict[str, float]] = []
        if query_vector.nnz:
            row = query_vector.tocsr()[0]
            for idx, weight in zip(row.indices, row.data):
                terms.append({"term": str(feature_names[idx]), "weight": round(float(weight), 4)})
            terms.sort(key=lambda item: item["weight"], reverse=True)

        candidates = self.retrieve(structured_query, top_n=top_n)
        return RetrievalTrace(
            approach=self.name,
            query_text=query_to_text(structured_query),
            score_type="tfidf-cosine",
            query_terms=terms[:20],
            candidates=candidates,
            extras={
                "top_n": top_n,
                "matrix_shape": list(self.retriever.recipe_vectors.shape),
                "staple_downweight": vectorizer.use_staple_downweight,
            },
        )


class DenseApproach(RetrievalApproach):
    """Sentence-transformer embeddings + cosine similarity over the same corpus."""

    name = "dense"
    description = "Sentence-transformer (all-MiniLM-L6-v2) dense vectors over canonical ingredient text, cosine similarity."

    def __init__(self, model=None, vectors_path: Path = DEFAULT_DENSE_VECTORS_PATH, meta_path: Path = DEFAULT_DENSE_META_PATH) -> None:
        self._model = model
        self._vectors: np.ndarray | None = None
        self._corpus = None
        self._vectors_path = Path(vectors_path)
        self._meta_path = Path(meta_path)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._load_meta()["model"])
        return self._model

    def _load_meta(self) -> dict:
        import json

        if not self._meta_path.exists():
            raise FileNotFoundError(
                f"dense artifacts not found at {self._meta_path}; run scripts/encode_dense_corpus.py first"
            )
        return json.loads(self._meta_path.read_text(encoding="utf-8"))

    def _ensure_artifacts(self) -> None:
        if self._vectors is not None:
            return
        meta = self._load_meta()
        self._vectors = np.load(self._vectors_path)
        import pandas as pd

        self._corpus = pd.read_parquet(meta["corpus_path"]).reset_index(drop=True)

    def score_all(self, structured_query: dict | list[str] | str) -> np.ndarray:
        self._ensure_artifacts()
        query_text = query_to_text(structured_query)
        if not query_text.strip():
            return np.zeros(self._vectors.shape[0], dtype=np.float32)
        query_vec = self.model.encode(query_text, normalize_embeddings=True)
        return self._vectors @ np.asarray(query_vec, dtype=np.float32)

    @property
    def recipe_ids(self) -> np.ndarray:
        self._ensure_artifacts()
        return self._corpus["recipe_id"].to_numpy()

    def retrieve_indices(self, structured_query: dict | list[str] | str, top_n: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Same contract as ``CandidateRetriever.retrieve_indices`` (harness compatibility)."""
        self._ensure_artifacts()
        scores = self.score_all(structured_query)
        total = len(scores)
        if total == 0 or top_n <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(top_n, total)
        top_idx = np.argpartition(scores, -k)[-k:]
        ordered = top_idx[np.argsort(scores[top_idx])[::-1]]
        return ordered, scores[ordered]

    def retrieve(self, structured_query: dict | list[str] | str, top_n: int = 50) -> list[dict]:
        self._ensure_artifacts()
        scores = self.score_all(structured_query)
        total = len(scores)
        if total == 0 or top_n <= 0:
            return []
        k = min(top_n, total)
        top_idx = np.argpartition(scores, -k)[-k:]
        ordered = top_idx[np.argsort(scores[top_idx])[::-1]]
        candidates: list[dict[str, object]] = []
        for idx in ordered:
            row = self._corpus.iloc[int(idx)]
            candidates.append(
                {
                    "recipe_id": int(row["recipe_id"]),
                    "title": str(row["title"]),
                    "ingredients": list(row["ingredients"]),
                    "canonical_ingredients": list(row["canonical_ingredients"]),
                    "directions": list(row["directions"]),
                    "total_calories": float(row["total_calories"]),
                    "calorie_coverage": float(row["calorie_coverage"]),
                    "tfidf_score": round(float(scores[idx]), 6),
                }
            )
        return candidates

    def trace(self, structured_query: dict | list[str] | str, top_n: int = 50) -> RetrievalTrace:
        query_text = query_to_text(structured_query)
        candidates = self.retrieve(structured_query, top_n=top_n)
        return RetrievalTrace(
            approach=self.name,
            query_text=query_text,
            score_type="dense-cosine",
            query_terms=[{"term": word, "weight": 1.0} for word in query_text.split()],
            candidates=candidates,
            extras={
                "top_n": top_n,
                "vectors_shape": list(self._vectors.shape) if self._vectors is not None else None,
                "model": self._load_meta()["model"],
            },
        )
