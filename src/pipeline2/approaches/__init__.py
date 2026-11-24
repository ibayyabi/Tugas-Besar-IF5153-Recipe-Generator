"""Pipeline 2 — pluggable retrieval approaches.

Both approaches answer one contract so the query parser output, the reranker,
and the UI can swap them freely:

* ``retrieve(structured_query, top_n) -> list[dict]`` — candidate dicts with the
  exact schema ``pipeline3.models.CandidateRecipe.from_dict`` consumes
  (``recipe_id``, ``title``, ``ingredients``, ``canonical_ingredients``,
  ``directions``, ``total_calories``, ``calories_per_serving``,
  ``calorie_coverage``, ``tfidf_score``);
* ``trace(structured_query, top_n) -> RetrievalTrace`` — query terms, weights,
  and per-candidate scores for the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.pipeline2.retriever import (
    CandidateRetriever,
    REQUIRED_CORPUS_COLUMNS,
    _row_calories_per_serving as _calories_per_serving,
)
from src.pipeline2.vectorizer import (
    DEFAULT_VECTORS_PATH,
    DEFAULT_VECTORIZER_PATH,
    RecipeVectorizer,
    corpus_signature,
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
    no_match_reason: str | None = None


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

        query_text = query_to_text(structured_query)
        candidates = self.retrieve(structured_query, top_n=top_n)
        no_match_reason = None
        if not candidates:
            no_match_reason = (
                "No query ingredients were provided."
                if not query_text.strip()
                else "No recipe shares a TF-IDF term with the query ingredients."
            )
        return RetrievalTrace(
            approach=self.name,
            query_text=query_text,
            score_type="tfidf-cosine",
            query_terms=terms[:20],
            candidates=candidates,
            no_match_reason=no_match_reason,
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
        vectors = np.load(self._vectors_path)
        import pandas as pd

        corpus_path = Path(meta["corpus_path"])
        if not corpus_path.is_absolute() and not corpus_path.exists():
            corpus_path = ROOT / corpus_path
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"dense corpus not found at {corpus_path} (meta corpus_path={meta['corpus_path']!r}); "
                "run scripts/encode_dense_corpus.py first"
            )
        corpus = pd.read_parquet(corpus_path).reset_index(drop=True)
        missing = [column for column in REQUIRED_CORPUS_COLUMNS if column not in corpus.columns]
        if missing:
            raise ValueError(
                f"dense corpus is missing required columns {missing}; "
                "rebuild it with src.preprocessing.build_corpus.build_processed_corpus"
            )
        if vectors.shape[0] != len(corpus):
            raise ValueError(
                f"dense vectors have {vectors.shape[0]} rows but corpus {corpus_path} has {len(corpus)} recipes; "
                "artifacts are stale — rerun scripts/encode_dense_corpus.py"
            )
        declared = meta.get("n_recipes")
        if declared is not None and int(declared) != vectors.shape[0]:
            raise ValueError(
                f"dense meta declares n_recipes={declared} but dense_vectors.npy has {vectors.shape[0]} rows; "
                "rerun scripts/encode_dense_corpus.py"
            )
        declared_dim = meta.get("dim")
        if declared_dim is not None and int(declared_dim) != vectors.shape[1]:
            raise ValueError(
                f"dense meta declares dim={declared_dim} but dense_vectors.npy has {vectors.shape[1]} columns; "
                "rerun scripts/encode_dense_corpus.py"
            )
        declared_signature = meta.get("corpus_signature")
        if declared_signature and declared_signature != corpus_signature(corpus):
            raise ValueError(
                "dense vectors were built from a different corpus snapshot "
                f"(meta {declared_signature}, corpus {corpus_signature(corpus)}); "
                "rerun scripts/encode_dense_corpus.py"
            )
        self._vectors = vectors
        self._corpus = corpus

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
        if total == 0 or top_n <= 0 or float(np.max(scores, initial=0.0)) <= 0.0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(top_n, total)
        top_idx = np.argpartition(scores, -k)[-k:]
        ordered = top_idx[np.argsort(scores[top_idx])[::-1]]
        return ordered, scores[ordered]

    def retrieve(self, structured_query: dict | list[str] | str, top_n: int = 50) -> list[dict]:
        self._ensure_artifacts()
        indices, scores = self.retrieve_indices(structured_query, top_n=top_n)
        candidates: list[dict[str, object]] = []
        for idx, score in zip(indices, scores):
            row = self._corpus.iloc[int(idx)]
            candidates.append(
                {
                    "recipe_id": int(row["recipe_id"]),
                    "title": str(row["title"]),
                    "ingredients": list(row["ingredients"]),
                    "canonical_ingredients": list(row["canonical_ingredients"]),
                    "directions": list(row["directions"]),
                    "total_calories": float(row["total_calories"]),
                    "calories_per_serving": _calories_per_serving(row),
                    "calorie_coverage": float(row["calorie_coverage"]),
                    "tfidf_score": round(float(score), 6),
                }
            )
        return candidates

    def trace(self, structured_query: dict | list[str] | str, top_n: int = 50) -> RetrievalTrace:
        query_text = query_to_text(structured_query)
        candidates = self.retrieve(structured_query, top_n=top_n)
        no_match_reason = None
        if not candidates:
            no_match_reason = (
                "No query ingredients were provided."
                if not query_text.strip()
                else "No recipe is semantically close to the query ingredients."
            )
        return RetrievalTrace(
            approach=self.name,
            query_text=query_text,
            score_type="dense-cosine",
            no_match_reason=no_match_reason,
            query_terms=[{"term": word, "weight": 1.0} for word in query_text.split()],
            candidates=candidates,
            extras={
                "top_n": top_n,
                "vectors_shape": list(self._vectors.shape) if self._vectors is not None else None,
                "model": self._load_meta()["model"],
            },
        )
