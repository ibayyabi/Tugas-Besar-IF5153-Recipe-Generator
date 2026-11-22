from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from src.preprocessing.ingredient_normalizer import DEFAULT_PROCESSED_DIR, normalize_ingredient

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_VECTORIZER_PATH = DEFAULT_ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
DEFAULT_VECTORS_PATH = DEFAULT_ARTIFACTS_DIR / "recipe_vectors.pkl"
DEFAULT_CORPUS_PARQUET = DEFAULT_PROCESSED_DIR / "recipes.parquet"
DEFAULT_STAPLES_PATH = DEFAULT_PROCESSED_DIR / "pantry_staples.csv"


def _load_staple_weights(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    weights: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = normalize_ingredient(row["ingredient"])
            if key:
                weights[key] = float(row["weight"])
    return weights


def query_to_text(query: dict | Iterable[str] | str) -> str:
    if isinstance(query, dict):
        items = query.get("available_ingredients") or []
        return " ".join(normalize_ingredient(str(item)) for item in items if str(item).strip())
    if isinstance(query, str):
        return normalize_ingredient(query)
    return " ".join(normalize_ingredient(str(item)) for item in query if str(item).strip())


class RecipeVectorizer:
    def __init__(
        self,
        max_features: int | None = None,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True,
        use_staple_downweight: bool = True,
        staples_path: Path | str | None = DEFAULT_STAPLES_PATH,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.use_staple_downweight = use_staple_downweight
        self.staples_path = Path(staples_path) if staples_path is not None else None

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            lowercase=True,
            token_pattern=r"(?u)\b[a-z0-9_]+\b",
        )
        self.staple_weights: dict[str, float] = (
            _load_staple_weights(self.staples_path) if self.use_staple_downweight else {}
        )
        self.feature_weights_: np.ndarray | None = None

    def _build_feature_weights(self) -> np.ndarray:
        feature_names = self.vectorizer.get_feature_names_out()
        weights = np.ones(len(feature_names), dtype=np.float32)
        if not self.use_staple_downweight or not self.staple_weights:
            return weights
        for idx, term in enumerate(feature_names):
            if term in self.staple_weights:
                weights[idx] = self.staple_weights[term]
        return weights

    def _apply_weights(self, matrix: csr_matrix) -> csr_matrix:
        if not self.use_staple_downweight or self.feature_weights_ is None:
            return matrix
        weighted = matrix @ diags(self.feature_weights_, format="csr")
        return sk_normalize(weighted, norm="l2", axis=1, copy=False)

    def fit_transform(self, documents: list[str]) -> csr_matrix:
        matrix = self.vectorizer.fit_transform(documents)
        self.feature_weights_ = self._build_feature_weights()
        return self._apply_weights(matrix)

    def transform_documents(self, documents: list[str]) -> csr_matrix:
        matrix = self.vectorizer.transform(documents)
        return self._apply_weights(matrix)

    def transform_query(self, query: dict | Iterable[str] | str) -> csr_matrix:
        text = query_to_text(query)
        matrix = self.vectorizer.transform([text])
        return self._apply_weights(matrix)

    def save(
        self,
        recipe_vectors: csr_matrix,
        vectorizer_path: Path | str = DEFAULT_VECTORIZER_PATH,
        vectors_path: Path | str = DEFAULT_VECTORS_PATH,
    ) -> tuple[Path, Path]:
        vec_file = Path(vectorizer_path)
        mat_file = Path(vectors_path)
        vec_file.parent.mkdir(parents=True, exist_ok=True)
        mat_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "use_staple_downweight": self.use_staple_downweight,
            "staple_weights": self.staple_weights,
            "feature_weights_": self.feature_weights_,
            "vectorizer": self.vectorizer,
        }
        with vec_file.open("wb") as fh:
            pickle.dump(state, fh)
        with mat_file.open("wb") as fh:
            pickle.dump(recipe_vectors, fh)
        return vec_file, mat_file

    @classmethod
    def load(
        cls,
        vectorizer_path: Path | str = DEFAULT_VECTORIZER_PATH,
        vectors_path: Path | str = DEFAULT_VECTORS_PATH,
    ) -> tuple[RecipeVectorizer, csr_matrix]:
        with Path(vectorizer_path).open("rb") as fh:
            state = pickle.load(fh)
        with Path(vectors_path).open("rb") as fh:
            recipe_vectors = pickle.load(fh)
        if isinstance(state, cls):
            return state, recipe_vectors
        instance = cls(
            max_features=state["max_features"],
            ngram_range=state["ngram_range"],
            min_df=state["min_df"],
            max_df=state["max_df"],
            sublinear_tf=state["sublinear_tf"],
            use_staple_downweight=state["use_staple_downweight"],
            staples_path=None,
        )
        instance.staple_weights = state["staple_weights"]
        instance.feature_weights_ = state["feature_weights_"]
        instance.vectorizer = state["vectorizer"]
        return instance, recipe_vectors


def fit_corpus_tfidf(
    corpus_parquet: Path | str = DEFAULT_CORPUS_PARQUET,
    vectorizer_path: Path | str = DEFAULT_VECTORIZER_PATH,
    vectors_path: Path | str = DEFAULT_VECTORS_PATH,
    max_features: int | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    sublinear_tf: bool = True,
    use_staple_downweight: bool = True,
) -> tuple[RecipeVectorizer, csr_matrix]:
    df = pd.read_parquet(corpus_parquet)
    if "canonical_text" in df.columns:
        documents = df["canonical_text"].fillna("").astype(str).tolist()
    else:
        documents = [" ".join(items) for items in df["canonical_ingredients"].tolist()]

    vectorizer = RecipeVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        use_staple_downweight=use_staple_downweight,
    )
    recipe_vectors = vectorizer.fit_transform(documents)
    vectorizer.save(
        recipe_vectors=recipe_vectors,
        vectorizer_path=vectorizer_path,
        vectors_path=vectors_path,
    )
    return vectorizer, recipe_vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PARQUET)
    parser.add_argument("--vectorizer-out", type=Path, default=DEFAULT_VECTORIZER_PATH)
    parser.add_argument("--vectors-out", type=Path, default=DEFAULT_VECTORS_PATH)
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--ngram-max", type=int, default=1)
    parser.add_argument("--no-sublinear-tf", action="store_true")
    parser.add_argument("--no-staple-downweight", action="store_true")
    args = parser.parse_args()

    max_features = None if args.max_features <= 0 else args.max_features
    _, vectors = fit_corpus_tfidf(
        corpus_parquet=args.corpus,
        vectorizer_path=args.vectorizer_out,
        vectors_path=args.vectors_out,
        max_features=max_features,
        ngram_range=(1, args.ngram_max),
        sublinear_tf=not args.no_sublinear_tf,
        use_staple_downweight=not args.no_staple_downweight,
    )
    print(f"fitted_shape={vectors.shape} nnz={vectors.nnz}")


if __name__ == "__main__":
    main()
