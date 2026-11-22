from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from evaluation.evaluate_retrieval import generate_in_memory_queries, rank_target
from src.pipeline2.approaches import DEFAULT_DENSE_META_PATH, DEFAULT_DENSE_VECTORS_PATH, DenseApproach
from src.pipeline2.retriever import CandidateRetriever
from src.pipeline2.vectorizer import (
    DEFAULT_CORPUS_PARQUET,
    DEFAULT_STAPLES_PATH,
    RecipeVectorizer,
    query_to_text,
)
from src.preprocessing.ingredient_normalizer import normalize_ingredient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = ROOT / "artifacts" / "p2_experiment_results.json"
DEFAULT_OUT_CHART = ROOT / "artifacts" / "p2_error_analysis.png"


def _load_staples(path: Path = DEFAULT_STAPLES_PATH) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as fh:
        return {normalize_ingredient(r["ingredient"]) for r in csv.DictReader(fh)}


def _build_approaches(corpus: pd.DataFrame) -> list[tuple[str, str, object, set[str] | None]]:
    documents = corpus["canonical_text"].fillna("").astype(str).tolist()
    specs = [
        (
            "tfidf-raw",
            "TF-IDF baseline (max_features=1000, ngram 1-2) tanpa down-weighting pantry staple",
            dict(max_features=1000, ngram_range=(1, 2), sublinear_tf=False, use_staple_downweight=False),
        ),
        (
            "tfidf-staple",
            "TF-IDF produksi (max_features=1000, ngram 1-2) dengan pantry staple down-weighting",
            dict(max_features=1000, ngram_range=(1, 2), sublinear_tf=False, use_staple_downweight=True),
        ),
        (
            "tfidf-unigram",
            "TF-IDF unigram saja (max_features=1000, ngram 1-1) dengan staple down-weighting",
            dict(max_features=1000, ngram_range=(1, 1), sublinear_tf=False, use_staple_downweight=True),
        ),
        (
            "tfidf-unigram-full",
            "TF-IDF unigram kosakata penuh (max_features=None, ngram 1-1, sublinear_tf) + staple down-weight",
            dict(max_features=None, ngram_range=(1, 1), sublinear_tf=True, use_staple_downweight=True),
        ),
    ]

    built: list[tuple[str, str, object, set[str] | None]] = []
    for name, desc, kwargs in specs:
        vec = RecipeVectorizer(**kwargs)
        mat = vec.fit_transform(documents)
        retriever = CandidateRetriever(vectorizer=vec, recipe_vectors=mat, corpus=corpus)
        vocab_terms = set(vec.vectorizer.get_feature_names_out())
        built.append((name, desc, retriever, vocab_terms))

    if DEFAULT_DENSE_VECTORS_PATH.exists() and DEFAULT_DENSE_META_PATH.exists():
        try:
            dense = DenseApproach()
            _ = dense.model
            built.append(
                (
                    "dense-minilm",
                    "Embedding semantik lokal all-MiniLM-L6-v2 (384-dim, L2-normalized) + cosine similarity",
                    dense,
                    None,
                )
            )
        except Exception:
            pass

    return built


def _classify_miss(
    item: dict[str, object],
    vocab_terms: set[str] | None,
    staples: set[str],
) -> str:
    pantry = [str(x) for x in item["pantry"]]
    required = [str(x) for x in item["required"]]

    if vocab_terms is not None:
        query_tokens = query_to_text(pantry).split()
        if any(tok not in vocab_terms for tok in query_tokens):
            return "miss_oov"

    staple_hits = sum(1 for ing in pantry if normalize_ingredient(ing) in staples)
    if pantry and (staple_hits / len(pantry)) >= 0.5:
        return "miss_staple"

    if len(required) >= 8:
        return "miss_long_recipe"

    return "miss_overlap"


def run_p2_experiment(
    corpus_path: Path = DEFAULT_CORPUS_PARQUET,
    num_queries: int = 100,
    pantry_fraction: float = 0.6,
    seed: int = 42,
    miss_cutoff: int = 10,
    ks: tuple[int, ...] = (1, 5, 10, 20, 50),
) -> list[dict[str, object]]:
    corpus = pd.read_parquet(corpus_path).reset_index(drop=True)
    queries = generate_in_memory_queries(
        corpus,
        num_queries=num_queries,
        pantry_fraction=pantry_fraction,
        seed=seed,
    )
    staples = _load_staples()
    approaches = _build_approaches(corpus)
    max_k = max(ks)
    results: list[dict[str, object]] = []

    for name, desc, retriever, vocab_terms in approaches:
        hits = {k: 0 for k in ks}
        rr_list: list[float] = []
        errors = {
            "miss_oov": 0,
            "miss_staple": 0,
            "miss_long_recipe": 0,
            "miss_overlap": 0,
        }
        t0 = time.perf_counter()
        for item in queries:
            rank = rank_target(retriever, item, max_k)

            if rank is not None:
                rr_list.append(1.0 / rank)
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
            else:
                rr_list.append(0.0)

            if rank is None or rank > miss_cutoff:
                cause = _classify_miss(item, vocab_terms, staples)
                errors[cause] += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n = max(len(queries), 1)
        row: dict[str, object] = {
            "approach": name,
            "description": desc,
            "queries": len(queries),
            "recall@1": round(hits[1] / n, 4),
            "recall@5": round(hits[5] / n, 4),
            "recall@10": round(hits[10] / n, 4),
            "recall@20": round(hits[20] / n, 4),
            "recall@50": round(hits[50] / n, 4),
            "mrr": round(sum(rr_list) / n, 4),
            "avg_ms": round(elapsed_ms / n, 2),
            "total_misses": sum(errors.values()),
            "errors": errors,
        }
        results.append(row)

    return results


MISS_CATEGORIES = [
    ("miss_oov", "Vocab terpotong (OOV)", "#175935"),
    ("miss_staple", "Dominasi pantry staple", "#2D8659"),
    ("miss_long_recipe", "Resep panjang (>=8 bahan)", "#D6702B"),
    ("miss_overlap", "Kompetitor / overlap leksikal", "#547EB2"),
]


def save_error_chart(
    results: list[dict[str, object]],
    out_path: Path = DEFAULT_OUT_CHART,
    miss_cutoff: int = 10,
) -> Path:
    """Stacked bar chart of retrieval miss causes per approach (matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_rows = sorted(results, key=lambda r: int(r["total_misses"]), reverse=True)
    labels = [str(r["approach"]) for r in sorted_rows]
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.62 * len(labels) + 2.5)))

    left = [0] * len(sorted_rows)
    for key, label, color in MISS_CATEGORIES:
        values = [int(r["errors"].get(key, 0)) for r in sorted_rows]
        ax.barh(labels, values, left=left, color=color, label=label)
        left = [l + v for l, v in zip(left, values)]

    ax.invert_yaxis()
    ax.set_title("Dari mana miss retrieval berasal? — Pipeline 2")
    ax.set_xlabel(f"jumlah miss (target gagal masuk Top-{miss_cutoff})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PARQUET)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--miss-cutoff", type=int, default=10)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-chart", type=Path, default=DEFAULT_OUT_CHART)
    args = parser.parse_args()

    results = run_p2_experiment(
        corpus_path=args.corpus,
        num_queries=args.queries,
        miss_cutoff=args.miss_cutoff,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_error_chart(results, out_path=args.out_chart, miss_cutoff=args.miss_cutoff)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
