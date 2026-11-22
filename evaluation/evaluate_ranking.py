#!/usr/bin/env python3
"""Standalone evaluation & ablation study for Pipeline 3 (PRD Sections 33-35).

Runs independently without waiting for Pipeline 1 & Pipeline 2 by:
1. Reading ground-truth structured queries from ``data/test/annotated_queries.json``
   (decoupling Pipeline 3 evaluation from Pipeline 1 parsing errors).
2. Running a self-contained local TF-IDF candidate retriever on
   ``data/samples/recipes_sample.csv`` (matching PRD §17: TfidfVectorizer with
   ``max_features=1000, ngram_range=(1, 2)`` + cosine similarity, Top-N=50).
3. Estimating recipe calories on-demand via ``RecipeCalorieEstimator`` using
   the committed tables in ``data/processed/``.
4. Comparing Baseline vs Proposed and running the full PRD §35 Ablation Study:
   - Model A: TF-IDF only (Baseline)
   - Model B: TF-IDF + Ingredient Coverage
   - Model C: TF-IDF + Ingredient Coverage + Missing Ingredient Penalty
   - Model D (Full Proposed): TF-IDF + Coverage + Missing Penalty + Calorie Score + Exclusion Filter
   - Model D (Unweighted): Full Proposed without pantry_staples.csv weights

Usage:
    python evaluation/evaluate_ranking.py
    python evaluation/evaluate_ranking.py --top-k 3 --candidate-n 50 --examples 3
    python evaluation/evaluate_ranking.py --output-json evaluation/results_pipeline3.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline3 import (
    CandidateRecipe,
    IngredientNormalizer,
    MultiCriteriaReranker,
    PantryStapleWeights,
    RankingConfig,
    RecipeCalorieEstimator,
    StructuredQuery,
    format_recommendation_card,
)

DEFAULT_CORPUS = REPO_ROOT / "data" / "samples" / "recipes_sample.csv"
DEFAULT_QUERIES = REPO_ROOT / "data" / "test" / "annotated_queries.json"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"


@dataclass
class RawRecipeRow:
    recipe_id: int
    title: str
    raw_ingredients: list[str]
    ner_ingredients: list[str]
    canonical_ingredients: list[str]
    directions: list[str]
    ingredient_doc: str


class LocalTfidfRetriever:
    """Self-contained PRD §17 retriever for standalone Pipeline 3 evaluation."""

    def __init__(
        self,
        corpus_path: Path,
        normalizer: IngredientNormalizer,
        max_features: int = 1000,
    ) -> None:
        self.normalizer = normalizer
        self.recipes: list[RawRecipeRow] = []

        with corpus_path.open(encoding="utf-8", newline="") as fh:
            for idx, row in enumerate(csv.DictReader(fh)):
                try:
                    ner = ast.literal_eval(row["NER"])
                    raw_ings = ast.literal_eval(row["ingredients"])
                    directions = ast.literal_eval(row["directions"])
                except (ValueError, SyntaxError):
                    continue
                canonical = normalizer.canonicalize_list(ner)
                if not canonical:
                    continue
                # Join canonical ingredients into space-separated document (PRD §17.2)
                doc = " ".join(canonical)
                self.recipes.append(
                    RawRecipeRow(
                        recipe_id=idx,
                        title=row["title"].strip(),
                        raw_ingredients=raw_ings,
                        ner_ingredients=ner,
                        canonical_ingredients=canonical,
                        directions=directions,
                        ingredient_doc=doc,
                    )
                )

        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([r.ingredient_doc for r in self.recipes])

    def retrieve(
        self,
        query: StructuredQuery,
        estimator: RecipeCalorieEstimator,
        calorie_cache: dict[int, float | None],
        top_n: int = 50,
        calorie_mode: str = "per_serving",
    ) -> list[CandidateRecipe]:
        query_terms = self.normalizer.canonicalize_list(query.available_ingredients)
        query_doc = " ".join(query_terms)
        if not query_doc:
            return []

        q_vec = self.vectorizer.transform([query_doc])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        k = min(top_n, len(self.recipes))
        if k <= 0:
            return []

        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        candidates: list[CandidateRecipe] = []
        for pos in top_indices:
            sim_score = float(sims[pos])
            if sim_score <= 0.0:
                continue
            row = self.recipes[int(pos)]
            if row.recipe_id not in calorie_cache:
                est = estimator.estimate_recipe(
                    row.raw_ingredients,
                    row.ner_ingredients,
                    row.directions,
                    title=row.title,
                )
                calorie_cache[row.recipe_id] = (
                    est.calories_per_serving if calorie_mode == "per_serving" else est.total_calories
                )
            candidates.append(
                CandidateRecipe(
                    recipe_id=row.recipe_id,
                    title=row.title,
                    canonical_ingredients=row.canonical_ingredients,
                    tfidf_score=sim_score,
                    ingredients=row.raw_ingredients,
                    directions=row.directions,
                    total_calories=calorie_cache[row.recipe_id],
                )
            )
        return candidates


def evaluate_model(
    model_name: str,
    config: RankingConfig,
    reranker: MultiCriteriaReranker,
    query_candidates: list[tuple[str, StructuredQuery, list[CandidateRecipe]]],
    top_k: int,
    mrr_coverage_threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute PRD §§33-34 metrics across all evaluated queries for a single config."""
    weighted_covs: list[float] = []
    unweighted_covs: list[float] = []
    missing_totals: list[int] = []
    missing_non_staples: list[int] = []
    calorie_distances: list[float] = []
    constraint_violations = 0
    exclusion_violations = 0
    calorie_bound_violations = 0
    total_recommended = 0
    total_with_exclusion_query = 0
    total_with_calorie_bound_query = 0
    total_with_any_constraint_query = 0
    reciprocal_ranks: list[float] = []
    fallback_queries = 0

    for _qid, query, candidates in query_candidates:
        result = reranker.rerank(candidates, query, top_k=top_k, config=config)
        if result.is_fallback:
            fallback_queries += 1

        recs = result.recommendations
        first_relevant_rr = 0.0

        for rec in recs:
            total_recommended += 1
            weighted_covs.append(rec.coverage.weighted_coverage)
            unweighted_covs.append(rec.coverage.unweighted_coverage)
            missing_totals.append(rec.coverage.missing_count)
            missing_non_staples.append(rec.coverage.missing_non_staple_count)

            if query.has_calorie_constraint and rec.calorie.calorie_distance is not None:
                calorie_distances.append(rec.calorie.calorie_distance)

            excl_bad = bool(query.excluded_ingredients) and rec.exclusion.violated
            cal_bad = query.has_hard_calorie_bound and (not rec.calorie.passed_hard_filter)
            any_bad = excl_bad or cal_bad

            if query.excluded_ingredients:
                total_with_exclusion_query += 1
                if excl_bad:
                    exclusion_violations += 1

            if query.has_hard_calorie_bound:
                total_with_calorie_bound_query += 1
                if cal_bad:
                    calorie_bound_violations += 1

            if query.excluded_ingredients or query.has_hard_calorie_bound:
                total_with_any_constraint_query += 1
                if any_bad:
                    constraint_violations += 1

            # Practical cookability relevance for MRR:
            # no exclusion violation, passes hard calorie bound, and weighted coverage >= threshold
            if first_relevant_rr == 0.0 and (not any_bad) and rec.coverage.weighted_coverage >= mrr_coverage_threshold:
                first_relevant_rr = 1.0 / rec.rank

        reciprocal_ranks.append(first_relevant_rr)

    def _mean(values: list[float] | list[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "model": model_name,
        "top_k": top_k,
        "num_queries": len(query_candidates),
        "weighted_coverage_at_k": round(_mean(weighted_covs), 4),
        "unweighted_coverage_at_k": round(_mean(unweighted_covs), 4),
        "avg_missing_at_k": round(_mean(missing_totals), 3),
        "avg_missing_non_staple_at_k": round(_mean(missing_non_staples), 3),
        "calorie_distance": round(_mean(calorie_distances), 1),
        "constraint_violation_rate": round(
            constraint_violations / total_with_any_constraint_query, 4
        ) if total_with_any_constraint_query else 0.0,
        "exclusion_violation_rate": round(
            exclusion_violations / total_with_exclusion_query, 4
        ) if total_with_exclusion_query else 0.0,
        "calorie_bound_violation_rate": round(
            calorie_bound_violations / total_with_calorie_bound_query, 4
        ) if total_with_calorie_bound_query else 0.0,
        "mrr": round(_mean(reciprocal_ranks), 4),
        "fallback_rate": round(fallback_queries / len(query_candidates), 4) if query_candidates else 0.0,
        "total_recommended": total_recommended,
    }


def format_markdown_table(rows: list[dict[str, Any]], top_k: int) -> str:
    headers = [
        "Model",
        f"Weighted Cov@{top_k} ↑",
        f"Unweighted Cov@{top_k} ↑",
        f"Missing@{top_k} ↓",
        f"Non-Staple Miss@{top_k} ↓",
        "Calorie Dist (kcal) ↓",
        "Constraint Viol ↓",
        "Excl Viol ↓",
        "Cal Viol ↓",
        "MRR ↑",
        "Fallback",
    ]
    sep = ["---"] + ["---:"] * (len(headers) - 1)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        lines.append(
            "| "
            + " | ".join([
                r["model"],
                f"{r['weighted_coverage_at_k']:.1%}",
                f"{r['unweighted_coverage_at_k']:.1%}",
                f"{r['avg_missing_at_k']:.2f}",
                f"{r['avg_missing_non_staple_at_k']:.2f}",
                f"{r['calorie_distance']:.1f}",
                f"{r['constraint_violation_rate']:.1%}",
                f"{r['exclusion_violation_rate']:.1%}",
                f"{r['calorie_bound_violation_rate']:.1%}",
                f"{r['mrr']:.3f}",
                f"{r['fallback_rate']:.1%}",
            ])
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="RecipeNLG CSV corpus")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="Annotated queries JSON")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED, help="Processed tables directory")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K recommendations to evaluate (PRD default: 3)")
    parser.add_argument("--candidate-n", type=int, default=50, help="Top-N candidates from TF-IDF retrieval (PRD: 50)")
    parser.add_argument(
        "--calorie-mode",
        choices=("per_serving", "total"),
        default="per_serving",
        help="Use per-serving or whole-recipe estimated calories (default: per_serving)",
    )
    parser.add_argument("--examples", type=int, default=2, help="Number of qualitative before/after examples to print")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save JSON evaluation report")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to save CSV ablation table")
    args = parser.parse_args()

    t0 = time.perf_counter()
    normalizer = IngredientNormalizer(args.processed_dir)
    staple_weights = PantryStapleWeights.from_csv(
        args.processed_dir / "pantry_staples.csv",
        normalizer=normalizer,
    )
    estimator = RecipeCalorieEstimator(processed_dir=args.processed_dir, normalizer=normalizer)
    reranker = MultiCriteriaReranker(
        processed_dir=args.processed_dir,
        normalizer=normalizer,
        staple_weights=staple_weights,
    )

    print(f"[info] Loading corpus: {args.corpus}")
    retriever = LocalTfidfRetriever(args.corpus, normalizer=normalizer)
    print(f"[info] Indexed {len(retriever.recipes):,} recipes with TF-IDF ({time.perf_counter() - t0:.2f}s)")

    raw_queries = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]
    calorie_cache: dict[int, float | None] = {}
    query_candidates: list[tuple[str, StructuredQuery, list[CandidateRecipe]]] = []

    t_ret = time.perf_counter()
    for entry in raw_queries:
        sq = StructuredQuery.from_dict(entry["expected"], raw_query=entry["query"])
        cands = retriever.retrieve(
            sq,
            estimator=estimator,
            calorie_cache=calorie_cache,
            top_n=args.candidate_n,
            calorie_mode=args.calorie_mode,
        )
        query_candidates.append((entry["id"], sq, cands))

    estimated_non_null = sum(1 for v in calorie_cache.values() if v is not None)
    print(
        f"[info] Retrieved Top-{args.candidate_n} candidates for {len(query_candidates)} queries; "
        f"estimated calories for {estimated_non_null}/{len(calorie_cache)} unique candidate recipes "
        f"({estimated_non_null / max(len(calorie_cache), 1):.1%}) in {time.perf_counter() - t_ret:.2f}s"
    )

    # PRD Section 35 Ablation Study + Pantry Staple Weight validation
    ablation_models: list[tuple[str, RankingConfig]] = [
        ("Model A: TF-IDF Only (Baseline)", RankingConfig.ablation_model_a()),
        ("Model B: TF-IDF + Coverage", RankingConfig.ablation_model_b()),
        ("Model C: TF-IDF + Cov + Missing Pen.", RankingConfig.ablation_model_c()),
        ("Model D: Full Proposed (Weighted)", RankingConfig.ablation_model_d()),
        ("Model D (Unweighted Staples)", RankingConfig.ablation_unweighted()),
    ]

    results = [
        evaluate_model(name, cfg, reranker, query_candidates, top_k=args.top_k)
        for name, cfg in ablation_models
    ]

    print("\n=== Pipeline 3 Evaluation & Ablation Study (PRD §§33-35) ===")
    print(
        f"Corpus: {args.corpus.name} | Queries: {args.queries.name} ({len(query_candidates)}) | "
        f"Top-N={args.candidate_n} -> Top-K={args.top_k} | Calorie mode: {args.calorie_mode}"
    )
    print("Note: MRR measures first Top-K recipe with 0 constraint violations & weighted coverage >= 50%.\n")
    print(format_markdown_table(results, top_k=args.top_k))

    # Print qualitative Baseline vs Proposed examples for combined-constraint queries
    if args.examples > 0:
        demo_ids = ("q041", "q042", "q044", "q045", "q056", "q060")
        selected_demos = [qc for qc in query_candidates if qc[0] in demo_ids][: args.examples]
        if not selected_demos:
            selected_demos = query_candidates[: args.examples]

        print("\n=== Qualitative Comparison: Baseline (Model A) vs Proposed (Model D) ===")
        for qid, sq, cands in selected_demos:
            base_res = reranker.rerank(cands, sq, top_k=1, config=RankingConfig.ablation_model_a())
            prop_res = reranker.rerank(cands, sq, top_k=1, config=RankingConfig.ablation_model_d())
            print(f"\n[{qid}] Query: {sq.raw_query}")
            print("  -- Baseline #1 (TF-IDF Only) --")
            if base_res.recommendations:
                print(
                    format_recommendation_card(
                        base_res.recommendations[0],
                        excluded_ingredients=sq.excluded_ingredients,
                        include_ingredients=False,
                        include_directions=False,
                    )
                )
            print("\n  -- Proposed #1 (Full Multi-Criteria Reranker) --")
            if prop_res.is_fallback and prop_res.fallback_message:
                print(f"  [NOTICE] {prop_res.fallback_message}")
            if prop_res.recommendations:
                print(
                    format_recommendation_card(
                        prop_res.recommendations[0],
                        excluded_ingredients=sq.excluded_ingredients,
                        include_ingredients=False,
                        include_directions=False,
                    )
                )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corpus": str(args.corpus),
            "queries": str(args.queries),
            "top_k": args.top_k,
            "candidate_n": args.candidate_n,
            "calorie_mode": args.calorie_mode,
            "unique_candidates_evaluated": len(calorie_cache),
            "calorie_estimation_coverage": round(estimated_non_null / max(len(calorie_cache), 1), 4),
            "ablation_results": results,
        }
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\n[ok] Saved JSON report -> {args.output_json}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"[ok] Saved CSV table -> {args.output_csv}")


if __name__ == "__main__":
    main()
