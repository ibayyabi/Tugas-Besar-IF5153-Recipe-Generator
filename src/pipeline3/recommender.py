"""Explainable recommendation entry point, formatting, and serialization (PRD §§26-27)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    CandidateRecipe,
    RankedRecipe,
    RankingConfig,
    RecommendationResult,
    StructuredQuery,
)
from .reranker import MultiCriteriaReranker


def recommend(
    candidates: Sequence[CandidateRecipe | Mapping[str, Any]],
    query: StructuredQuery | Mapping[str, Any],
    top_k: int = 3,
    config: RankingConfig | None = None,
    reranker: MultiCriteriaReranker | None = None,
    processed_dir: Path | None = None,
) -> RecommendationResult:
    """High-level Pipeline 3 entry point accepting raw Pipeline 1 & 2 dicts or typed models."""
    engine = reranker or MultiCriteriaReranker(config=config, processed_dir=processed_dir)
    return engine.rerank(candidates=candidates, query=query, top_k=top_k, config=config)


def format_recommendation_card(
    ranked: RankedRecipe,
    excluded_ingredients: Iterable[str] = (),
    include_ingredients: bool = True,
    include_directions: bool = True,
) -> str:
    """Format a single ranked recipe as an explainable text card (PRD Section 26)."""
    recipe = ranked.recipe
    cov = ranked.coverage
    cal = ranked.calorie
    excl = ranked.exclusion

    cov_pct = round(cov.coverage_score * 100)
    cal_pct = round(cal.score * 100)
    cal_str = f"{round(cal.estimated_calories)} kcal" if cal.estimated_calories is not None else "N/A (not estimated)"

    header = f"#{ranked.rank} {recipe.title}"
    if ranked.is_fallback_candidate:
        header += f" [Closest Alternative — relaxed: {', '.join(ranked.relaxed_constraints)}]"

    lines = [
        header,
        f"Overall Match       : {ranked.overall_match_pct}% (score={ranked.final_score:.3f})",
        f"Ingredient Coverage : {cov_pct}% ({len(cov.available_matched)}/{len(cov.recipe_ingredients)} ingredients)",
        f"Estimated Calories  : {cal_str}",
    ]
    if cov.parse_confidence < 1.0:
        lines.append(
            f"Ingredient Parsing  : {round(cov.parse_confidence * 100)}% confidence "
            f"({len(cov.recipe_ingredients)} usable ingredient(s))"
        )
    if cal.constraint_type != "none":
        lines.append(f"Calorie Match       : {cal_pct}%")

    lines.append("")
    lines.append("YOU HAVE:")
    if cov.available_matched:
        for ing in cov.available_matched:
            lines.append(f"  ✓ {ing}")
    else:
        lines.append("  (none matched)")

    lines.append("")
    lines.append("MISSING:")
    if cov.missing_ingredients:
        for ing in cov.missing_non_staples:
            lines.append(f"  ✗ {ing}")
        for ing in cov.missing_staples:
            lines.append(f"  ✗ {ing} (pantry staple)")
    else:
        lines.append("  ✓ nothing missing!")

    lines.append("")
    lines.append("CONSTRAINTS:")
    if cal.constraint_type == "none":
        lines.append("  ✓ no calorie constraint specified")
    elif cal.passed_hard_filter:
        lines.append(f"  ✓ {cal.status_label}")
    else:
        lines.append(f"  ✗ relaxed calorie bound: {cal.status_label}")

    excl_list = [e for e in excluded_ingredients if e]
    if excl.violated:
        lines.append(f"  ✗ excluded ingredient detected: {', '.join(excl.violated_ingredients)}")
    elif excl_list:
        lines.append(f"  ✓ no {', '.join(excl_list)} detected")
    else:
        lines.append("  ✓ no excluded ingredients detected")

    if include_ingredients and recipe.ingredients:
        lines.append("")
        lines.append("INGREDIENTS:")
        for raw_ing in recipe.ingredients:
            lines.append(f"  - {raw_ing}")

    if include_directions and recipe.directions:
        lines.append("")
        lines.append("DIRECTIONS:")
        for idx, step in enumerate(recipe.directions, start=1):
            lines.append(f"  {idx}. {step}")

    return "\n".join(lines)


def format_recommendations(
    result: RecommendationResult,
    include_ingredients: bool = True,
    include_directions: bool = True,
) -> str:
    """Render a full ``RecommendationResult`` into human-readable explainable text."""
    blocks: list[str] = []
    if result.is_fallback and result.fallback_message:
        blocks.append(f"[NOTICE] {result.fallback_message}")

    if not result.recommendations:
        blocks.append(result.no_match_reason or "No exact recommendation found.")
        return "\n\n".join(blocks)

    for ranked in result.recommendations:
        blocks.append(
            format_recommendation_card(
                ranked,
                excluded_ingredients=result.query.excluded_ingredients,
                include_ingredients=include_ingredients,
                include_directions=include_directions,
            )
        )
    return "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join(blocks)


def to_dict(result: RecommendationResult) -> dict[str, Any]:
    """Serialize ``RecommendationResult`` into a JSON-friendly explainable dict."""
    return {
        "query": {
            "raw_query": result.query.raw_query,
            "available_ingredients": list(result.query.available_ingredients),
            "excluded_ingredients": list(result.query.excluded_ingredients),
            "calorie_target": result.query.calorie_target,
            "calorie_min": result.query.calorie_min,
            "calorie_max": result.query.calorie_max,
        },
        "total_candidates": result.total_candidates,
        "strict_match_count": result.strict_match_count,
        "filtered_out_exclusion": result.filtered_out_exclusion,
        "filtered_out_calorie": result.filtered_out_calorie,
        "is_fallback": result.is_fallback,
        "fallback_message": result.fallback_message,
        "relaxed_constraints": list(result.relaxed_constraints),
        "no_match_reason": result.no_match_reason,
        "recommendations": [
            {
                "rank": item.rank,
                "recipe_id": item.recipe.recipe_id,
                "title": item.recipe.title,
                "final_score": item.final_score,
                "overall_match_pct": item.overall_match_pct,
                "tfidf_score": round(item.tfidf_score, 4),
                "coverage_score": round(item.coverage.coverage_score, 4),
                "weighted_coverage": round(item.coverage.weighted_coverage, 4),
                "unweighted_coverage": round(item.coverage.unweighted_coverage, 4),
                "missing_penalty": round(item.coverage.missing_penalty, 4),
                "parse_confidence": round(item.coverage.parse_confidence, 4),
                "available_matched": item.coverage.available_matched,
                "missing_ingredients": item.coverage.missing_ingredients,
                "missing_non_staples": item.coverage.missing_non_staples,
                "missing_staples": item.coverage.missing_staples,
                "estimated_calories": item.calorie.estimated_calories,
                "calorie_coverage": item.recipe.metadata.get("calorie_coverage"),
                "calorie_score": round(item.calorie.score, 4),
                "calorie_distance": item.calorie.calorie_distance,
                "calorie_status": item.calorie.status_label,
                "exclusion_violated": item.exclusion.violated,
                "violated_ingredients": item.exclusion.violated_ingredients,
                "is_fallback_candidate": item.is_fallback_candidate,
                "relaxed_constraints": item.relaxed_constraints,
                "ingredients": item.recipe.ingredients,
                "directions": item.recipe.directions,
            }
            for item in result.recommendations
        ],
    }
