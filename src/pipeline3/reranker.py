"""Multi-criteria recipe reranker with constraint filtering and fallback (PRD §§24-25 & Risk 6)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calorie_score import evaluate_calorie_compatibility
from .coverage import PantryStapleWeights, compute_coverage
from .exclusion import ExclusionFilter
from .models import (
    CandidateRecipe,
    RankedRecipe,
    RankingConfig,
    RecommendationResult,
    StructuredQuery,
)
from src.preprocessing.calorie_estimator import extract_servings
from src.preprocessing.ingredient_normalizer import DEFAULT_PROCESSED_DIR, IngredientNormalizer


def _coerce_query(query: StructuredQuery | Mapping[str, Any]) -> StructuredQuery:
    if isinstance(query, StructuredQuery):
        return query
    return StructuredQuery.from_dict(query)


def _coerce_candidate(candidate: CandidateRecipe | Mapping[str, Any]) -> CandidateRecipe:
    if isinstance(candidate, CandidateRecipe):
        return candidate
    return CandidateRecipe.from_dict(candidate)


class MultiCriteriaReranker:
    """Pipeline 3 reranker combining TF-IDF, pantry coverage, calories, and exclusions.

    Final scoring formula (PRD Section 24):
        Score(r, q) = alpha * S_tfidf + beta * S_coverage + gamma * S_calorie - delta * P_missing
    """

    def __init__(
        self,
        config: RankingConfig | None = None,
        processed_dir: Path | None = None,
        normalizer: IngredientNormalizer | None = None,
        staple_weights: PantryStapleWeights | None = None,
        exclusion_filter: ExclusionFilter | None = None,
    ) -> None:
        self.config = config or RankingConfig()
        data_dir = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_DIR
        self.normalizer = normalizer or IngredientNormalizer(data_dir)
        self.staple_weights = staple_weights or PantryStapleWeights.from_csv(
            data_dir / "pantry_staples.csv",
            normalizer=self.normalizer,
        )
        self.exclusion_filter = exclusion_filter or ExclusionFilter(normalizer=self.normalizer)

    def score_candidate(
        self,
        candidate: CandidateRecipe | Mapping[str, Any],
        query: StructuredQuery | Mapping[str, Any],
        config: RankingConfig | None = None,
        apply_exclusion_soft_penalty: bool = False,
    ) -> RankedRecipe:
        """Compute all component scores and the final weighted score for one candidate."""
        cfg = config or self.config
        cand = _coerce_candidate(candidate)
        sq = _coerce_query(query)

        cov = compute_coverage(
            user_ingredients=sq.available_ingredients,
            recipe_ingredients=cand.canonical_ingredients,
            staple_weights=self.staple_weights,
            normalizer=self.normalizer,
            use_weights=cfg.use_pantry_staple_weights,
            use_weighted_missing_penalty=cfg.use_weighted_missing_penalty,
        )

        effective_calories = cand.total_calories
        if (
            cfg.normalize_servings_from_directions
            and effective_calories is not None
            and (cand.directions or cand.title)
        ):
            servings = extract_servings(cand.directions, title=cand.title)
            if servings is not None and servings > 1:
                effective_calories = round(effective_calories / servings, 1)

        cal = evaluate_calorie_compatibility(
            estimated_calories=effective_calories,
            calorie_target=sq.calorie_target,
            calorie_min=sq.calorie_min,
            calorie_max=sq.calorie_max,
            unknown_calorie_score=cfg.unknown_calorie_score,
            no_constraint_score=cfg.no_calorie_constraint_score,
            reject_unknown_on_hard_bound=cfg.reject_unknown_calorie_on_hard_bound,
        )

        excl = self.exclusion_filter.check(
            excluded_ingredients=sq.excluded_ingredients,
            recipe_ingredients=cand.canonical_ingredients,
        )

        s_tfidf = max(0.0, min(1.0, float(cand.tfidf_score)))
        s_cov = cov.coverage_score
        s_cal = cal.score if cfg.enable_calorie_score else 0.0
        p_miss = cov.missing_penalty

        final_score = (
            cfg.alpha * s_tfidf
            + cfg.beta * s_cov
            + (cfg.gamma * s_cal if cfg.enable_calorie_score else 0.0)
            - cfg.delta * p_miss
        )

        if apply_exclusion_soft_penalty and excl.violated:
            final_score -= cfg.fallback_exclusion_penalty * excl.violation_count

        max_pos = cfg.max_positive_score
        if max_pos > 0:
            match_ratio = max(0.0, min(1.0, final_score / max_pos))
        else:
            match_ratio = 0.0
        overall_match_pct = round(match_ratio * 100)

        return RankedRecipe(
            rank=0,
            recipe=cand,
            final_score=round(final_score, 6),
            overall_match_pct=overall_match_pct,
            tfidf_score=s_tfidf,
            coverage=cov,
            calorie=cal,
            exclusion=excl,
        )

    @staticmethod
    def _sort_key(item: RankedRecipe) -> tuple:
        return (
            -item.final_score,
            -item.coverage.coverage_score,
            item.coverage.missing_count,
            -item.tfidf_score,
            str(item.recipe.recipe_id),
        )

    def rerank(
        self,
        candidates: Sequence[CandidateRecipe | Mapping[str, Any]],
        query: StructuredQuery | Mapping[str, Any],
        top_k: int = 3,
        config: RankingConfig | None = None,
    ) -> RecommendationResult:
        """Filter and rerank candidate recipes, applying Risk 6 fallback if needed.

        Accepts either typed ``CandidateRecipe`` / ``StructuredQuery`` objects
        or raw dicts directly from Pipeline 1 (``parse_query``) and Pipeline 2
        (``CandidateRetriever.retrieve``).
        """
        cfg = config or self.config
        sq = _coerce_query(query)
        coerced_candidates = [_coerce_candidate(c) for c in candidates]
        if top_k <= 0 or not coerced_candidates:
            return RecommendationResult(
                query=sq,
                recommendations=[],
                total_candidates=len(coerced_candidates),
            )

        scored_all = [self.score_candidate(c, sq, config=cfg) for c in coerced_candidates]

        strict_matches: list[RankedRecipe] = []
        calorie_rejected: list[RankedRecipe] = []
        exclusion_rejected: list[RankedRecipe] = []

        for item in scored_all:
            if cfg.hard_exclusion_filter and item.exclusion.violated:
                exclusion_rejected.append(item)
                continue
            if cfg.hard_calorie_filter and not item.calorie.passed_hard_filter:
                calorie_rejected.append(item)
                continue
            strict_matches.append(item)

        strict_matches.sort(key=self._sort_key)

        selected: list[RankedRecipe] = list(strict_matches[:top_k])
        is_fallback = False
        relaxed: list[str] = []
        fallback_message: str | None = None

        need_fallback = (
            cfg.enable_fallback
            and (
                len(strict_matches) == 0
                or (cfg.fill_fallback_to_top_k and len(strict_matches) < top_k)
            )
        )

        if need_fallback:
            remaining = top_k - len(selected)

            # Tier 1 fallback: relax calorie hard bounds while keeping exclusions strict
            if remaining > 0 and calorie_rejected:
                calorie_rejected.sort(key=self._sort_key)
                for item in calorie_rejected[:remaining]:
                    selected.append(
                        replace(
                            item,
                            is_fallback_candidate=True,
                            relaxed_constraints=["calorie_bounds"],
                        )
                    )
                if "calorie_bounds" not in relaxed:
                    relaxed.append("calorie_bounds")
                remaining = top_k - len(selected)

            # Tier 2 fallback: only if still short, relax exclusion filter with heavy penalty
            if remaining > 0 and exclusion_rejected:
                penalized = [
                    replace(
                        self.score_candidate(
                            item.recipe,
                            sq,
                            config=cfg,
                            apply_exclusion_soft_penalty=True,
                        ),
                        is_fallback_candidate=True,
                        relaxed_constraints=(
                            ["excluded_ingredients", "calorie_bounds"]
                            if (cfg.hard_calorie_filter and not item.calorie.passed_hard_filter)
                            else ["excluded_ingredients"]
                        ),
                    )
                    for item in exclusion_rejected
                ]
                penalized.sort(key=self._sort_key)
                for item in penalized[:remaining]:
                    selected.append(item)
                    for constraint_name in item.relaxed_constraints:
                        if constraint_name not in relaxed:
                            relaxed.append(constraint_name)

            if relaxed:
                is_fallback = True
                relaxed_str = ", ".join(relaxed)
                if len(strict_matches) == 0:
                    fallback_message = (
                        f"No exact recommendation found. Showing closest alternatives "
                        f"with relaxed constraints: {relaxed_str}."
                    )
                else:
                    fallback_message = (
                        f"Only {len(strict_matches)} exact match(es) found for Top-{top_k}. "
                        f"Remaining slots show closest alternatives with relaxed constraints: {relaxed_str}."
                    )

        ranked_final = [
            replace(item, rank=idx)
            for idx, item in enumerate(selected, start=1)
        ]

        return RecommendationResult(
            query=sq,
            recommendations=ranked_final,
            total_candidates=len(coerced_candidates),
            filtered_out_exclusion=len(exclusion_rejected),
            filtered_out_calorie=len(calorie_rejected),
            strict_match_count=len(strict_matches),
            is_fallback=is_fallback,
            fallback_message=fallback_message,
            relaxed_constraints=relaxed,
        )
