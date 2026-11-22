"""Ingredient coverage, missing ingredient penalty, and pantry staple weights (PRD §§19-21)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from .models import CoverageBreakdown
from src.preprocessing.ingredient_normalizer import (
    DEFAULT_PROCESSED_DIR,
    IngredientNormalizer,
    normalize_ingredient,
    read_csv_rows,
)


class PantryStapleWeights:
    """Lookup table for pantry staple weights loaded from ``pantry_staples.csv`` (PRD §21).

    Common pantry staples (e.g. water=0.05, salt=0.15, pepper=0.15) receive lower
    weights so they do not disproportionately penalize recipes when omitted from
    the user's query. Ingredients absent from the table default to ``1.0``.
    """

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        default_weight: float = 1.0,
        normalizer: IngredientNormalizer | None = None,
    ) -> None:
        self.default_weight = float(default_weight)
        self.normalizer = normalizer
        self.weights: dict[str, float] = {}
        if weights:
            for key, value in weights.items():
                norm = normalize_ingredient(key)
                if norm:
                    self.weights[norm] = float(value)
                if normalizer is not None:
                    canon = normalizer.canonicalize(key, drop_qualifiers=False)
                    if canon and canon not in self.weights:
                        self.weights[canon] = float(value)

    @classmethod
    def from_csv(
        cls,
        path: Path | None = None,
        normalizer: IngredientNormalizer | None = None,
        default_weight: float = 1.0,
    ) -> PantryStapleWeights:
        csv_path = Path(path) if path is not None else DEFAULT_PROCESSED_DIR / "pantry_staples.csv"
        weights: dict[str, float] = {}
        if csv_path.exists():
            for row in read_csv_rows(csv_path):
                ingredient = row.get("ingredient", "").strip()
                weight_str = row.get("weight", "").strip()
                if not ingredient or not weight_str:
                    continue
                try:
                    w = float(weight_str)
                except ValueError:
                    continue
                norm = normalize_ingredient(ingredient)
                if norm:
                    weights[norm] = w
        return cls(weights=weights, default_weight=default_weight, normalizer=normalizer)

    def get_weight(self, ingredient: str) -> float:
        norm = normalize_ingredient(ingredient)
        if not norm:
            return self.default_weight
        if norm in self.weights:
            return self.weights[norm]
        if self.normalizer is not None:
            canon = self.normalizer.canonicalize(norm, drop_qualifiers=False)
            if canon in self.weights:
                return self.weights[canon]
        return self.default_weight

    def is_staple(self, ingredient: str, threshold: float = 0.5) -> bool:
        """Return True if an ingredient's weight is below ``threshold``."""
        return self.get_weight(ingredient) < threshold


def compute_coverage(
    user_ingredients: Iterable[str],
    recipe_ingredients: Iterable[str],
    staple_weights: PantryStapleWeights | None = None,
    normalizer: IngredientNormalizer | None = None,
    use_weights: bool = True,
    use_weighted_missing_penalty: bool = True,
    staple_threshold: float = 0.5,
) -> CoverageBreakdown:
    """Compute ingredient coverage and missing ingredient penalty (PRD §§19-21).

    Unweighted formulas (PRD §§19-20):
        S_coverage_unweighted = |U ∩ R| / |R|
        P_missing_unweighted  = |R - U| / |R|

    Weighted formulas with pantry staples (PRD §21):
        S_coverage_weighted   = sum(w_i for i in U ∩ R) / sum(w_i for i in R)
        P_missing_weighted    = sum(w_i for i in R - U) / sum(w_i for i in R)
    """
    if normalizer is not None:
        user_list = normalizer.canonicalize_list(user_ingredients)
        recipe_list = normalizer.canonicalize_list(recipe_ingredients)
    else:
        user_list = list(dict.fromkeys(n for i in user_ingredients if (n := normalize_ingredient(i))))
        recipe_list = list(dict.fromkeys(n for i in recipe_ingredients if (n := normalize_ingredient(i))))

    user_set = set(user_list)

    if not recipe_list:
        return CoverageBreakdown(
            coverage_score=0.0,
            missing_penalty=0.0,
            weighted_coverage=0.0,
            unweighted_coverage=0.0,
            weighted_missing_penalty=0.0,
            unweighted_missing_penalty=0.0,
            available_matched=[],
            missing_ingredients=[],
            missing_non_staples=[],
            missing_staples=[],
            recipe_ingredients=[],
            ingredient_weights={},
        )

    available_matched = [ing for ing in recipe_list if ing in user_set]
    missing_ingredients = [ing for ing in recipe_list if ing not in user_set]

    unweighted_coverage = len(available_matched) / len(recipe_list)
    unweighted_missing_penalty = len(missing_ingredients) / len(recipe_list)

    weights_map: dict[str, float] = {}
    for ing in recipe_list:
        weights_map[ing] = staple_weights.get_weight(ing) if staple_weights is not None else 1.0

    total_weight = sum(weights_map.values())
    if total_weight > 0:
        matched_weight = sum(weights_map[ing] for ing in available_matched)
        missing_weight = sum(weights_map[ing] for ing in missing_ingredients)
        weighted_coverage = matched_weight / total_weight
        weighted_missing_penalty = missing_weight / total_weight
    else:
        weighted_coverage = unweighted_coverage
        weighted_missing_penalty = unweighted_missing_penalty

    missing_non_staples = [ing for ing in missing_ingredients if weights_map[ing] >= staple_threshold]
    missing_staples = [ing for ing in missing_ingredients if weights_map[ing] < staple_threshold]

    coverage_score = weighted_coverage if use_weights else unweighted_coverage
    missing_penalty = (
        weighted_missing_penalty
        if (use_weights and use_weighted_missing_penalty)
        else unweighted_missing_penalty
    )

    return CoverageBreakdown(
        coverage_score=coverage_score,
        missing_penalty=missing_penalty,
        weighted_coverage=weighted_coverage,
        unweighted_coverage=unweighted_coverage,
        weighted_missing_penalty=weighted_missing_penalty,
        unweighted_missing_penalty=unweighted_missing_penalty,
        available_matched=available_matched,
        missing_ingredients=missing_ingredients,
        missing_non_staples=missing_non_staples,
        missing_staples=missing_staples,
        recipe_ingredients=recipe_list,
        ingredient_weights=weights_map,
    )
