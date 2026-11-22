"""Pipeline 3 — Pantry-Aware Multi-Criteria Recipe Reranking (PRD §§18-27)."""

from .calorie_score import evaluate_calorie_compatibility
from .coverage import PantryStapleWeights, compute_coverage
from .exclusion import ExclusionFilter
from .models import (
    CalorieBreakdown,
    CandidateRecipe,
    CoverageBreakdown,
    ExclusionBreakdown,
    RankedRecipe,
    RankingConfig,
    RecommendationResult,
    StructuredQuery,
)
from .recommender import format_recommendation_card, format_recommendations, recommend, to_dict
from .reranker import MultiCriteriaReranker
from src.preprocessing.calorie_estimator import (
    CalorieEstimate,
    IngredientCalorieItem,
    ParsedIngredient,
    RecipeCalorieEstimator,
    extract_servings,
    parse_ingredient_line,
)
from src.preprocessing.ingredient_normalizer import IngredientNormalizer, normalize_ingredient, singularize

__all__ = [
    "CalorieBreakdown",
    "CalorieEstimate",
    "CandidateRecipe",
    "CoverageBreakdown",
    "ExclusionBreakdown",
    "ExclusionFilter",
    "IngredientCalorieItem",
    "IngredientNormalizer",
    "MultiCriteriaReranker",
    "PantryStapleWeights",
    "ParsedIngredient",
    "RankedRecipe",
    "RankingConfig",
    "RecipeCalorieEstimator",
    "RecommendationResult",
    "StructuredQuery",
    "compute_coverage",
    "evaluate_calorie_compatibility",
    "extract_servings",
    "format_recommendation_card",
    "format_recommendations",
    "normalize_ingredient",
    "parse_ingredient_line",
    "recommend",
    "singularize",
    "to_dict",
]
