"""Data models and configuration schemas for Pipeline 3 (Multi-Criteria Reranking).

These dataclasses define the clean integration contract between:
- Pipeline 1 (Query & Constraint Understanding) -> ``StructuredQuery``
- Pipeline 2 (Candidate Recipe Retrieval) -> ``CandidateRecipe``
- Pipeline 3 (Multi-Criteria Recipe Reranking) -> ``RecommendationResult``
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


def _finite_float(value: Any) -> float | None:
    """Coerce ``value`` to a finite float, or ``None`` when unusable."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clean_terms(values: Any) -> tuple[str, ...]:
    """Return non-empty string terms from ``values``; never raises on None/mixed input."""
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    elif not isinstance(values, Iterable):
        return ()
    cleaned: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        term = item.strip()
        if term:
            cleaned.append(term)
    return tuple(cleaned)


def _clean_positive_bound(value: Any) -> float | None:
    result = _finite_float(value)
    if result is None or result <= 0:
        return None
    return result


@dataclass(frozen=True)
class StructuredQuery:
    """Structured representation of a user's pantry query (PRD Section 12.1)."""

    available_ingredients: tuple[str, ...]
    excluded_ingredients: tuple[str, ...] = ()
    calorie_target: float | None = None
    calorie_min: float | None = None
    calorie_max: float | None = None
    raw_query: str | None = None

    def __init__(
        self,
        available_ingredients: list[str] | tuple[str, ...] | None,
        excluded_ingredients: list[str] | tuple[str, ...] | None = (),
        calorie_target: float | None = None,
        calorie_min: float | None = None,
        calorie_max: float | None = None,
        raw_query: str | None = None,
    ) -> None:
        target = _clean_positive_bound(calorie_target)
        min_bound = _clean_positive_bound(calorie_min)
        max_bound = _clean_positive_bound(calorie_max)
        if min_bound is not None and max_bound is not None and min_bound > max_bound:
            min_bound, max_bound = max_bound, min_bound

        object.__setattr__(self, "available_ingredients", _clean_terms(available_ingredients))
        object.__setattr__(self, "excluded_ingredients", _clean_terms(excluded_ingredients))
        object.__setattr__(self, "calorie_target", target)
        object.__setattr__(self, "calorie_min", min_bound)
        object.__setattr__(self, "calorie_max", max_bound)
        object.__setattr__(
            self,
            "raw_query",
            None if raw_query is None else str(raw_query),
        )

    @property
    def has_calorie_constraint(self) -> bool:
        return (
            self.calorie_target is not None
            or self.calorie_min is not None
            or self.calorie_max is not None
        )

    @property
    def has_hard_calorie_bound(self) -> bool:
        return self.calorie_min is not None or self.calorie_max is not None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], raw_query: str | None = None) -> StructuredQuery:
        if not isinstance(data, Mapping):
            raise TypeError(
                "StructuredQuery.from_dict() expects a mapping (e.g. dict), "
                f"got {type(data).__name__}"
            )
        return cls(
            available_ingredients=data.get("available_ingredients"),
            excluded_ingredients=data.get("excluded_ingredients"),
            calorie_target=data.get("calorie_target"),
            calorie_min=data.get("calorie_min"),
            calorie_max=data.get("calorie_max"),
            raw_query=raw_query if raw_query is not None else data.get("raw_query"),
        )


@dataclass
class CandidateRecipe:
    """A candidate recipe passed from Pipeline 2 retrieval into Pipeline 3."""

    recipe_id: int | str
    title: str
    canonical_ingredients: list[str]
    tfidf_score: float = 0.0
    ingredients: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    total_calories: float | None = None
    calories_per_serving: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.title = "" if self.title is None else str(self.title)
        self.canonical_ingredients = list(_clean_terms(self.canonical_ingredients))
        self.ingredients = list(_clean_terms(self.ingredients))
        self.directions = list(_clean_terms(self.directions))
        self.tfidf_score = _finite_float(self.tfidf_score) or 0.0
        self.total_calories = _finite_float(self.total_calories)
        self.calories_per_serving = _clean_positive_bound(self.calories_per_serving)
        if not isinstance(self.metadata, Mapping):
            self.metadata = {}
        else:
            self.metadata = dict(self.metadata)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateRecipe:
        """Build a ``CandidateRecipe`` from a Pipeline 2 retriever dict.

        Directly accepts the output schema of ``CandidateRetriever.retrieve()``
        (including top-level ``calorie_coverage``, ``source``, ``link``) and
        treats non-finite calories or unmapped 0-coverage recipes as ``None``.
        """
        metadata = dict(data.get("metadata") or {})
        calorie_coverage: float | None = None
        if data.get("calorie_coverage") is not None:
            try:
                cov_val = float(data["calorie_coverage"])
                if math.isfinite(cov_val):
                    calorie_coverage = cov_val
                    metadata.setdefault("calorie_coverage", cov_val)
            except (TypeError, ValueError):
                pass
        for extra_key in ("source", "link"):
            if extra_key in data and extra_key not in metadata:
                metadata[extra_key] = data[extra_key]

        raw_cal = data.get("total_calories")
        parsed_cal: float | None = None
        if raw_cal is not None:
            try:
                cal_val = float(raw_cal)
                if math.isfinite(cal_val):
                    # Corpus build emits 0.0 when calorie_coverage == 0.0
                    # (no ingredients matched); treat that as unavailable calories.
                    if not (cal_val == 0.0 and calorie_coverage == 0.0):
                        parsed_cal = cal_val
            except (TypeError, ValueError):
                parsed_cal = None

        raw_tfidf = data.get("tfidf_score", 0.0)
        try:
            tfidf_val = float(raw_tfidf)
            if not math.isfinite(tfidf_val):
                tfidf_val = 0.0
        except (TypeError, ValueError):
            tfidf_val = 0.0

        parsed_per_serving = _clean_positive_bound(data.get("calories_per_serving"))

        return cls(
            recipe_id=data["recipe_id"],
            title=data.get("title") or "",
            canonical_ingredients=list(data.get("canonical_ingredients") or []),
            tfidf_score=tfidf_val,
            ingredients=list(data.get("ingredients") or []),
            directions=list(data.get("directions") or []),
            total_calories=parsed_cal,
            calories_per_serving=parsed_per_serving,
            metadata=metadata,
        )


@dataclass(frozen=True)
class RankingConfig:
    """Configurable weights and filter switches for Pipeline 3 (PRD Section 24).

    Initial weights follow PRD Section 24 and are experimental until validated
    through the ablation study (PRD Section 35 / BEAM note).
    """

    alpha: float = 0.30  # TF-IDF similarity
    beta: float = 0.35   # Ingredient coverage
    gamma: float = 0.20  # Calorie compatibility
    delta: float = 0.15  # Missing ingredient penalty

    use_pantry_staple_weights: bool = True
    use_weighted_missing_penalty: bool = True
    hard_calorie_filter: bool = True
    hard_exclusion_filter: bool = True
    enable_calorie_score: bool = True

    enable_fallback: bool = True
    fill_fallback_to_top_k: bool = True

    unknown_calorie_score: float = 0.50
    no_calorie_constraint_score: float = 1.0
    reject_unknown_calorie_on_hard_bound: bool = True
    normalize_servings_from_directions: bool = False

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma", "delta"):
            value = getattr(self, name)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"RankingConfig.{name} must be a finite number >= 0, got {value!r}") from None
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"RankingConfig.{name} must be a finite number >= 0, got {value!r}")
            object.__setattr__(self, name, numeric)

    @property
    def max_positive_score(self) -> float:
        return self.alpha + self.beta + (self.gamma if self.enable_calorie_score else 0.0)

    @classmethod
    def ablation_model_a(cls) -> RankingConfig:
        """Model A (Baseline): TF-IDF cosine similarity only (PRD Section 35)."""
        return cls(
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            delta=0.0,
            use_pantry_staple_weights=False,
            use_weighted_missing_penalty=False,
            hard_calorie_filter=False,
            hard_exclusion_filter=False,
            enable_calorie_score=False,
            enable_fallback=False,
        )

    @classmethod
    def ablation_model_b(cls) -> RankingConfig:
        """Model B: TF-IDF + Ingredient Coverage (PRD Section 35)."""
        return cls(
            alpha=0.30,
            beta=0.35,
            gamma=0.0,
            delta=0.0,
            use_pantry_staple_weights=True,
            use_weighted_missing_penalty=False,
            hard_calorie_filter=False,
            hard_exclusion_filter=False,
            enable_calorie_score=False,
            enable_fallback=False,
        )

    @classmethod
    def ablation_model_c(cls) -> RankingConfig:
        """Model C: TF-IDF + Coverage + Missing Ingredient Penalty (PRD Section 35)."""
        return cls(
            alpha=0.30,
            beta=0.35,
            gamma=0.0,
            delta=0.15,
            use_pantry_staple_weights=True,
            use_weighted_missing_penalty=True,
            hard_calorie_filter=False,
            hard_exclusion_filter=False,
            enable_calorie_score=False,
            enable_fallback=False,
        )

    @classmethod
    def ablation_model_d(cls) -> RankingConfig:
        """Model D (Full Proposed): All criteria + Calorie & Exclusion filters (PRD Section 35)."""
        return cls()

    @classmethod
    def ablation_unweighted(cls) -> RankingConfig:
        """Full Proposed variant without pantry staple down-weighting (data/README.md check)."""
        return cls(
            use_pantry_staple_weights=False,
            use_weighted_missing_penalty=False,
        )


@dataclass
class CoverageBreakdown:
    """Detailed ingredient coverage and missing ingredient metrics (PRD Sections 19-21)."""

    coverage_score: float
    missing_penalty: float
    weighted_coverage: float
    unweighted_coverage: float
    weighted_missing_penalty: float
    unweighted_missing_penalty: float
    available_matched: list[str]
    missing_ingredients: list[str]
    missing_non_staples: list[str]
    missing_staples: list[str]
    recipe_ingredients: list[str]
    ingredient_weights: dict[str, float]
    parse_confidence: float = 1.0

    @property
    def missing_count(self) -> int:
        return len(self.missing_ingredients)

    @property
    def missing_non_staple_count(self) -> int:
        return len(self.missing_non_staples)


@dataclass
class CalorieBreakdown:
    """Detailed calorie compatibility assessment (PRD Section 22)."""

    score: float
    estimated_calories: float | None
    calorie_distance: float | None
    passed_hard_filter: bool
    constraint_type: str  # "none" | "target" | "max" | "min" | "range" | "target+bounds"
    status_label: str


@dataclass
class ExclusionBreakdown:
    """Detailed excluded ingredient check (PRD Section 23)."""

    violated: bool
    violation_count: int
    violated_ingredients: list[str]
    matched_rules: dict[str, list[str]]  # excluded_term -> matched recipe ingredients


@dataclass
class RankedRecipe:
    """A scored, explainable recipe recommendation produced by Pipeline 3."""

    rank: int
    recipe: CandidateRecipe
    final_score: float
    overall_match_pct: int
    tfidf_score: float
    coverage: CoverageBreakdown
    calorie: CalorieBreakdown
    exclusion: ExclusionBreakdown
    is_fallback_candidate: bool = False
    relaxed_constraints: list[str] = field(default_factory=list)


@dataclass
class RecommendationResult:
    """Top-K reranking output including filtering telemetry and fallback status."""

    query: StructuredQuery
    recommendations: list[RankedRecipe]
    total_candidates: int
    filtered_out_exclusion: int = 0
    filtered_out_calorie: int = 0
    strict_match_count: int = 0
    is_fallback: bool = False
    fallback_message: str | None = None
    relaxed_constraints: list[str] = field(default_factory=list)
    no_match_reason: str | None = None
