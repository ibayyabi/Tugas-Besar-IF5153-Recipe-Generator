"""Calorie compatibility scoring and hard-bound checking (PRD Section 22)."""
from __future__ import annotations

import math

from .models import CalorieBreakdown


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_or_none(value: float | None) -> float | None:
    result = _finite(value)
    return result if result is not None and result > 0 else None


def evaluate_calorie_compatibility(
    estimated_calories: float | None,
    calorie_target: float | None = None,
    calorie_min: float | None = None,
    calorie_max: float | None = None,
    unknown_calorie_score: float = 0.50,
    no_constraint_score: float = 1.0,
    reject_unknown_on_hard_bound: bool = True,
) -> CalorieBreakdown:
    """Evaluate how well a recipe's estimated calories match user calorie constraints.

    Supported constraint types (PRD §§12.6 & 22):
    - ``none``   : no target/min/max specified -> neutral score (``no_constraint_score``)
    - ``target`` : approximate target C_t -> ``max(0, 1 - |C_r - C_t| / C_t)`` (soft)
    - ``max``    : upper bound C_r <= C_max (hard filter + proximity decay if relaxed)
    - ``min``    : lower bound C_r >= C_min (hard filter + proximity decay if relaxed)
    - ``range``  : C_min <= C_r <= C_max (hard filter + proximity decay if relaxed)
    - ``target+bounds``: combined target proximity with explicit min/max hard bounds
    """
    estimated_calories = _finite(estimated_calories)
    calorie_target = _positive_or_none(calorie_target)
    calorie_min = _positive_or_none(calorie_min)
    calorie_max = _positive_or_none(calorie_max)
    if calorie_min is not None and calorie_max is not None and calorie_min > calorie_max:
        calorie_min, calorie_max = calorie_max, calorie_min

    has_target = calorie_target is not None
    has_min = calorie_min is not None
    has_max = calorie_max is not None

    if not (has_target or has_min or has_max):
        return CalorieBreakdown(
            score=no_constraint_score,
            estimated_calories=estimated_calories,
            calorie_distance=None,
            passed_hard_filter=True,
            constraint_type="none",
            status_label="no calorie constraint",
        )

    if has_target and (has_min or has_max):
        constraint_type = "target+bounds"
    elif has_target:
        constraint_type = "target"
    elif has_min and has_max:
        constraint_type = "range"
    elif has_max:
        constraint_type = "max"
    else:
        constraint_type = "min"

    if estimated_calories is None:
        has_hard_bound = has_min or has_max
        passed = not (has_hard_bound and reject_unknown_on_hard_bound)
        return CalorieBreakdown(
            score=unknown_calorie_score,
            estimated_calories=None,
            calorie_distance=None,
            passed_hard_filter=passed,
            constraint_type=constraint_type,
            status_label="estimated calories unavailable",
        )

    cal = float(estimated_calories)

    # Check hard bounds when present
    passed_min = True if not has_min else cal >= float(calorie_min)
    passed_max = True if not has_max else cal <= float(calorie_max)
    passed_hard = passed_min and passed_max

    if constraint_type == "target":
        target = float(calorie_target)
        distance = abs(cal - target)
        score = max(0.0, 1.0 - distance / target)
        label = f"approximately {round(target)} kcal (est. {round(cal)} kcal)"
        return CalorieBreakdown(
            score=score,
            estimated_calories=cal,
            calorie_distance=distance,
            passed_hard_filter=True,
            constraint_type=constraint_type,
            status_label=label,
        )

    if constraint_type == "target+bounds":
        target = float(calorie_target)
        distance = abs(cal - target)
        score = max(0.0, 1.0 - distance / target)
        bounds_desc = []
        if has_min:
            bounds_desc.append(f"min {round(float(calorie_min))}")
        if has_max:
            bounds_desc.append(f"max {round(float(calorie_max))}")
        label = f"around {round(target)} kcal ({', '.join(bounds_desc)} kcal; est. {round(cal)} kcal)"
        return CalorieBreakdown(
            score=score,
            estimated_calories=cal,
            calorie_distance=distance,
            passed_hard_filter=passed_hard,
            constraint_type=constraint_type,
            status_label=label,
        )

    if constraint_type == "range":
        cmin = float(calorie_min)
        cmax = float(calorie_max)
        if cmin <= cal <= cmax:
            distance = 0.0
            score = 1.0
        elif cal < cmin:
            distance = cmin - cal
            ref = max((cmin + cmax) / 2.0, 1.0)
            score = max(0.0, 1.0 - distance / ref)
        else:
            distance = cal - cmax
            ref = max((cmin + cmax) / 2.0, 1.0)
            score = max(0.0, 1.0 - distance / ref)
        label = f"between {round(cmin)} and {round(cmax)} kcal (est. {round(cal)} kcal)"
        return CalorieBreakdown(
            score=score,
            estimated_calories=cal,
            calorie_distance=distance,
            passed_hard_filter=passed_hard,
            constraint_type=constraint_type,
            status_label=label,
        )

    if constraint_type == "max":
        cmax = float(calorie_max)
        if cal <= cmax:
            distance = 0.0
            score = 1.0
        else:
            distance = cal - cmax
            score = max(0.0, 1.0 - distance / max(cmax, 1.0))
        label = f"under {round(cmax)} kcal (est. {round(cal)} kcal)"
        return CalorieBreakdown(
            score=score,
            estimated_calories=cal,
            calorie_distance=distance,
            passed_hard_filter=passed_hard,
            constraint_type=constraint_type,
            status_label=label,
        )

    # constraint_type == "min"
    cmin = float(calorie_min)
    if cal >= cmin:
        distance = 0.0
        score = 1.0
    else:
        distance = cmin - cal
        score = max(0.0, 1.0 - distance / max(cmin, 1.0))
    label = f"at least {round(cmin)} kcal (est. {round(cal)} kcal)"
    return CalorieBreakdown(
        score=score,
        estimated_calories=cal,
        calorie_distance=distance,
        passed_hard_filter=passed_hard,
        constraint_type=constraint_type,
        status_label=label,
    )
