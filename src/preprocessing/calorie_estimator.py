"""Recipe calorie estimator (PRD Sections 14-16) — the single one in the repo.

Merges the former ``CalorieMapper`` (dict API) and
``RecipeCalorieEstimator`` (typed API used by Pipeline 3, evaluation, UI)
into one engine over the committed tables in ``data/processed/``:

- ``calorie_baseline.csv``         (deterministic canonical kcal/100g + fdc_id)
- ``supplement_calories.csv``      (fallback kcal/100g)
- ``zero_calorie_ingredients.csv`` (zero/negligible calorie items)
- ``calorie_reference.csv``        (7,793 USDA SR Legacy foods for fallback matching)
- ``unit_grams_reference.csv``     (per-fdc_id household unit -> grams)
- ``unit_grams_fallback.csv``      (generic unit -> grams)

Lookup priority: zero -> baseline -> reference -> supplement -> fuzzy.
Reference rows collapse to the median kcal per base ingredient (deterministic).
Values returned are *Estimated Calories*, not authoritative nutritional
labels (PRD Risk 2).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.preprocessing.ingredient_normalizer import (
    DEFAULT_PROCESSED_DIR,
    IngredientNormalizer,
    normalize_ingredient,
    normalize_text,
    read_csv_rows,
    singularize,
)
from src.preprocessing.ingredient_parser import parse_ingredient

# Scaled volume units folded to cups before the gram tables are consulted.
VOLUME_TO_CUPS = {
    "pt": 2.0,
    "qt": 4.0,
    "gal": 16.0,
}

# Bare-count defaults when no food-specific or fallback unit row exists.
DEFAULT_ITEM_GRAMS = {
    "small": 75.0,
    "medium": 110.0,
    "large": 150.0,
    "whole": 100.0,
}

_SERVINGS_PATTERNS = [
    re.compile(r"\bserves\s+(\d+)(?:\s*(?:-|to)\s*(\d+))?\b", re.IGNORECASE),
    re.compile(r"\bmakes\s+(\d+)(?:\s*(?:-|to)\s*(\d+))?\s+(?:servings?|portions?)\b", re.IGNORECASE),
    re.compile(r"\byields?\s*:?\s*(\d+)(?:\s*(?:-|to)\s*(\d+))?\s*(?:servings?|portions?)?\b", re.IGNORECASE),
    re.compile(r"\b(\d+)(?:\s*(?:-|to)\s*(\d+))?\s+servings?\b", re.IGNORECASE),
]


@dataclass
class ParsedIngredient:
    """Structured parse of a raw RecipeNLG ingredient string (PRD Section 14)."""

    raw: str
    quantity: float | None
    unit: str | None
    ingredient: str


def parse_ingredient_line(raw_line: str) -> ParsedIngredient:
    """Parse a raw ingredient line into the typed ``ParsedIngredient`` shape."""
    parsed = parse_ingredient(raw_line)
    return ParsedIngredient(
        raw=str(parsed["raw"]),
        quantity=parsed["quantity"],
        unit=parsed["unit"],
        ingredient=str(parsed["ingredient"] or ""),
    )


@dataclass
class IngredientCalorieItem:
    """Per-ingredient calorie estimation result."""

    raw: str
    quantity: float | None
    unit: str | None
    ingredient: str
    canonical: str | None
    grams: float | None
    kcal_per_100g: float | None
    calories: float | None
    match_type: str  # zero | baseline | reference | supplement | fuzzy | unmatched


@dataclass
class CalorieEstimate:
    """Estimated calorie summary for a recipe (PRD Section 16)."""

    total_calories: float | None
    calories_per_serving: float | None
    estimated_servings: int
    servings_source: str  # parsed | heuristic | single
    matched_count: int
    total_count: int
    items: list[IngredientCalorieItem] = field(default_factory=list)

    @property
    def match_ratio(self) -> float:
        return self.matched_count / self.total_count if self.total_count else 0.0


def aligned_hint(raw_line: str, hint: object) -> str | None:
    """Return ``hint`` only when it literally occurs in ``raw_line``.

    RecipeNLG's NER list drifts out of positional alignment with the raw
    ingredient lines (1997/10000 sample rows differ in length), so an
    index-matched hint can describe a different ingredient: raw
    ``KRAFT Macaroni & Cheese Dinner`` at index 0 against NER
    ``rotisserie chicken``. Trust the hint only when its normalized tokens
    form a contiguous token subsequence (whole phrase included) of the
    normalized line; otherwise ignore it.
    """
    if not isinstance(hint, str) or not hint.strip():
        return None
    # Compare token-wise with every token singularized, so plural hints mid-line
    # ("potatoes" vs "potato") still align. Whole-phrase equality is the
    # width-equals-line case of the same subsequence check.
    hint_tokens = [singularize(t) for t in normalize_text(hint).split()]
    line_tokens = [singularize(t) for t in normalize_text(raw_line).split()]
    if not hint_tokens or not line_tokens:
        return None
    width = len(hint_tokens)
    for start in range(len(line_tokens) - width + 1):
        if line_tokens[start : start + width] == hint_tokens:
            return hint
    return None


def extract_servings(directions: Iterable[str], title: str = "") -> int | None:
    """Extract stated serving count from recipe directions/title when present."""
    combined = " ".join([title, *directions])
    for pattern in _SERVINGS_PATTERNS:
        m = pattern.search(combined)
        if m:
            n1 = int(m.group(1))
            n2 = int(m.group(2)) if m.group(2) else n1
            servings = round((n1 + n2) / 2.0)
            if 1 <= servings <= 50:
                return servings
    return None


class RecipeCalorieEstimator:
    """Estimates recipe calories from raw ingredient lines using data/processed/ tables.

    Exposes both historical entry points:

    - typed:  :meth:`estimate_recipe` -> ``CalorieEstimate`` (Pipeline 3, UI, evaluation)
    - dict:  :meth:`estimate_recipe_calories` / :meth:`estimate_ingredient_calories`
      (legacy raw-line helpers kept for evaluation scripts)
    """

    def __init__(
        self,
        processed_dir: Path | None = None,
        normalizer: IngredientNormalizer | None = None,
        enable_fuzzy: bool = True,
        fuzzy_cutoff: float = 0.84,
    ) -> None:
        self.processed_dir = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_DIR
        self.normalizer = normalizer or IngredientNormalizer(self.processed_dir)
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_cutoff = fuzzy_cutoff

        self.baseline: dict[str, tuple[float, str | None]] = {}
        self.reference: dict[str, tuple[float, str | None]] = {}
        self.supplement: dict[str, tuple[float, str | None]] = {}
        self.zero_calorie: set[str] = set()
        self.unit_grams_by_fdc: dict[tuple[str, str], float] = {}
        self.unit_grams_by_ingredient: dict[tuple[str, str], float] = {}
        self.unit_grams_fallback: dict[str, float] = {}

        self._fuzzy_keys: list[str] = []
        self._lookup_cache: dict[str, dict[str, float | str | None] | None] = {}
        self._load_tables()

    def _load_tables(self) -> None:
        processed = self.processed_dir
        path = processed / "calorie_baseline.csv"
        if path.exists():
            for row in read_csv_rows(path):
                key = normalize_ingredient(row.get("ingredient", ""))
                if key:
                    self.baseline[key] = (float(row["kcal_per_100g"]), row.get("fdc_id") or None)

        path = processed / "calorie_reference.csv"
        if path.exists():
            # One kcal per base ingredient: median over its USDA variants
            # (robust against outlier descriptions; deterministic).
            buckets: dict[str, list[tuple[float, str]]] = {}
            for row in read_csv_rows(path):
                key = normalize_ingredient(row.get("base_ingredient", ""))
                if not key:
                    continue
                try:
                    kcal = float(row["kcal_per_100g"])
                except (ValueError, KeyError):
                    continue
                buckets.setdefault(key, []).append((kcal, row.get("fdc_id") or None))
            for key, values in buckets.items():
                values.sort(key=lambda v: v[0])
                self.reference[key] = values[len(values) // 2]

        path = processed / "supplement_calories.csv"
        if path.exists():
            for row in read_csv_rows(path):
                key = normalize_ingredient(row.get("ingredient", ""))
                if key:
                    self.supplement[key] = (float(row["kcal_per_100g"]), None)

        path = processed / "zero_calorie_ingredients.csv"
        if path.exists():
            self.zero_calorie = {
                normalize_ingredient(row.get("ingredient", "")) for row in read_csv_rows(path)
            } - {""}

        path = processed / "unit_grams_reference.csv"
        if path.exists():
            for row in read_csv_rows(path):
                fdc_id = row.get("fdc_id", "")
                ing = normalize_ingredient(row.get("ingredient", ""))
                unit = row.get("unit", "").strip().lower()
                try:
                    grams = float(row["grams_per_unit"])
                except (ValueError, KeyError):
                    continue
                if fdc_id and unit:
                    self.unit_grams_by_fdc[(fdc_id, unit)] = grams
                if ing and unit:
                    self.unit_grams_by_ingredient.setdefault((ing, unit), grams)

        path = processed / "unit_grams_fallback.csv"
        if path.exists():
            for row in read_csv_rows(path):
                unit = row.get("unit", "").strip().lower()
                try:
                    grams = float(row["grams"])
                except (ValueError, KeyError):
                    continue
                if unit:
                    self.unit_grams_fallback[unit] = grams

        self._fuzzy_keys = sorted(set(self.baseline) | set(self.supplement) | set(self.reference))

    def lookup_calorie_per_100g(self, ingredient: str) -> dict[str, float | str | None] | None:
        """Match an ingredient phrase to kcal/100g; ``None`` when unmatched.

        Candidate keys: alias-mapped full form, raw normalized form, then
        qualifier-dropped suffixes (each alias-mapped too). Tables are consulted
        in priority order zero -> baseline -> supplement -> reference, then the
        longest known contiguous n-gram of the name (up to 3 tokens), then a
        fuzzy match across all three kcal tables.
        """
        norm = normalize_ingredient(ingredient)
        if not norm:
            return None
        if norm in self._lookup_cache:
            return self._lookup_cache[norm]
        result = self._lookup_uncached(norm)
        self._lookup_cache[norm] = result
        return result

    def _lookup_uncached(self, norm: str) -> dict[str, float | str | None] | None:
        aliases = self.normalizer.aliases
        candidates: list[str] = []
        for cand in (aliases.get(norm, norm), norm):
            if cand not in candidates:
                candidates.append(cand)
        tokens = norm.split()
        for start in range(1, len(tokens)):
            sub = " ".join(tokens[start:])
            for cand in (aliases.get(sub, sub), sub):
                if cand not in candidates:
                    candidates.append(cand)

        # Curated tables in ascending scope, then the generic USDA reference.
        for table, match_type in (
            (self.zero_calorie, "zero"),
            (self.baseline, "baseline"),
            (self.supplement, "supplement"),
            (self.reference, "reference"),
        ):
            for cand in candidates:
                if cand in table:
                    kcal, fdc_id = (0.0, None) if match_type == "zero" else table[cand]
                    return {
                        "canonical": cand,
                        "kcal_per_100g": kcal,
                        "fdc_id": fdc_id,
                        "match_type": match_type,
                    }

        # Conservative substring fallback: contiguous token n-grams of the
        # normalized name (up to 3 tokens, longest first) that exist in a known
        # table, e.g. "kraft macaroni cheese dinner" -> "macaroni". Only table
        # keys are consulted, so this is deterministic and never scans
        # unrelated words. Curated tables win over the generic USDA reference
        # at each width, so a brand token ("kraft") cannot outrank a real food.
        for width in range(min(3, len(tokens)), 0, -1):
            grams = [" ".join(tokens[s : s + width]) for s in range(len(tokens) - width + 1)]
            for table, match_type in (
                (self.zero_calorie, "zero"),
                (self.baseline, "baseline"),
                (self.supplement, "supplement"),
                (self.reference, "reference"),
            ):
                for gram in grams:
                    for cand in (aliases.get(gram, gram), gram):
                        if cand in table:
                            kcal, fdc_id = (0.0, None) if match_type == "zero" else table[cand]
                            return {
                                "canonical": cand,
                                "kcal_per_100g": kcal,
                                "fdc_id": fdc_id,
                                "match_type": match_type,
                            }

        if self.enable_fuzzy and self._fuzzy_keys:
            mapped = aliases.get(norm, norm)
            close = difflib.get_close_matches(mapped, self._fuzzy_keys, n=1, cutoff=self.fuzzy_cutoff)
            if close:
                matched = close[0]
                for table in (self.baseline, self.supplement, self.reference):
                    if matched in table:
                        kcal, fdc_id = table[matched]
                        return {
                            "canonical": matched,
                            "kcal_per_100g": kcal,
                            "fdc_id": fdc_id,
                            "match_type": "fuzzy",
                        }
        return None

    def to_grams(
        self,
        quantity: float | None,
        unit: str | None,
        canonical_ingredient: str,
        fdc_id: str | None = None,
    ) -> float | None:
        """Convert (quantity, unit) to grams: food-specific table -> fallback -> defaults."""
        if quantity is None and unit is None:
            return None
        qty = quantity if quantity is not None else 1.0
        if qty <= 0:
            return None

        if unit is None:
            if fdc_id and (fdc_id, "piece") in self.unit_grams_by_fdc:
                return qty * self.unit_grams_by_fdc[(fdc_id, "piece")]
            if (canonical_ingredient, "piece") in self.unit_grams_by_ingredient:
                return qty * self.unit_grams_by_ingredient[(canonical_ingredient, "piece")]
            return qty * DEFAULT_ITEM_GRAMS["whole"]

        norm_unit = unit.strip().lower()
        if norm_unit in VOLUME_TO_CUPS:
            qty *= VOLUME_TO_CUPS[norm_unit]
            norm_unit = "cup"

        if fdc_id and (fdc_id, norm_unit) in self.unit_grams_by_fdc:
            return qty * self.unit_grams_by_fdc[(fdc_id, norm_unit)]
        if (canonical_ingredient, norm_unit) in self.unit_grams_by_ingredient:
            return qty * self.unit_grams_by_ingredient[(canonical_ingredient, norm_unit)]
        if norm_unit in self.unit_grams_fallback:
            return qty * self.unit_grams_fallback[norm_unit]
        if norm_unit in DEFAULT_ITEM_GRAMS:
            return qty * DEFAULT_ITEM_GRAMS[norm_unit]
        return qty * self.unit_grams_fallback.get("whole", 100.0)

    def estimate_ingredient_calories(
        self,
        raw_ingredient: str,
        ner_hint: str | None = None,
    ) -> dict[str, float | str | None]:
        """Estimate calories for one raw ingredient line (dict API)."""
        parsed = parse_ingredient(raw_ingredient)
        base = {
            "raw": parsed["raw"],
            "quantity": parsed["quantity"],
            "unit": parsed["unit"],
            "ingredient": parsed["ingredient"],
        }
        lookup = self.lookup_calorie_per_100g(str(parsed["ingredient"] or ""))
        if lookup is None:
            # Only trust an index-matched NER hint when it actually describes
            # this raw line (see aligned_hint); positional drift otherwise
            # misprices e.g. a macaroni line as "rotisserie chicken".
            hint = aligned_hint(raw_ingredient, ner_hint)
            if hint:
                lookup = self.lookup_calorie_per_100g(hint)
        if lookup is None:
            return {
                **base,
                "canonical": None,
                "grams": None,
                "kcal_per_100g": None,
                "calories": None,
                "match_type": "unmatched",
            }

        canonical = str(lookup["canonical"])
        kcal = float(lookup["kcal_per_100g"])
        fdc_id = lookup["fdc_id"]
        match_type = str(lookup["match_type"])

        if match_type == "zero":
            return {
                **base,
                "canonical": canonical,
                "grams": 0.0,
                "kcal_per_100g": 0.0,
                "calories": 0.0,
                "match_type": "zero",
            }

        grams = self.to_grams(
            parsed["quantity"],
            parsed["unit"],
            canonical,
            fdc_id=str(fdc_id) if fdc_id else None,
        )
        if grams is None:
            grams = self.unit_grams_fallback.get("whole", 100.0)
        return {
            **base,
            "canonical": canonical,
            "grams": round(grams, 2),
            "kcal_per_100g": kcal,
            "calories": round((grams / 100.0) * kcal, 2),
            "match_type": match_type,
        }

    def estimate_recipe_calories(
        self,
        raw_ingredients: list[str],
        ner_ingredients: list[str] | None = None,
    ) -> dict[str, object]:
        """Estimate total calories + coverage for a recipe (dict API, corpus build)."""
        total_calories = 0.0
        matched_count = 0
        details: list[dict[str, float | str | None]] = []

        for idx, raw in enumerate(raw_ingredients):
            if not isinstance(raw, str) or not raw.strip():
                continue
            hint = ner_ingredients[idx] if ner_ingredients and idx < len(ner_ingredients) else None
            entry = self.estimate_ingredient_calories(raw, ner_hint=hint)
            details.append(entry)
            if entry["calories"] is not None:
                total_calories += float(entry["calories"])
                matched_count += 1

        total_count = len(details)
        coverage = round(matched_count / total_count, 4) if total_count > 0 else 0.0
        return {
            "total_calories": round(total_calories, 2),
            "matched_count": matched_count,
            "total_count": total_count,
            "calorie_coverage": coverage,
            "parsed_ingredients": details,
        }

    def estimate_recipe(
        self,
        raw_ingredients: Iterable[str],
        ner_ingredients: Iterable[str] | None = None,
        directions: Iterable[str] = (),
        title: str = "",
        default_servings: int = 4,
    ) -> CalorieEstimate:
        """Estimate total and per-serving calories for a recipe (typed API, PRD Section 16)."""
        raw_source = list(raw_ingredients)
        raw_list = [r for r in raw_source if isinstance(r, str) and r.strip()]
        ner_list = list(ner_ingredients) if ner_ingredients is not None else []

        items: list[IngredientCalorieItem] = []
        total_kcal = 0.0
        matched_count = 0

        # Iterate the *unfiltered* list so NER hints stay paired with their
        # original raw line; dropping bad entries first would shift indices.
        for idx, raw_line in enumerate(raw_source):
            if not isinstance(raw_line, str) or not raw_line.strip():
                continue
            parsed = parse_ingredient(raw_line)
            quantity = parsed["quantity"]
            unit = parsed["unit"]
            name = str(parsed["ingredient"] or "")
            hint = ner_list[idx] if idx < len(ner_list) else None

            lookup = self.lookup_calorie_per_100g(name)
            if lookup is None:
                # Same alignment guard as the dict API: an index-matched NER
                # term from a drifted list must not price this raw line.
                hint = aligned_hint(raw_line, hint)
                if hint:
                    lookup = self.lookup_calorie_per_100g(hint)

            if lookup is None:
                items.append(
                    IngredientCalorieItem(
                        raw=raw_line,
                        quantity=quantity,
                        unit=unit,
                        ingredient=name,
                        canonical=None,
                        grams=None,
                        kcal_per_100g=None,
                        calories=None,
                        match_type="unmatched",
                    )
                )
                continue

            canonical = str(lookup["canonical"])
            kcal = float(lookup["kcal_per_100g"])
            fdc_id = lookup["fdc_id"]
            match_type = str(lookup["match_type"])

            if match_type == "zero" and quantity is None:
                grams, item_kcal = 0.0, 0.0
            else:
                grams = self.to_grams(
                    quantity, unit, canonical, fdc_id=str(fdc_id) if fdc_id else None
                )
                if grams is None:
                    grams = 0.0
                item_kcal = (grams / 100.0) * kcal

            total_kcal += item_kcal
            matched_count += 1
            items.append(
                IngredientCalorieItem(
                    raw=raw_line,
                    quantity=quantity,
                    unit=unit,
                    ingredient=name,
                    canonical=canonical,
                    grams=round(grams, 2),
                    kcal_per_100g=kcal,
                    calories=round(item_kcal, 2),
                    match_type=match_type,
                )
            )

        if not raw_list or matched_count == 0:
            return CalorieEstimate(
                total_calories=None,
                calories_per_serving=None,
                estimated_servings=1,
                servings_source="single",
                matched_count=0,
                total_count=len(raw_list),
                items=items,
            )

        parsed_servings = extract_servings(directions, title=title)
        if parsed_servings is not None:
            servings = parsed_servings
            servings_source = "parsed"
        elif total_kcal > 850:
            servings = max(1, default_servings)
            servings_source = "heuristic"
        elif total_kcal > 500:
            servings = 2
            servings_source = "heuristic"
        else:
            servings = 1
            servings_source = "single"

        total_rounded = round(total_kcal, 1)
        return CalorieEstimate(
            total_calories=total_rounded,
            calories_per_serving=round(total_rounded / servings, 1),
            estimated_servings=servings,
            servings_source=servings_source,
            matched_count=matched_count,
            total_count=len(raw_list),
            items=items,
        )
