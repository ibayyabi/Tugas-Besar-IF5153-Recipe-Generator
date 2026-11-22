"""Excluded ingredient detection and filtering (PRD Section 23).

Handles both exact canonical matches and generic-category exclusions such as
"no cheese" (matching cheddar cheese, mozzarella cheese, parmesan, cream cheese,
etc.) or "no nuts" (matching almond, walnut, pecan, peanut, mixed nuts) while
guarding common false-positive compound names (e.g., "peanut butter" is not
dairy butter; "cream of tartar" is not dairy cream; "nutmeg" is not a nut).

Note (PRD Section 6): Excluded ingredients are treated as user preference
constraints, not medical allergy guarantees.
"""
from __future__ import annotations

import re
from typing import Iterable

from .models import ExclusionBreakdown
from src.preprocessing.ingredient_normalizer import IngredientNormalizer, normalize_ingredient

# Explicit family expansions for generic exclusion categories observed in PRD
# and data/test/annotated_queries.json.
CATEGORY_EXPANSIONS: dict[str, frozenset[str]] = {
    "cheese": frozenset({
        "cheese",
        "cheddar",
        "cheddar cheese",
        "mozzarella",
        "mozzarella cheese",
        "parmesan",
        "parmesan cheese",
        "parmigiano",
        "swiss cheese",
        "cream cheese",
        "cottage cheese",
        "ricotta",
        "ricotta cheese",
        "process cheese",
        "american cheese",
        "velveeta",
        "feta",
        "feta cheese",
        "brie",
        "gouda",
        "gruyere",
        "provolone",
        "monterey jack",
        "colby",
        "blue cheese",
        "goat cheese",
        "mascarpone",
        "gorgonzola",
        "paneer",
    }),
    "milk": frozenset({
        "milk",
        "whole milk",
        "skim milk",
        "low fat milk",
        "buttermilk",
        "condensed milk",
        "evaporated milk",
        "half and half",
    }),
    "cream": frozenset({
        "cream",
        "heavy cream",
        "whipping cream",
        "sour cream",
        "cream cheese",
        "half and half",
        "ice cream",
        "cream of chicken soup",
        "cream of mushroom soup",
        "cream of celery soup",
    }),
    "butter": frozenset({
        "butter",
        "unsalted butter",
        "salted butter",
        "sweet butter",
        "melted butter",
        "garlic butter",
        "clarified butter",
        "ghee",
    }),
    "nut": frozenset({
        "nut",
        "mixed nut",
        "mixed nuts",
        "almond",
        "walnut",
        "pecan",
        "cashew",
        "peanut",
        "pistachio",
        "hazelnut",
        "macadamia",
        "pine nut",
        "brazil nut",
        "chestnut",
        "peanut butter",
        "almond butter",
        "cashew butter",
        "almond milk",
    }),
    "peanut": frozenset({
        "peanut",
        "peanut butter",
        "peanut oil",
    }),
}

# Compound names that contain an excluded word lexically but belong to a
# different food class and must NOT trigger false-positive exclusions.
FALSE_POSITIVE_GUARDS: dict[str, frozenset[str]] = {
    "butter": frozenset({
        "peanut butter",
        "almond butter",
        "cashew butter",
        "apple butter",
        "cocoa butter",
        "shea butter",
        "sunflower butter",
        "cookie butter",
        "buttermilk",
        "butternut squash",
        "butterscotch",
        "butterscotch chip",
    }),
    "cream": frozenset({
        "cream of tartar",
        "coconut cream",
    }),
    "milk": frozenset({
        "almond milk",
        "coconut milk",
        "soy milk",
        "oat milk",
        "rice milk",
    }),
    "nut": frozenset({
        "nutmeg",
        "butternut squash",
        "water chestnut",
        "coconut",
        "coconut milk",
        "coconut oil",
        "coconut cream",
    }),
    "egg": frozenset({
        "eggplant",
    }),
    "pea": frozenset({
        "peanut",
        "peanut butter",
        "peach",
        "pear",
    }),
}


def _contains_whole_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return bool(re.search(rf"(?:^| ){re.escape(phrase)}(?:$| )", text))


class ExclusionFilter:
    """Checks candidate recipes against user-specified excluded ingredients (PRD §23)."""

    def __init__(self, normalizer: IngredientNormalizer | None = None) -> None:
        self.normalizer = normalizer

    def _matches(self, excluded_raw: str, excluded_canon: str, candidate_forms: set[str]) -> bool:
        guards = FALSE_POSITIVE_GUARDS.get(excluded_raw, frozenset()) | FALSE_POSITIVE_GUARDS.get(
            excluded_canon, frozenset()
        )
        # If any form of the candidate is explicitly guarded, skip matching it.
        if candidate_forms & guards:
            return False

        # 1. Exact match against either raw-normalized or canonicalized excluded term
        if excluded_raw in candidate_forms or (excluded_canon and excluded_canon in candidate_forms):
            return True

        # 2. Category expansion (e.g., excluded="cheese" matches "mozzarella cheese")
        expanded = CATEGORY_EXPANSIONS.get(excluded_raw)
        if expanded:
            if candidate_forms & expanded:
                return True
            if excluded_raw in ("cheese", "cream", "milk", "butter"):
                for form in candidate_forms:
                    if _contains_whole_phrase(form, excluded_raw):
                        return True
            return False

        # 3. Safe whole-phrase subtype match (e.g., "mushroom" matches "cream of mushroom soup",
        #    "onion" matches "green onion", "garlic" matches "garlic powder")
        for form in candidate_forms:
            if _contains_whole_phrase(form, excluded_raw):
                return True
            if excluded_canon and excluded_canon != excluded_raw and _contains_whole_phrase(
                form, excluded_canon
            ):
                return True
        return False

    def check(
        self,
        excluded_ingredients: Iterable[str],
        recipe_ingredients: Iterable[str],
    ) -> ExclusionBreakdown:
        """Evaluate excluded ingredient violations for a single recipe."""
        excluded_list = [e for e in excluded_ingredients if normalize_ingredient(e)]
        if not excluded_list:
            return ExclusionBreakdown(
                violated=False,
                violation_count=0,
                violated_ingredients=[],
                matched_rules={},
            )

        # Precompute normalized + canonicalized forms per recipe ingredient
        recipe_entries: list[tuple[str, set[str]]] = []
        for ing in recipe_ingredients:
            raw_norm = normalize_ingredient(ing)
            if not raw_norm:
                continue
            forms = {raw_norm}
            display_name = raw_norm
            if self.normalizer is not None:
                canon = self.normalizer.canonicalize(raw_norm)
                if canon:
                    forms.add(canon)
                    display_name = canon
            recipe_entries.append((display_name, forms))

        matched_rules: dict[str, list[str]] = {}
        violated_set: dict[str, None] = {}

        for exc in excluded_list:
            exc_raw = normalize_ingredient(exc)
            if not exc_raw:
                continue
            # Keep exc_raw intact for category matching (e.g. "cheese" must stay
            # "cheese" even though ingredient_aliases maps cheese -> cheddar cheese).
            exc_canon = exc_raw
            if self.normalizer is not None and exc_raw not in CATEGORY_EXPANSIONS:
                exc_canon = self.normalizer.canonicalize(exc_raw, drop_qualifiers=False)

            hits: list[str] = []
            for display_name, forms in recipe_entries:
                if self._matches(exc_raw, exc_canon, forms):
                    if display_name not in hits:
                        hits.append(display_name)
                    violated_set.setdefault(display_name, None)

            if hits:
                matched_rules[exc_raw] = hits

        violated_ingredients = list(violated_set)
        return ExclusionBreakdown(
            violated=len(violated_ingredients) > 0,
            violation_count=len(violated_ingredients),
            violated_ingredients=violated_ingredients,
            matched_rules=matched_rules,
        )
