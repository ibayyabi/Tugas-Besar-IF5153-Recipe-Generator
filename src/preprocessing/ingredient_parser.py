from __future__ import annotations

import re
from fractions import Fraction

UNICODE_FRACTIONS = {
    "\u00bc": " 1/4",
    "\u00bd": " 1/2",
    "\u00be": " 3/4",
    "\u2150": " 1/7",
    "\u2151": " 1/9",
    "\u2152": " 1/10",
    "\u2153": " 1/3",
    "\u2154": " 2/3",
    "\u2155": " 1/5",
    "\u2156": " 2/5",
    "\u2157": " 3/5",
    "\u2158": " 4/5",
    "\u2159": " 1/6",
    "\u215a": " 5/6",
    "\u215b": " 1/8",
    "\u215c": " 3/8",
    "\u215d": " 5/8",
    "\u215e": " 7/8",
}

UNIT_ALIASES = {
    "c": "cup",
    "cup": "cup",
    "cups": "cup",
    "tbsp": "tbsp",
    "tbs": "tbsp",
    "tb": "tbsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "tsp": "tsp",
    "ts": "tsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "fl oz": "fl_oz",
    "fluid ounce": "fl_oz",
    "fluid ounces": "fl_oz",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "g": "g",
    "gm": "g",
    "gms": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "l": "l",
    "liter": "l",
    "liters": "l",
    "pt": "pt",
    "pts": "pt",
    "pint": "pt",
    "pints": "pt",
    "qt": "qt",
    "qts": "qt",
    "quart": "qt",
    "quarts": "qt",
    "gal": "gal",
    "gallon": "gal",
    "gallons": "gal",
    "stick": "stick",
    "sticks": "stick",
    "clove": "clove",
    "cloves": "clove",
    "slice": "slice",
    "slices": "slice",
    "piece": "piece",
    "pieces": "piece",
    "whole": "whole",
    "can": "can",
    "cans": "can",
    "jar": "jar",
    "jars": "jar",
    "package": "package",
    "packages": "package",
    "pkg": "pkg",
    "pkgs": "pkg",
    "box": "box",
    "boxes": "box",
    "bunch": "bunch",
    "bunches": "bunch",
    "head": "head",
    "heads": "head",
    "stalk": "stalk",
    "stalks": "stalk",
    "sprig": "sprig",
    "sprigs": "sprig",
    "dash": "dash",
    "dashes": "dash",
    "pinch": "pinch",
    "pinches": "pinch",
    "drop": "drop",
    "drops": "drop",
    "handful": "handful",
    "handfuls": "handful",
    "medium": "medium",
    "med": "medium",
    "large": "large",
    "lg": "large",
    "small": "small",
    "sm": "small",
    "bag": "package",
    "bags": "package",
    "carton": "package",
    "cartons": "package",
    "container": "package",
    "containers": "package",
    "pack": "package",
    "packs": "package",
}

PREP_WORDS = {
    "chopped",
    "diced",
    "minced",
    "sliced",
    "drained",
    "rinsed",
    "peeled",
    "seeded",
    "grated",
    "shredded",
    "crushed",
    "melted",
    "softened",
    "beaten",
    "sifted",
    "divided",
    "optional",
    "trimmed",
    "halved",
    "quartered",
    "cubed",
    "mashed",
    "cooked",
    "uncooked",
    "boiled",
    "toasted",
    "roasted",
    "freshly",
    "finely",
    "coarsely",
    "thinly",
    "lightly",
    "well",
    "about",
    "approximately",
    "approx",
}

NUMBER_PART = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
QTY_RANGE_RE = re.compile(
    rf"^\s*(?P<first>{NUMBER_PART})(?:\s*(?:-|to|or)\s*(?P<second>{NUMBER_PART}))?\b",
    re.IGNORECASE,
)
PAREN_MEASURE_RE = re.compile(
    rf"\(\s*(?P<qty>{NUMBER_PART})\s*(?P<unit>[a-zA-Z\. ]+?)\s*\)",
    re.IGNORECASE,
)
SORTED_UNITS = sorted(UNIT_ALIASES.keys(), key=len, reverse=True)
UNIT_PATTERN = "|".join(re.escape(u) for u in SORTED_UNITS)
LEADING_UNIT_RE = re.compile(
    rf"^\s*(?P<unit>{UNIT_PATTERN})\.?(?:\s+of\b|\b)",
    re.IGNORECASE,
)
CONTAINER_UNITS = {"can", "jar", "package", "pkg", "box", "stick"}


def _replace_unicode_fractions(text: str) -> str:
    for symbol, replacement in UNICODE_FRACTIONS.items():
        if symbol in text:
            text = text.replace(symbol, replacement)
    return text


def _parse_number(token: str) -> float | None:
    token = token.strip()
    if not token:
        return None
    parts = token.split()
    total = 0.0
    try:
        for part in parts:
            if "/" in part:
                total += float(Fraction(part))
            else:
                total += float(part)
        return total
    except (ValueError, ZeroDivisionError):
        return None


def _leading_unit(text: str) -> tuple[str | None, str]:
    """Match a leading unit token; returns (unit, remaining text).

    Case matters for the bare abbreviations: uppercase ``T`` = tablespoon,
    lowercase ``t`` = teaspoon (RecipeNLG convention).
    """
    match = LEADING_UNIT_RE.match(text)
    if not match:
        return None, text
    raw_unit = re.sub(r"\s+", " ", match.group("unit").strip())
    if raw_unit == "T":
        unit = "tbsp"
    elif raw_unit == "t":
        unit = "tsp"
    else:
        unit = UNIT_ALIASES.get(raw_unit.lower())
    return unit, text[match.end():].strip()


def _clean_ingredient_name(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.split(",")[0]
    text = re.sub(r"\bor\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bto taste\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s\-']", " ", text)
    tokens = [t.lower().strip("'-") for t in text.split()]
    while tokens and tokens[0] in {"of", "a", "an", "the"}:
        tokens.pop(0)
    filtered = [t for t in tokens if t and t not in PREP_WORDS]
    if filtered:
        return " ".join(filtered)
    return " ".join(t for t in tokens if t)


def parse_ingredient(raw: str) -> dict[str, float | str | None]:
    original = raw.strip()
    working = _replace_unicode_fractions(original)
    working = re.sub(r"\s+", " ", working).strip()

    quantity: float | None = None
    unit: str | None = None

    # Phrasing like "juice of 1/2 lemon" — the quantity attaches to the noun
    juice_match = re.match(rf"^juice\s+of\s+({NUMBER_PART})\s+(.+)$", working, flags=re.IGNORECASE)
    if juice_match:
        quantity = _parse_number(juice_match.group(1))
        unit = "whole"
        working = juice_match.group(2).split(",")[0].strip()

    qty_match = QTY_RANGE_RE.match(working)
    if qty_match:
        first_val = _parse_number(qty_match.group("first"))
        second_raw = qty_match.group("second")
        if first_val is not None and second_raw:
            second_val = _parse_number(second_raw)
            quantity = (first_val + second_val) / 2.0 if second_val is not None else first_val
        else:
            quantity = first_val
        working = working[qty_match.end():].strip()

    paren_qty: float | None = None
    paren_unit: str | None = None
    paren_match = PAREN_MEASURE_RE.search(working)
    if paren_match:
        candidate_unit = paren_match.group("unit").strip().rstrip(".").lower()
        candidate_unit = re.sub(r"\s+", " ", candidate_unit)
        if candidate_unit in UNIT_ALIASES:
            paren_qty = _parse_number(paren_match.group("qty"))
            paren_unit = UNIT_ALIASES[candidate_unit]
            working = (working[: paren_match.start()] + " " + working[paren_match.end():]).strip()

    unit, working = _leading_unit(working)

    if paren_qty is not None and paren_unit is not None:
        if unit in CONTAINER_UNITS or unit is None:
            multiplier = quantity if quantity is not None else 1.0
            quantity = multiplier * paren_qty
            unit = paren_unit

    if unit is None:
        second_unit, working = _leading_unit(working)
        if second_unit is not None:
            unit = second_unit

    ingredient_name = _clean_ingredient_name(working)
    if not ingredient_name:
        ingredient_name = _clean_ingredient_name(original)

    return {
        "raw": original,
        "quantity": quantity,
        "unit": unit,
        "ingredient": ingredient_name,
    }


def parse_ingredients(raw_list: list[str]) -> list[dict[str, float | str | None]]:
    return [parse_ingredient(item) for item in raw_list if isinstance(item, str) and item.strip()]
