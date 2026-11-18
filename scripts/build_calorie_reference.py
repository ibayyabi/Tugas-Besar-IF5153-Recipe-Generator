#!/usr/bin/env python3
"""Build the calorie reference tables from USDA FoodData Central CSVs (stdlib only).

Input  : data/raw/calories/sr_legacy/     (download_raw_data.py --calories)
Output : data/processed/calorie_reference.csv      one row per USDA food
         data/processed/unit_grams_reference.csv   (fdc_id, unit) -> grams

Every value is kcal per 100 g edible portion, taken from nutrient 1008 (Energy)
with the Atwater factors (2047/2048) as fallback. FoodData Central is public
domain (CC0); see data/README.md.
"""
from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "calories"
OUT = ROOT / "data" / "processed"

ENERGY_IDS = ("1008", "2047", "2048")  # Energy kcal, Atwater general, Atwater specific

# USDA `modifier` values -> unit tokens the ingredient parser emits.
UNIT_PATTERNS = [
    (r"^cup", "cup"),
    (r"^(tbsp|tablespoon)", "tbsp"),
    (r"^(tsp|teaspoon)", "tsp"),
    (r"^(fl oz|fluid ounce)", "fl_oz"),
    (r"^(oz|ounce)", "oz"),
    (r"^(lb|pound)", "lb"),
    (r"^(g|gram)", "g"),
    (r"^(ml|milliliter)", "ml"),
    (r"^slice", "slice"),
    (r"^clove", "clove"),
    (r"^(can|jar|package|pkg|stick|bar|bunch|head|stalk|piece|pinch|dash|drop|sprig)", None),
]


def canonical_unit(modifier: str) -> str | None:
    m = modifier.strip().lower()
    for pattern, unit in UNIT_PATTERNS:
        if re.match(pattern, m):
            return unit or m.split(",")[0].split()[0]
    return None


def base_ingredient(description: str) -> str:
    """'Butter, salted' -> 'butter'; 'Carrots, raw' -> 'carrot'."""
    head = description.split(",")[0]
    head = re.sub(r"\(.*?\)", " ", head)
    head = re.sub(r"[^a-z0-9 ]+", " ", head.lower())
    head = " ".join(head.split())
    if not head:
        return head
    tokens = head.split(" ")
    tokens[-1] = singularize(tokens[-1])
    return " ".join(tokens)


def singularize(word: str) -> str:
    """Cheap plural folding so 'eggs'/'egg' and 'carrots'/'carrot' share a key."""
    if len(word) < 4 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_energy() -> tuple[dict[str, tuple[float, str]], dict[str, dict]]:
    """fdc_id -> (kcal per 100 g, nutrient id used)."""
    rows: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for source in sorted(RAW.iterdir()):
        if not (source / "food_nutrient.csv").exists():
            continue
        for row in read_csv(source / "food_nutrient.csv"):
            nid = row["nutrient_id"]
            if nid not in ENERGY_IDS or not row["amount"]:
                continue
            try:
                amount = float(row["amount"])
            except ValueError:
                continue
            rows[row["fdc_id"]].append((ENERGY_IDS.index(nid), amount, nid))
    best = {}
    for fdc_id, candidates in rows.items():
        candidates.sort()
        rank, amount, nid = candidates[0]
        best[fdc_id] = (amount, nid)
    return best


def load_foods() -> dict[str, dict]:
    foods = {}
    for source in sorted(RAW.iterdir()):
        if not (source / "food.csv").exists():
            continue
        for row in read_csv(source / "food.csv"):
            foods[row["fdc_id"]] = row
    return foods


def build_reference(foods: dict[str, dict], energy: dict) -> list[dict]:
    reference = []
    for fdc_id, row in foods.items():
        if fdc_id not in energy:
            continue
        kcal, nid = energy[fdc_id]
        reference.append(
            {
                "fdc_id": fdc_id,
                "description": row["description"].strip(),
                "base_ingredient": base_ingredient(row["description"]),
                "kcal_per_100g": round(kcal, 2),
                "energy_nutrient_id": nid,
                "data_type": row["data_type"],
                "food_category_id": row["food_category_id"],
            }
        )
    reference.sort(key=lambda r: (r["base_ingredient"], r["description"]))
    return reference


def build_unit_grams(reference: list[dict]) -> list[dict]:
    base_by_fdc = {row["fdc_id"]: row["base_ingredient"] for row in reference}
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for source in sorted(RAW.iterdir()):
        path = source / "food_portion.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            ingredient = base_by_fdc.get(row["fdc_id"])
            unit = canonical_unit(row["modifier"] or row["portion_description"] or "")
            if not ingredient or not unit or not row["gram_weight"]:
                continue
            try:
                amount = float(row["amount"] or 1)
                grams = float(row["gram_weight"])
            except ValueError:
                continue
            if amount <= 0 or grams <= 0 or grams / amount > 5000:
                continue
            buckets[(row["fdc_id"], ingredient, unit)].append(grams / amount)
    out = []
    for (fdc_id, ingredient, unit), values in sorted(buckets.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        out.append(
            {
                "fdc_id": fdc_id,
                "ingredient": ingredient,
                "unit": unit,
                "grams_per_unit": round(statistics.median(values), 2),
                "n_portions": len(values),
                "source": "USDA FDC food_portion",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {path.relative_to(ROOT)}: {len(rows)} rows")


def main() -> None:
    if not RAW.exists() or not any(RAW.iterdir()):
        raise SystemExit("data/raw/calories is empty: run download_raw_data.py --calories")

    foods = load_foods()
    energy = load_energy()
    reference = build_reference(foods, energy)
    covered = len(reference) / max(len(foods), 1)
    print(f"[info] {len(foods)} USDA foods, {len(reference)} with an energy value ({covered:.1%})")

    write_csv(OUT / "calorie_reference.csv", reference)
    write_csv(OUT / "unit_grams_reference.csv", build_unit_grams(reference))


if __name__ == "__main__":
    main()
