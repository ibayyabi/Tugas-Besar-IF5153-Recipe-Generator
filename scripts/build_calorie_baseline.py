#!/usr/bin/env python3
"""Build the curated ingredient -> kcal/100 g baseline used before fuzzy matching.

`calorie_reference.csv` is the raw USDA search space (7,793 foods, many variants
per ingredient). This script pins one deliberate USDA description per canonical
ingredient so the calorie mapper has a deterministic first hit; anything not
listed here falls through to exact -> normalized -> fuzzy -> supplement.

Input  : data/processed/calorie_reference.csv
Output : data/processed/calorie_baseline.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "processed" / "calorie_reference.csv"
OUT = ROOT / "data" / "processed" / "calorie_baseline.csv"

# canonical ingredient -> tokens that must all appear in the USDA description.
# The first non-processed, shortest match wins, so keep tokens discriminating.
MAP: dict[str, tuple[str, ...]] = {
    # pantry / baking
    "salt": ("salt, table",),
    "sugar": ("sugars, granulated",),
    "brown sugar": ("sugars, brown",),
    "powdered sugar": ("sugars, powdered",),
    "flour": ("wheat flour", "all-purpose"),
    "whole wheat flour": ("wheat flour", "whole-grain"),
    "cornstarch": ("cornstarch",),
    "baking powder": ("leavening agents, baking powder", "double-acting"),
    "baking soda": ("leavening agents, baking soda",),
    "yeast": ("yeast, baker's",),
    "cornmeal": ("cornmeal, whole-grain, yellow",),
    "oat": ("cereals, oats", "regular and quick", "dry"),
    "bread crumb": ("bread, crumbs", "plain"),
    "cracker crumb": ("crackers, saltines",),
    "gelatin": ("gelatins, dry powder",),
    "cocoa": ("cocoa, dry powder, unsweetened",),
    "chocolate chip": ("candies, semisweet chocolate",),
    "vanilla": ("vanilla extract",),
    "honey": ("honey",),
    "molasses": ("molasses",),
    "maple syrup": ("syrup, maple",),
    "peanut butter": ("peanut butter, smooth",),
    # fats and dairy
    "butter": ("butter, salted",),
    "margarine": ("margarine, regular",),
    "shortening": ("shortening, household",),
    "oil": ("oil, soybean, salad or cooking",),
    "olive oil": ("oil, olive",),
    "vegetable oil": ("oil, soybean, salad or cooking",),
    "milk": ("milk, whole, 3.25%",),
    "evaporated milk": ("milk, evaporated, 2%",),
    "condensed milk": ("milk, canned, condensed, sweetened",),
    "buttermilk": ("milk, buttermilk, fluid, cultured, lowfat",),
    "cream": ("cream, fluid, heavy whipping",),
    "heavy cream": ("cream, fluid, heavy whipping",),
    "half and half": ("cream, fluid, half and half",),
    "sour cream": ("cream, sour, cultured",),
    "yogurt": ("yogurt, plain, whole milk",),
    "cream cheese": ("cheese, cream",),
    "cheddar cheese": ("cheese, cheddar",),
    "parmesan cheese": ("cheese, parmesan, hard",),
    "mozzarella cheese": ("cheese, mozzarella, whole milk",),
    "swiss cheese": ("cheese, swiss",),
    "ricotta cheese": ("cheese, ricotta, whole milk",),
    "cottage cheese": ("cheese, cottage, creamed", "curd"),
    "egg": ("egg, whole, raw, fresh",),
    "egg white": ("egg, white, raw, fresh",),
    "egg yolk": ("egg, yolk, raw, fresh",),
    # vegetables
    "onion": ("onions, raw",),
    "green onion": ("onions, spring or scallions",),
    "garlic": ("garlic, raw",),
    "potato": ("potatoes, flesh and skin, raw",),
    "sweet potato": ("sweet potato, raw",),
    "carrot": ("carrots, raw",),
    "celery": ("celery, raw",),
    "tomato": ("tomatoes, red, ripe, raw",),
    "tomato paste": ("tomato products, canned, paste",),
    "tomato sauce": ("tomato products, canned, sauce",),
    "green pepper": ("peppers, sweet, green, raw",),
    "red pepper": ("peppers, sweet, red, raw",),
    "bell pepper": ("peppers, sweet, green, raw",),
    "jalapeno": ("peppers, jalapeno, raw",),
    "mushroom": ("mushrooms, white, raw",),
    "spinach": ("spinach, raw",),
    "lettuce": ("lettuce, iceberg",),
    "cabbage": ("cabbage, raw",),
    "broccoli": ("broccoli, raw",),
    "cauliflower": ("cauliflower, raw",),
    "zucchini": ("squash, summer, zucchini", "raw"),
    "eggplant": ("eggplant, raw",),
    "arugula": ("arugula, raw",),
    "asparagus": ("asparagus, raw",),
    "lime": ("limes, raw",),
    "shallot": ("shallots, raw",),
    "mint": ("peppermint, fresh",),
    "caper": ("capers, canned",),
    "tarragon": ("spices, tarragon, dried",),
    "coffee": ("beverages, coffee, brewed, breakfast blend",),
    "mango": ("mangos, raw",),
    "raspberry": ("raspberries, raw",),
    "butternut squash": ("squash, winter, butternut, raw",),
    "summer squash": ("squash, summer, all varieties, raw",),
    "rhubarb": ("rhubarb, raw",),
    "hazelnut": ("nuts, hazelnuts or filberts",),
    "beet": ("beets, raw",),
    "pumpkin puree": ("pumpkin, canned",),
    "bean sprout": ("mung beans, mature seeds, sprouted, raw",),
    "couscous": ("couscous, dry",),
    "radish": ("radishes, raw",),
    "pork tenderloin": ("pork, fresh, loin, tenderloin", "cooked, broiled"),
    "beer": ("alcoholic beverage, beer, regular, all",),
    "sake": ("alcoholic beverage, rice (sake)",),
    "whiskey": ("alcoholic beverage, distilled, all", "80 proof"),
    "pretzel": ("snacks, pretzels, hard, plain",),
    "cucumber": ("cucumber, with peel, raw",),
    "green bean": ("beans, snap, green, raw",),
    "pea": ("peas, green, raw",),
    "corn": ("corn, sweet, yellow, raw",),
    "pumpkin": ("pumpkin, raw",),
    "avocado": ("avocados, raw, all commercial varieties",),
    "olive": ("olives, ripe, canned",),
    "ginger": ("ginger root, raw",),
    "pineapple": ("pineapple, raw, all varieties",),
    "peach": ("peaches, yellow, raw",),
    "cherry": ("cherries, sweet, raw",),
    "date": ("dates, medjool",),
    "tomato juice": ("tomato juice, canned",),
    "horseradish": ("horseradish, prepared",),
    "sauerkraut": ("sauerkraut, canned",),
    "applesauce": ("applesauce, canned, unsweetened",),
    "apple juice": ("apple juice, canned or bottled, unsweetened",),
    "potato chip": ("snacks, potato chips, plain, salted",),
    "mixed vegetable": ("vegetables, mixed, canned",),
    "cranberry sauce": ("cranberry sauce, canned, sweetened",),
    "red wine": ("alcoholic beverage, wine, table, red",),
    "chocolate cake": ("cake, chocolate, prepared from recipe without frosting",),
    "apple": ("apples, raw, with skin",),
    "banana": ("bananas, raw",),
    "orange": ("oranges, raw, all commercial varieties",),
    "lemon": ("lemons, raw, without peel",),
    "lemon juice": ("lemon juice, raw",),
    "lime juice": ("lime juice, raw",),
    "strawberry": ("strawberries, raw",),
    "blueberry": ("blueberries, raw",),
    "raisin": ("raisins, dark, seedless",),
    "coconut": ("nuts, coconut meat, raw",),
    # proteins
    "chicken": ("chicken, broilers or fryers, meat and skin, raw",),
    "chicken breast": ("chicken, broilers or fryers, breast", "raw"),
    "chicken thigh": ("chicken, broilers or fryers, thigh", "raw"),
    "ground beef": ("beef, ground, 80% lean meat / 20% fat", "raw"),
    "beef": ("beef, chuck, arm pot roast", "raw"),
    "pork": ("pork, fresh, loin, blade", "raw"),
    "bacon": ("pork, cured, bacon, unprepared",),
    "ham": ("ham, sliced, regular",),
    "sausage": ("pork sausage, link/patty, unprepared",),
    "turkey": ("turkey, whole, meat and skin, raw",),
    "shrimp": ("crustaceans, shrimp, raw",),
    "crab": ("crustaceans, crab, blue, raw",),
    "tuna": ("fish, tuna, light, canned in water", "drained"),
    "salmon": ("fish, salmon, atlantic, farmed, raw",),
    "cod": ("fish, cod, atlantic, raw",),
    "tilapia": ("fish, tilapia, raw",),
    "anchovy": ("fish, anchovy, european, raw",),
    "sardine": ("fish, sardine, atlantic, canned in oil",),
    "tofu": ("tofu, raw, firm",),
    "lentil": ("lentils, raw",),
    "chickpea": ("chickpeas", "mature seeds, raw"),
    "kidney bean": ("beans, kidney, all types", "raw"),
    "black bean": ("beans, black, mature seeds, raw",),
    "pinto bean": ("beans, pinto, mature seeds, raw",),
    # grains and pasta
    "rice": ("rice, white, long-grain, regular, raw",),
    "brown rice": ("rice, brown, long-grain, raw",),
    "pasta": ("pasta, dry, enriched",),
    "spaghetti": ("pasta, dry, enriched",),
    "macaroni": ("pasta, dry, enriched",),
    "noodle": ("noodles, egg, dry, enriched",),
    "quinoa": ("quinoa, uncooked",),
    "bread": ("bread, white, commercially prepared",),
    "tortilla": ("tortillas, ready-to-bake or -fry, flour",),
    "oatmeal": ("cereals, oats, regular and quick", "dry"),
    # nuts and seeds
    "almond": ("nuts, almonds",),
    "walnut": ("nuts, walnuts, english",),
    "pecan": ("nuts, pecans",),
    "cashew": ("nuts, cashew nuts, raw",),
    "peanut": ("peanuts, all types, raw",),
    "pistachio": ("nuts, pistachio nuts, raw",),
    "sunflower seed": ("seeds, sunflower seed kernels, dried",),
    "pumpkin seed": ("seeds, pumpkin and squash seed kernels, dried",),
    "sesame seed": ("seeds, sesame seeds, whole, dried",),
    "poppy seed": ("spices, poppy seed",),
    "flaxseed": ("seeds, flaxseed",),
    "chia seed": ("seeds, chia seeds, dried",),
    # condiments and seasoning
    "pepper": ("spices, pepper, black",),
    "salt and pepper": ("spices, pepper, black",),
    "cinnamon": ("spices, cinnamon, ground",),
    "nutmeg": ("spices, nutmeg, ground",),
    "paprika": ("spices, paprika",),
    "cumin": ("spices, cumin seed",),
    "oregano": ("spices, oregano, dried",),
    "thyme": ("spices, thyme, dried",),
    "rosemary": ("spices, rosemary, dried",),
    "basil": ("spices, basil, dried",),
    "parsley": ("parsley, fresh",),
    "cilantro": ("coriander (cilantro) leaves, raw",),
    "bay leaf": ("spices, bay leaf",),
    "chili powder": ("spices, chili powder",),
    "cayenne": ("spices, pepper, red or cayenne",),
    "garlic powder": ("spices, garlic powder",),
    "onion powder": ("spices, onion powder",),
    "mustard": ("mustard, prepared, yellow",),
    "mayonnaise": ("mayonnaise, regular",),
    "ketchup": ("catsup",),
    "vinegar": ("vinegar, distilled",),
    "soy sauce": ("soy sauce made from soy and wheat", "shoyu"),
    "worcestershire sauce": ("sauce, worcestershire",),
    "hot sauce": ("sauce, ready-to-serve, pepper or hot",),
    "salsa": ("sauce, salsa, ready-to-serve",),
    "chicken broth": ("soup, chicken broth, ready-to-serve",),
    "beef broth": ("soup, beef broth or bouillon canned, ready-to-serve",),
    "water": ("water, tap",),
}

EXCLUDE = (
    "meatless", "imitation", "substitute", "low-sodium", "low sodium", "low fat",
    "fat free", "reduced", "toasted", "sharp", "babyfood", "fast food", "restaurant",
)

PROCESSED = (
    "dehydrated", "dried", "canned", "frozen", "cooked", "powder", "syrup",
    "sweetened", "juice", "concentrate", "fried", "babyfood", "restaurant",
    "fast food", "candies", "cereal", "snack", "mix", "prepared", "pasteurized",
)


def load_reference() -> list[dict]:
    with REFERENCE.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def pick(reference: list[dict], tokens: tuple[str, ...]) -> dict | None:
    hits = [
        row
        for row in reference
        if all(token in row["description"].lower() for token in tokens)
        and not any(word in row["description"].lower() for word in EXCLUDE)
    ]
    if not hits:
        return None
    hits.sort(key=lambda r: (any(w in r["description"].lower() for w in PROCESSED), len(r["description"])))
    return hits[0]


def main() -> None:
    if not REFERENCE.exists():
        raise SystemExit("missing data/processed/calorie_reference.csv: run build_calorie_reference.py")
    reference = load_reference()
    rows, unresolved = [], []
    for ingredient, tokens in MAP.items():
        match = pick(reference, tokens)
        if match is None:
            unresolved.append((ingredient, tokens))
            continue
        rows.append(
            {
                "ingredient": ingredient,
                "kcal_per_100g": match["kcal_per_100g"],
                "matched_description": match["description"],
                "fdc_id": match["fdc_id"],
                "source": "USDA FDC SR Legacy",
            }
        )
    rows.sort(key=lambda r: r["ingredient"])
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {OUT.relative_to(ROOT)}: {len(rows)} ingredients")
    for ingredient, tokens in unresolved:
        print(f"[warn] no USDA match for {ingredient!r} ({' '.join(tokens)})", file=sys.stderr)


if __name__ == "__main__":
    main()
