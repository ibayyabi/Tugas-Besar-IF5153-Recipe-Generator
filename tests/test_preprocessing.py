from __future__ import annotations

from pathlib import Path

from src.preprocessing.build_corpus import build_processed_corpus
from src.preprocessing.calorie_estimator import RecipeCalorieEstimator, aligned_hint
from src.preprocessing.ingredient_normalizer import (
    IngredientNormalizer,
    normalize_ingredient,
    normalize_text,
    singularize,
)
from src.preprocessing.ingredient_parser import parse_ingredient, parse_ingredients

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "samples" / "recipes_sample.csv"


def test_singularize_and_normalize_text() -> None:
    assert singularize("potatoes") == "potato"
    assert singularize("tomatoes") == "tomato"
    assert singularize("eggs") == "egg"
    assert singularize("berries") == "berry"
    # normalize_text is a pure query-time fold (no stemming); only
    # normalize_ingredient folds the head noun.
    assert normalize_text("Fresh Green Onions!") == "fresh green onions"
    assert normalize_ingredient("Fresh Green Onions!") == "fresh green onion"


def test_ingredient_normalizer_aliases_and_qualifiers() -> None:
    normalizer = IngredientNormalizer()
    assert normalizer.resolve("scallions") == "green onion"
    assert normalizer.resolve("catsup") == "ketchup"
    assert normalizer.resolve("granulated sugar") == "sugar"
    assert normalizer.resolve("unsalted butter") == "butter"
    assert normalizer.extract_canonical_ingredients(
        ["chickens", "potatoes", "scallions", "green onions", "salt"]
    ) == ["chicken", "potato", "green onion", "salt"]


def test_parse_ingredient_quantities_units_and_containers() -> None:
    p1 = parse_ingredient("2 tbsp butter")
    assert p1["quantity"] == 2.0
    assert p1["unit"] == "tbsp"
    assert p1["ingredient"] == "butter"

    p2 = parse_ingredient("1 1/2 c. sugar")
    assert p2["quantity"] == 1.5
    assert p2["unit"] == "cup"
    assert p2["ingredient"] == "sugar"

    p3 = parse_ingredient("1 (16 oz.) can crushed pineapple, drained")
    assert p3["quantity"] == 16.0
    assert p3["unit"] == "oz"
    assert p3["ingredient"] == "pineapple"

    p4 = parse_ingredient("200 g chicken breast")
    assert p4["quantity"] == 200.0
    assert p4["unit"] == "g"
    assert p4["ingredient"] == "chicken breast"

    p5 = parse_ingredient("salt and pepper")
    assert p5["quantity"] is None
    assert p5["unit"] is None
    assert p5["ingredient"] == "salt and pepper"

    parsed_many = parse_ingredients(["1/2 cup milk", "3 medium apples, chopped"])
    assert len(parsed_many) == 2
    assert parsed_many[0]["quantity"] == 0.5
    assert parsed_many[1]["unit"] == "medium"


def test_calorie_estimator_estimation() -> None:
    mapper = RecipeCalorieEstimator()
    butter_lookup = mapper.lookup_calorie_per_100g("unsalted butter")
    assert butter_lookup is not None
    assert butter_lookup["canonical"] == "butter"
    assert float(butter_lookup["kcal_per_100g"]) > 500.0

    water_lookup = mapper.lookup_calorie_per_100g("cold water")
    assert water_lookup is not None
    assert water_lookup["match_type"] == "zero"
    assert float(water_lookup["kcal_per_100g"]) == 0.0

    recipe_est = mapper.estimate_recipe_calories(
        [
            "2 tbsp butter",
            "1/2 cup milk",
            "200 g chicken breast",
            "1 tsp salt",
        ],
        ner_ingredients=["butter", "milk", "chicken breast", "salt"],
    )
    assert float(recipe_est["total_calories"]) > 300.0
    assert float(recipe_est["calorie_coverage"]) == 1.0


def test_build_processed_corpus_subset(tmp_path: Path) -> None:
    out_parquet = tmp_path / "recipes.parquet"
    out_vocab = tmp_path / "ingredient_vocabulary.pkl"
    df = build_processed_corpus(
        corpus_path=SAMPLE_CSV,
        output_path=out_parquet,
        vocab_path=out_vocab,
        limit=25,
    )
    assert len(df) == 25
    assert out_parquet.exists()
    assert out_vocab.exists()
    assert {
        "recipe_id",
        "title",
        "ingredients",
        "canonical_ingredients",
        "canonical_text",
        "directions",
        "total_calories",
        "calories_per_serving",
        "estimated_servings",
        "servings_source",
        "calorie_coverage",
    }.issubset(df.columns)
    assert df["recipe_id"].tolist() == list(range(25))
    # Per-serving can never exceed the whole-recipe total.
    assert (df["calories_per_serving"] <= df["total_calories"]).all()
    assert (df["estimated_servings"] >= 1).all()
    assert df["servings_source"].isin({"parsed", "heuristic", "single"}).all()
    assert df["calorie_coverage"].between(0.0, 1.0).all()
    # Canonicals are the NER union estimator canonicals: de-duplicated lists.
    assert all(len(names) == len(set(names)) for names in df["canonical_ingredients"])
    vocab = IngredientNormalizer.load_vocabulary(out_vocab)
    assert len(vocab) > 100


def test_ner_hint_alignment_guard() -> None:
    # Positional drift: NER index 0 is "rotisserie chicken" while the raw line
    # is macaroni and cheese. The hint must be ignored, not used as the food.
    assert aligned_hint("18.5 oz. box KRAFT Macaroni & Cheese Dinner", "rotisserie chicken") is None
    # Whole-phrase and contiguous-subsequence hints are accepted.
    assert aligned_hint("1 lb rotisserie chicken", "rotisserie chicken") == "rotisserie chicken"
    assert aligned_hint("2 c. chopped onions", "onion") == "onion"
    # Non-string / empty hints are ignored.
    assert aligned_hint("1 c. sugar", None) is None
    assert aligned_hint("1 c. sugar", "") is None

    estimator = RecipeCalorieEstimator()
    misaligned = estimator.estimate_ingredient_calories(
        "18.5 oz. box KRAFT Macaroni & Cheese Dinner",
        ner_hint="rotisserie chicken",
    )
    assert misaligned["canonical"] != "chicken"
    assert misaligned["match_type"] != "unmatched"

    # An unrelated hint cannot rescue an otherwise unknown food.
    unknown = estimator.estimate_ingredient_calories(
        "1 c. zzqqunknownfood", ner_hint="rotisserie chicken"
    )
    assert unknown["match_type"] == "unmatched"

    typed = estimator.estimate_recipe(
        ["18.5 oz. box KRAFT Macaroni & Cheese Dinner"],
        ner_ingredients=["rotisserie chicken"],
    )
    assert typed.items[0].canonical != "chicken"

    # An aligned hint still rescues a line whose parsed head noun is a bare
    # qualifier ("plain" -> hint "yogurt" is present in the raw text).
    rescued = estimator.estimate_ingredient_calories(
        "1/2 c. plain, low-fat yogurt", ner_hint="yogurt"
    )
    assert rescued["canonical"] == "yogurt"

    # Null lines must not shift hint pairing for later lines.
    shifted = estimator.estimate_recipe(
        [None, "1 lb rotisserie chicken"], ner_ingredients=["sugar", "rotisserie chicken"]
    )
    assert len(shifted.items) == 1
    assert shifted.items[0].canonical == "chicken"


def test_lookup_ngram_fallback() -> None:
    estimator = RecipeCalorieEstimator()
    # Brand-heavy name: canonicalize/resolve fail, the n-gram fallback finds a
    # known food token rather than a brand row or None.
    lookup = estimator.lookup_calorie_per_100g("box kraft macaroni cheese dinner")
    assert lookup is not None
    assert lookup["canonical"] in {"macaroni", "pasta"}
    assert float(lookup["kcal_per_100g"]) > 300.0
    # Unknown words still return None: no unrelated scanning.
    assert estimator.lookup_calorie_per_100g("zzqqunknownfood") is None
