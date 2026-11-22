from __future__ import annotations

from pathlib import Path

from src.preprocessing.build_corpus import build_processed_corpus
from src.preprocessing.calorie_estimator import RecipeCalorieEstimator
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
        "calorie_coverage",
    }.issubset(df.columns)
    vocab = IngredientNormalizer.load_vocabulary(out_vocab)
    assert len(vocab) > 100
