from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import pandas as pd

from src.preprocessing.calorie_estimator import RecipeCalorieEstimator
from src.preprocessing.ingredient_normalizer import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_VOCAB_PATH,
    IngredientNormalizer,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_CSV = ROOT / "data" / "samples" / "recipes_sample.csv"
DEFAULT_OUTPUT_PARQUET = DEFAULT_PROCESSED_DIR / "recipes.parquet"


def _parse_list_field(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str) and item.strip()]


def build_processed_corpus(
    corpus_path: Path | str = DEFAULT_CORPUS_CSV,
    output_path: Path | str = DEFAULT_OUTPUT_PARQUET,
    vocab_path: Path | str = DEFAULT_VOCAB_PATH,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    limit: int | None = None,
) -> pd.DataFrame:
    corpus_file = Path(corpus_path)
    out_file = Path(output_path)
    normalizer = IngredientNormalizer(processed_dir=processed_dir)
    calorie_estimator = RecipeCalorieEstimator(processed_dir=processed_dir, normalizer=normalizer)

    records: list[dict[str, object]] = []
    with corpus_file.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            raw_ingredients = _parse_list_field(row.get("ingredients", ""))
            ner_terms = _parse_list_field(row.get("NER", ""))
            directions = _parse_list_field(row.get("directions", ""))

            canonical = normalizer.extract_canonical_ingredients(ner_terms, known_only=False)
            calorie_info = calorie_estimator.estimate_recipe_calories(
                raw_ingredients,
                ner_ingredients=ner_terms,
            )

            if not canonical:
                parsed_names = [
                    str(entry["canonical"] or entry["ingredient"] or "")
                    for entry in calorie_info["parsed_ingredients"]
                ]
                canonical = normalizer.extract_canonical_ingredients(parsed_names, known_only=False)

            records.append(
                {
                    "recipe_id": index,
                    "title": (row.get("title") or "").strip(),
                    "ingredients": raw_ingredients,
                    "canonical_ingredients": canonical,
                    "canonical_text": " ".join(canonical),
                    "directions": directions,
                    "total_calories": float(calorie_info["total_calories"]),
                    "calorie_coverage": float(calorie_info["calorie_coverage"]),
                    "source": (row.get("source") or "").strip(),
                    "link": (row.get("link") or "").strip(),
                }
            )

    df = pd.DataFrame(records)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False)

    vocabulary = normalizer.build_vocabulary(df["canonical_ingredients"].tolist())
    normalizer.save_vocabulary(vocabulary, path=vocab_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = build_processed_corpus(
        corpus_path=args.corpus,
        output_path=args.output,
        vocab_path=args.vocab,
        limit=args.limit,
    )
    reliable_ratio = float((df["calorie_coverage"] >= 0.7).mean()) if len(df) else 0.0
    print(f"processed_recipes={len(df)} reliable_calorie_ratio={reliable_ratio:.4f}")


if __name__ == "__main__":
    main()
