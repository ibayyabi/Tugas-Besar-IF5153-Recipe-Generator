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
            title = (row.get("title") or "").strip()

            estimate = calorie_estimator.estimate_recipe(
                raw_ingredients,
                ner_ingredients=ner_terms,
                directions=directions,
                title=title,
            )

            # NER alone under-parses some recipes. Only for those (fewer than
            # three canonical terms) borrow the estimator's matched canonicals;
            # applying it to well-parsed recipes would let one component stand
            # in for a composite raw line and silently rewrite retrieval text.
            canonical_names = list(
                normalizer.extract_canonical_ingredients(ner_terms, known_only=False)
            )
            if len(canonical_names) < 3:
                for item in estimate.items:
                    if item.canonical and item.canonical not in canonical_names:
                        canonical_names.append(item.canonical)

            records.append(
                {
                    "recipe_id": index,
                    "title": title,
                    "ingredients": raw_ingredients,
                    "canonical_ingredients": canonical_names,
                    "canonical_text": " ".join(canonical_names),
                    "directions": directions,
                    "total_calories": float(estimate.total_calories or 0.0),
                    "calories_per_serving": float(estimate.calories_per_serving or 0.0),
                    "estimated_servings": int(estimate.estimated_servings),
                    "servings_source": estimate.servings_source,
                    "calorie_coverage": round(float(estimate.match_ratio), 4),
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
