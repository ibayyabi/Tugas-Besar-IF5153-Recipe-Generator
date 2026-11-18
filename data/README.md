# Dataset manifest

Everything the three pipelines read. Raw corpora are downloaded, never committed;
derived and hand-curated tables are committed so the repo works offline.

```
data/
├── raw/            # downloaded, gitignored
│   ├── recipenlg/  # RecipeNLG.csv (2.1 GB, 2,231,142 recipes)
│   └── calories/   # USDA FoodData Central CSVs
├── samples/
│   └── recipes_sample.csv          # 10,000 sampled recipes, committed dev corpus
├── processed/                      # committed reference tables
│   ├── calorie_reference.csv       # 7,793 USDA foods
│   ├── calorie_baseline.csv        # 199 canonical ingredients -> kcal/100 g
│   ├── supplement_calories.csv     # 85 fallback ingredients
│   ├── zero_calorie_ingredients.csv  # 62 entries
│   ├── unit_grams_reference.csv    # per-food household unit weights
│   ├── unit_grams_fallback.csv     # generic unit weights
│   ├── pantry_staples.csv
│   └── ingredient_aliases.csv        # 291 aliases
└── test/
    ├── annotated_queries.json            # Pipeline 1 evaluation
    └── synthetic_retrieval_queries.json  # Pipeline 2 evaluation
```

## What each pipeline needs

| Pipeline | Dataset | File |
| :--- | :--- | :--- |
| 1 — Query understanding | Ingredient vocabulary | `samples/recipes_sample.csv` `NER` column, or `processed/calorie_baseline.csv` + `ingredient_aliases.csv` as the closed vocabulary |
| 1 — Query understanding | Synonym/alias dictionary (Risk 3) | `processed/ingredient_aliases.csv` |
| 1 — Evaluation | Annotated queries with ground truth | `test/annotated_queries.json` |
| 2 — Retrieval | Recipe corpus | `raw/recipenlg/RecipeNLG.csv` or `samples/recipes_sample.csv` |
| 2 — Evaluation | Synthetic queries + relevance target | `test/synthetic_retrieval_queries.json` |
| 3 — Reranking | Calorie reference (ingredient → kcal/100 g) | `processed/calorie_reference.csv`, `processed/calorie_baseline.csv` |
| 3 — Reranking | Supplement table (§10.2) | `processed/supplement_calories.csv` |
| 3 — Reranking | Negligible-calorie list (§10.3) | `processed/zero_calorie_ingredients.csv` |
| 3 — Reranking | Pantry staple weights (§21) | `processed/pantry_staples.csv` |
| 3 — Reranking | Unit → gram conversion for W_i (§16) | `processed/unit_grams_reference.csv`, `processed/unit_grams_fallback.csv` |

Pipeline 2 also needs `artifacts/tfidf_vectorizer.pkl` and `artifacts/recipe_vectors.pkl`;
those are produced by the TF-IDF fitting step, not by this dataset layer.

## External datasets

| Dataset | Source | License | Size | Committed |
| :--- | :--- | :--- | :--- | :--- |
| RecipeNLG | [paper](https://aclanthology.org/2020.inlg-1.4/); local file or HuggingFace mirror | CC BY-NC 4.0 | 2.1 GB (2,231,142 recipes) | 7 MB sample (10,000 recipes) |
| USDA FoodData Central, SR Legacy (April 2018) | https://fdc.nal.usda.gov/download-datasets/ | Public domain (CC0) | 6.8 MB zip / 54 MB CSV | derived tables only |

RecipeNLG mirrors are used because `recipenlg.cs.put.poznan.pl` no longer responds and the
Kaggle copy requires an authenticated API token. Column layout is identical
(`title, ingredients, directions, link, source, NER`); both the local file and the HF mirror
carry a leading unnamed pandas index column, so read it by column name and ignore the extra
field. `download_raw_data.py --recipes full` skips the download when
`data/raw/recipenlg/RecipeNLG.csv` already exists.

When the full corpus is on disk, `--recipes sample` builds the sample from it by reservoir
sampling 10,000 uniformly random recipes (seed 42, one pass, ~8 s). Without the full file it
falls back to 12 byte windows spread across the remote file, because the corpus is ordered by
source and a prefix would be single-source.

USDA FoodData Central is US-centric, which matches RecipeNLG's mostly American recipe
vocabulary. Foundation Foods is not used: its current CSV layout is per-sample and has no
`food_nutrient.csv`, and SR Legacy already covers generic ingredients completely.

## Retrieval procedure

```bash
python scripts/download_raw_data.py --all                 # USDA CSVs + 8 MB recipe sample
python scripts/build_calorie_reference.py                 # USDA -> calorie_reference.csv, unit_grams_reference.csv
python scripts/build_calorie_baseline.py                  # curated canonical ingredient baseline
python scripts/build_synthetic_queries.py                 # Pipeline 2 evaluation set
python scripts/check_ingredient_coverage.py                # validates the curated tables

# full corpus: 2.1 GB, 2,231,142 recipes (skipped when the file already exists)
python scripts/download_raw_data.py --recipes full
python scripts/build_synthetic_queries.py --corpus data/raw/recipenlg/RecipeNLG.csv --queries 500
python scripts/check_ingredient_coverage.py --corpus data/raw/recipenlg/RecipeNLG.csv
```

`build_calorie_reference.py` and `build_synthetic_queries.py` are deterministic.
`build_calorie_baseline.py` fails loudly if a curated ingredient no longer matches a USDA
description, so a re-released SR Legacy cannot silently change values.

## Schemas

`calorie_reference.csv` — one row per USDA food.

| column | meaning |
| :--- | :--- |
| `fdc_id` | USDA FoodData Central id |
| `description` | full USDA description, the search space for exact/fuzzy matching |
| `base_ingredient` | first noun of the description, plural-folded |
| `kcal_per_100g` | nutrient 1008 (Energy), falling back to Atwater factors 2047/2048 |
| `energy_nutrient_id` | which nutrient id supplied the value |
| `data_type` | always `sr_legacy_food` |
| `food_category_id` | USDA food category |

`calorie_baseline.csv` — the deterministic first hit for a canonical ingredient; use it before
fuzzy matching. `matched_description` proves which USDA row produced the value.

`unit_grams_reference.csv` — `(fdc_id, unit)` → median grams. Look it up with the `fdc_id`
you matched during calorie mapping. Units are folded to `cup`, `tbsp`, `tsp`, `fl_oz`, `oz`,
`lb`, `g`, `ml`, `slice`, `clove`, and the container words in `unit_grams_fallback.csv`.
When a food has no portion row, fall back to `unit_grams_fallback.csv` (water-density cup).

`pantry_staples.csv` — `weight` is the multiplier used in weighted coverage (§21) for an
ingredient the user did not mention. Ingredients not listed default to `1.0`. Weights are
initial values; validate them in the ablation study.

`zero_calorie_ingredients.csv` — ingredients whose calorie contribution is ignored when the
quantity cannot be parsed. Only names in this file get the zero treatment; everything else
falls through the normal calorie mapping.

`ingredient_aliases.csv` — applied after lowercasing, punctuation stripping, and plural
folding. Inclusions are never aliased across calorie classes (e.g. `buttermilk` stays
`buttermilk`); only surface variants map to the same food.

`annotated_queries.json` — 60 queries covering plain pantries, `under/below/max/less than/up
to`, `around/about/approximately`, `between A and B`, `at least`, `no/without/exclude/don't
want`, and combined constraints. Two negative cases (`q052`, `q053`) document out-of-scope
phrasings.

`synthetic_retrieval_queries.json` — `recipe_id` is the 0-based row index in the corpus that
generated it; `required` is that recipe's full canonical ingredient set; `pantry` and
`held_out` partition it. Use `pantry` as the simulated user input and score whether
`recipe_id` appears in the retrieved candidate list (`Recall@K`, `MRR`).

## Coverage and known gaps

`check_ingredient_coverage.py` resolves 89.3% of the 18.9 M RecipeNLG `NER` mentions on the
full corpus (199 baseline + 85 supplement + 62 zero-calorie + 291 aliases). The remaining
terms are generic recipe components (`filling`, `crust`, `dressing`, `topping`, `pastry`),
composite products (`crescent rolls`), and parser artifacts (`t`, `¼`, `red`, `skinless`). Extend `supplement_calories.csv` or
`ingredient_aliases.csv` and rerun the checker; it fails on dangling references.

Known limitations:

* Calories are estimates: USDA values are per 100 g edible portion, and recipe quantities are
  frequently missing or ambiguous. Surface the number as *Estimated Calories* (PRD Risk 2).
* `calorie_baseline.csv` pins one variant per ingredient (e.g. `chicken breast` with skin).
  `calorie_reference.csv` holds the alternatives if you want descriptor-aware matching.
* Supplement values marked `manual` or `estimated` are not from SR Legacy; treat them as
  approximations and keep the `source` column visible in any published table.
* The committed sample is 10,000 uniformly random recipes (seed 42), 73.8% `Gathered` /
  26.2% `Recipes1M` versus the corpus's own mix. It is for development only; use the full
  corpus (`--corpus data/raw/recipenlg/RecipeNLG.csv`) for reported retrieval metrics.
* Full-corpus scans (coverage, synthetic queries) read 2.1 GB and take ~45 s each.

## Citations

```bibtex
@inproceedings{bien-etal-2020-recipenlg,
  title = "{R}ecipe{NLG}: A Cooking Recipes Dataset for Semi-Structured Text Generation",
  author = "Bie{\'n}, Micha{\l} and Gilski, Micha{\l} and Maciejewska, Martyna and Taisner, Wojciech and Wisniewski, Dawid and Lawrynowicz, Agnieszka",
  booktitle = "Proceedings of the 13th International Conference on Natural Language Generation",
  year = "2020",
  pages = "22--28"
}
```

U.S. Department of Agriculture, Agricultural Research Service. 2018. USDA National Nutrient
Database for Standard Reference, Release 28 (SR Legacy). FoodData Central.
