# CookFromPantry — Pantry-Aware & Calorie-Constrained Recipe Recommender

NLP/IR course project (IF5153). Recipe recommender that jointly scores textual ingredient
relevance, pantry coverage, missing ingredients, calorie compatibility, and explicit
ingredient exclusions — no generative LLM in the loop.

* Requirements: [`prd_cookfrompantry.md`](prd_cookfrompantry.md)
* Design: [`project_design_recipe_generator.md`](project_design_recipe_generator.md)
* Data: [`data/README.md`](data/README.md) — every dataset, its source, license, and which
  pipeline consumes it.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_raw_data.py --all      # USDA reference + 10k-recipe sample
python scripts/build_calorie_reference.py
python scripts/build_calorie_baseline.py
python scripts/build_synthetic_queries.py
python scripts/check_ingredient_coverage.py    # exits non-zero if the curated tables are inconsistent
```

Full corpus (2.3 GB):

```bash
python scripts/download_raw_data.py --recipes full
```

## Pipelines

| Stage | Responsibility | Owner |
| :--- | :--- | :--- |
| Pipeline 1 | Query & constraint understanding → structured query | 18223050 |
| Pipeline 2 | TF-IDF candidate retrieval (high recall) | 18223024 |
| Pipeline 3 | Pantry-aware multi-criteria reranking → Top-K | 18223086 |
