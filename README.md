# YummyTummy — Pantry-Aware & Calorie-Constrained Recipe Recommender

NLP/IR course project (IF5153). Recipe recommender that jointly scores textual ingredient
relevance, pantry coverage, missing ingredients, calorie compatibility, and explicit
ingredient exclusions — no generative LLM in the loop.

* Requirements: [`prd_yummytummy.md`](prd_yummytummy.md)
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

## Pipeline 1 — query understanding

```python
from src.pipeline1.query_parser import parse_query
parse_query("I have chicken, potato and garlic. Around 500 calories, no cheese.")
# {'available_ingredients': ['chicken', 'potato', 'garlic'],
#  'excluded_ingredients': ['cheese'],
#  'calorie_target': 500, 'calorie_min': None, 'calorie_max': None}
```

```bash
python -m evaluation.evaluate_extraction --verbose   # P/R/F1 + calorie accuracy on the 60 annotated queries
python -m evaluation.experiment                      # compare rule, exact-surface, difflib, and char n-gram matching
python -m evaluation.experiment --verbose            # include per-query error details
python -m pytest tests/                              # Pipeline 1 regression guards
```

Constraint patterns: `under/below/max/maximum/less than/up to/no more than/at most/not exceeding/less than or equal to`,
`at least/minimum/min/over/more than/above`, `around/about/approximately/roughly/nearly`,
`between A and B` and `A to B`, plus unit-bearing suffix forms (`500 calories max/or less`).
Numbers followed by a time/volume/count unit (`under 20 minutes`, `over 2 cups`, `no more than 3 eggs`)
are not calorie constraints. Exclusions follow
`no/without/exclude/avoid/don't want/can't eat/allergic to/except/skip` and
`X-free` / `X free`; an exclusion clause stops at punctuation or an availability
transition (`... but I have ...`).
A bare number (`450 kcal`) is deliberately not a constraint.

## Pipelines

| Stage | Responsibility | Owner |
| :--- | :--- | :--- |
| Pipeline 1 | Query & constraint understanding → structured query | 18223050 |
| Pipeline 2 | Candidate retrieval (high recall) | 18223024 |
| Pipeline 3 | Pantry-aware multi-criteria reranking → Top-K | 18223086 |

## Approach registry — experiment per pipeline

Each pipeline exposes pluggable approaches behind one contract; the constraint
grammar (P1) and the candidate schema (P2→P3) stay fixed so experiments isolate
the swapped component.

| Pipeline | Approach | Type |
| :--- | :--- | :--- |
| 1 | `rule` | production baseline (vocabulary + aliases + plural folding) |
| 1 | `surface-exact` | ablation |
| 1 | `difflib-0.82`, `char-ngram-0.60` | fuzzy fallbacks |
| 1 | `ner` | fine-tuned DistilBERT token classification (local, no LLM API) |
| 2 | `tfidf` | production baseline (max_features=1000, staple down-weighting) |
| 2 | `dense` | sentence-transformer `all-MiniLM-L6-v2` embeddings |
| 3 | weighted α/β/γ/δ + ablation models A–D | PRD §24/§35 |

## Build offline artifacts

```bash
python -m src.preprocessing.build_corpus        # → data/processed/recipes.parquet
python -m src.pipeline2.vectorizer              # → artifacts/tfidf_*.pkl
python scripts/encode_dense_corpus.py           # → artifacts/dense_vectors.npy
python scripts/generate_ner_training_data.py --count 15000
python scripts/train_ner.py                     # → artifacts/ner_model/ (MPS/CPU)
```

## Run the interactive UI

```bash
streamlit run app/streamlit_app.py
```

Sidebar picks an approach per pipeline (registry-aware), reranking weights
(including ablation presets), Top-K. Tabs: explainable result cards (PRD §26),
behind-the-scenes pipeline trace — spans, matching provenance, TF-IDF term
weights, score composition, and the BERT visualizer (token tags + attention
heatmap) — a live approach comparison, and the architecture map.

## Run the experiments (CLI)

```bash
python -m evaluation.experiment --verbose            # P1: all approaches, error taxonomy
python -m evaluation.experiment --output-json out.json
python -m evaluation.evaluate_extraction --verbose   # P1: production parser only
python -m evaluation.evaluate_retrieval              # P2: recall@k / MRR, synthetic queries
python -m evaluation.evaluate_ranking                # P3: ablation A–D (PRD §35)
python -m pytest tests/
```

Experimental approaches need `torch`, `transformers`, `sentence-transformers`
(already in `requirements.txt`); the lexical baselines do not.
