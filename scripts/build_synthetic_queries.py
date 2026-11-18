#!/usr/bin/env python3
"""Generate the Pipeline 2 evaluation set from the recipe corpus (PRD section 32).

For each sampled recipe, a random subset of its canonical ingredients becomes a
simulated pantry. The source recipe is the relevance target: retrieval succeeds
when it appears in the candidate list.

    python scripts/build_synthetic_queries.py
    python scripts/build_synthetic_queries.py --corpus data/raw/recipenlg/RecipeNLG.csv --queries 500

Deterministic for a fixed --seed and corpus. Output:
data/test/synthetic_retrieval_queries.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from pathlib import Path

from ingredient_tables import PROCESSED, Resolver

ROOT = PROCESSED.parents[1]
SAMPLE = ROOT / "data" / "samples" / "recipes_sample.csv"
OUT = ROOT / "data" / "test" / "synthetic_retrieval_queries.json"


def canonical_ingredients(ner_field: str, resolver: Resolver) -> list[str]:
    try:
        terms = ast.literal_eval(ner_field)
    except (ValueError, SyntaxError):
        return []
    seen: dict[str, None] = {}
    for term in terms:
        canonical = resolver.resolve(term)
        if canonical:
            seen.setdefault(canonical, None)
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=SAMPLE)
    ap.add_argument("--queries", type=int, default=200, help="number of synthetic queries")
    ap.add_argument("--pantry-fraction", type=float, default=0.6, help="share of ingredients assumed on hand")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-ingredients", type=int, default=4)
    args = ap.parse_args()

    args.corpus = args.corpus.resolve()
    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}\nRun scripts/download_raw_data.py --recipes sample")
    resolver = Resolver()
    rng = random.Random(args.seed)

    # Reservoir sample so the full 2.2M-recipe corpus needs O(--queries) memory.
    chosen: list[tuple[int, str, list[str]]] = []
    eligible = 0
    with args.corpus.open(encoding="utf-8", newline="") as fh:
        for index, row in enumerate(csv.DictReader(fh)):
            ingredients = canonical_ingredients(row["NER"], resolver)
            if len(ingredients) < args.min_ingredients:
                continue
            candidate = (index, row["title"], ingredients)
            if len(chosen) < args.queries:
                chosen.append(candidate)
            else:
                slot = rng.randrange(eligible + 1)
                if slot < args.queries:
                    chosen[slot] = candidate
            eligible += 1

    if not chosen:
        raise SystemExit("no recipes with enough resolvable ingredients")
    chosen.sort(key=lambda c: c[0])
    print(f"[info] {eligible} eligible recipes, {len(chosen)} sampled")

    queries = []
    for index, title, ingredients in chosen:
        count = max(3, round(len(ingredients) * args.pantry_fraction))
        pantry = rng.sample(ingredients, min(count, len(ingredients) - 1))
        held_out = [i for i in ingredients if i not in pantry]
        queries.append(
            {
                "id": f"syn{len(queries) + 1:04d}",
                "recipe_id": index,
                "title": title,
                "query": "I have " + ", ".join(pantry[:-1]) + (" and " if len(pantry) > 1 else "") + pantry[-1] + ".",
                "pantry": pantry,
                "held_out": held_out,
                "required": ingredients,
            }
        )

    payload = {
        "version": "1.0.0",
        "description": (
            "Synthetic retrieval queries for Pipeline 2 evaluation. recipe_id is the 0-based row "
            "index in the source corpus; required is the full canonical ingredient set of that recipe."
        ),
        "generator": {
            "corpus": str(args.corpus.relative_to(ROOT)),
            "queries": len(queries),
            "pantry_fraction": args.pantry_fraction,
            "seed": args.seed,
            "min_ingredients": args.min_ingredients,
        },
        "queries": queries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] {OUT.relative_to(ROOT)}: {len(queries)} queries from {args.corpus.name}")


if __name__ == "__main__":
    main()
