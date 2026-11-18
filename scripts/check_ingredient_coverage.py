#!/usr/bin/env python3
"""Validate the curated ingredient tables and report corpus coverage.

Checks (exit code 1 on failure):
  * every alias canonical exists in the calorie baseline, supplement, or zero list
  * every pantry-staple ingredient has a calorie treatment
  * no duplicate aliases after normalization
  * every annotated-query label resolves to the ingredient vocabulary

Prints how many RecipeNLG `NER` mentions resolve to a calorie source and the
most frequent unresolved terms (the next alias/supplement candidates).

    python scripts/check_ingredient_coverage.py
    python scripts/check_ingredient_coverage.py --corpus data/raw/recipenlg/RecipeNLG.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from ingredient_tables import PROCESSED, Resolver, normalize, read_rows

SAMPLE = PROCESSED.parent / "samples" / "recipes_sample.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=SAMPLE, help="CSV with a RecipeNLG NER column")
    ap.add_argument("--top", type=int, default=25, help="unresolved terms to list")
    args = ap.parse_args()

    resolver = Resolver()
    failures: list[str] = []

    alias_rows = read_rows(PROCESSED / "ingredient_aliases.csv")
    for alias, count in Counter(normalize(r["alias"]) for r in alias_rows).items():
        if count > 1:
            failures.append(f"duplicate alias: {alias}")
    for row in alias_rows:
        if normalize(row["canonical"]) not in resolver.known:
            failures.append(f"alias {row['alias']!r} -> unknown canonical {row['canonical']!r}")
    for row in read_rows(PROCESSED / "pantry_staples.csv"):
        if resolver.resolve(row["ingredient"]) is None:
            failures.append(f"pantry staple {row['ingredient']!r} has no calorie treatment")

    annotations = json.loads((PROCESSED.parent / "test" / "annotated_queries.json").read_text(encoding="utf-8"))
    for query in annotations["queries"]:
        for field in ("available_ingredients", "excluded_ingredients"):
            for ingredient in query["expected"][field]:
                if resolver.resolve(ingredient) is None:
                    failures.append(f"{query['id']}: labeled {field} {ingredient!r} is outside the vocabulary")

    unresolved: Counter[str] = Counter()
    total = hit = 0
    if args.corpus.exists():
        with args.corpus.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    terms = ast.literal_eval(row["NER"])
                except (ValueError, SyntaxError):
                    continue
                for term in terms:
                    total += 1
                    if resolver.resolve(term):
                        hit += 1
                    else:
                        unresolved[term.lower().strip()] += 1
    else:
        print(f"[warn] corpus not found: {args.corpus}", file=sys.stderr)

    print(
        f"baseline={len(resolver.baseline)} supplement={len(resolver.supplement)} "
        f"zero={len(resolver.zero)} aliases={len(resolver.aliases)}"
    )
    if total:
        print(f"corpus coverage: {hit}/{total} ingredient mentions resolved ({hit / total:.1%})")
        print(f"distinct unresolved terms: {len(unresolved)}")
        for term, count in unresolved.most_common(args.top):
            print(f"  {count:>5}  {term}")

    if failures:
        print("\nFAIL", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        raise SystemExit(1)
    print("\nOK: curated tables are internally consistent")


if __name__ == "__main__":
    main()
