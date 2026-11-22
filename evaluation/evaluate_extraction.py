#!/usr/bin/env python3
"""Pipeline 1 evaluation against data/test/annotated_queries.json (PRD §31).

    python -m evaluation.evaluate_extraction
    python -m evaluation.evaluate_extraction --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline1.query_parser import parse_query

QUERIES = Path(__file__).resolve().parents[1] / "data" / "test" / "annotated_queries.json"


def _scores(pairs: list[tuple[set[str], set[str]]]) -> tuple[float, float, float, int, int, int]:
    tp = sum(len(pred & gold) for pred, gold in pairs)
    fp = sum(len(pred - gold) for pred, gold in pairs)
    fn = sum(len(gold - pred) for pred, gold in pairs)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, default=QUERIES)
    ap.add_argument("--verbose", action="store_true", help="print every mismatching query")
    args = ap.parse_args()

    cases = json.loads(args.queries.read_text(encoding="utf-8"))["queries"]
    available: list[tuple[set[str], set[str]]] = []
    excluded: list[tuple[set[str], set[str]]] = []
    calorie_hits = {"calorie_target": 0, "calorie_min": 0, "calorie_max": 0}
    exact = 0
    mismatches = []

    for case in cases:
        expected = case["expected"]
        predicted = parse_query(case["query"])
        available.append((set(predicted["available_ingredients"]), set(expected["available_ingredients"])))
        excluded.append((set(predicted["excluded_ingredients"]), set(expected["excluded_ingredients"])))
        for field in calorie_hits:
            calorie_hits[field] += predicted[field] == expected[field]
        if predicted == expected:
            exact += 1
        else:
            mismatches.append((case["id"], case["query"], expected, predicted))

    for label, pairs in (("available_ingredients", available), ("excluded_ingredients", excluded)):
        precision, recall, f1, tp, fp, fn = _scores(pairs)
        print(f"{label:<24} P={precision:.3f} R={recall:.3f} F1={f1:.3f}  (tp={tp} fp={fp} fn={fn})")

    total = len(cases)
    for field, hits in calorie_hits.items():
        print(f"{field:<24} accuracy={hits / total:.3f} ({hits}/{total})")
    print(f"{'structured query exact':<24} {exact}/{total} ({exact / total:.1%})")

    if args.verbose:
        for query_id, query, expected, predicted in mismatches:
            print(f"\n{query_id} | {query}")
            for field in expected:
                if expected[field] != predicted[field]:
                    print(f"  {field}: expected {expected[field]} got {predicted[field]}")


if __name__ == "__main__":
    main()
