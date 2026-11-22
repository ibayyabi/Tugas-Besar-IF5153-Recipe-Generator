"""Compare Pipeline 1 ingredient-matching approaches and report their errors.

Run ``python -m evaluation.experiment``; add ``--verbose`` for query-level diffs
and ``--output-json PATH`` to persist the metrics table (used by the Streamlit
comparison tab). All variants share the production calorie/exclusion parser so
this experiment isolates ingredient matching. Approaches come from the
``src.pipeline1.approaches`` registry (single source of truth); no LLM is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.pipeline1.approaches import available_approaches, get_approach
from src.pipeline1.query_parser import load_vocabulary, parse_query
from src.preprocessing.ingredient_normalizer import Vocabulary

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "annotated": ROOT / "data/test/annotated_queries.json",
    "stress": ROOT / "data/test/stress_queries.json",
}
CALORIE_FIELDS = ("calorie_target", "calorie_min", "calorie_max")
FIELDS = ("available_ingredients", "excluded_ingredients", *CALORIE_FIELDS)


def _errors(expected: dict, predicted: dict) -> list[tuple[str, str]]:
    errors = []
    for field, category in (("available_ingredients", "ingredient"), ("excluded_ingredients", "exclusion")):
        gold, got = set(expected[field]), set(predicted[field])
        errors.extend((f"{category}_miss", item) for item in sorted(gold - got))
        errors.extend((f"{category}_extra", item) for item in sorted(got - gold))

    swapped_fields = set()
    for field in CALORIE_FIELDS:
        value = expected[field]
        if value is not None and predicted[field] != value:
            other = next((name for name in CALORIE_FIELDS if name != field and predicted[name] == value), None)
            if other:
                errors.append(("calorie_field", f"{field}->{other}"))
                swapped_fields.add(other)
            elif predicted[field] is None:
                errors.append(("calorie_miss", field))
            else:
                errors.append(("calorie_value", field))
    for field in CALORIE_FIELDS:
        if field not in swapped_fields and expected[field] is None and predicted[field] is not None:
            errors.append(("calorie_extra", field))
    return errors


def _f1(pairs: list[tuple[set[str], set[str]]]) -> float:
    tp = sum(len(pred & gold) for pred, gold in pairs)
    fp = sum(len(pred - gold) for pred, gold in pairs)
    fn = sum(len(gold - pred) for pred, gold in pairs)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _approach_names() -> list[str]:
    """All registry approaches with loadable artifacts, in registry order."""
    return [entry["name"] for entry in available_approaches() if entry["available"]]


def run_experiment(
    approach_names: list[str] | None = None,
    datasets: dict[str, Path] | None = None,
    vocab: Vocabulary | None = None,
) -> dict[str, dict]:
    """Evaluate each approach over the annotated + stress fixtures.

    Returns ``{approach: {total, exact, exact_pct, available_f1, excluded_f1,
    calorie_accuracy, errors, failures}}`` — the single source both the CLI
    table and the Streamlit comparison tab render.
    """
    datasets = datasets or DATASETS
    vocab = vocab or load_vocabulary()
    names = approach_names or _approach_names()
    cases = [(name, item) for name, path in datasets.items() for item in json.loads(path.read_text(encoding="utf-8"))["queries"]]

    results: dict[str, dict] = {}
    for name in names:
        approach = get_approach(name)
        exact = 0
        available: list[tuple[set[str], set[str]]] = []
        excluded: list[tuple[set[str], set[str]]] = []
        calorie_hits = 0
        taxonomy: Counter[str] = Counter()
        failures = []
        for dataset, case in cases:
            expected = case["expected"]
            predicted = parse_query(case["query"], vocab, approach.extract)
            errs = _errors(expected, predicted)
            taxonomy.update(category for category, _ in errs)
            exact += predicted == expected
            available.append((set(predicted["available_ingredients"]), set(expected["available_ingredients"])))
            excluded.append((set(predicted["excluded_ingredients"]), set(expected["excluded_ingredients"])))
            calorie_hits += sum(predicted[field] == expected[field] for field in CALORIE_FIELDS)
            if errs:
                failures.append(
                    {
                        "dataset": dataset,
                        "id": case["id"],
                        "query": case["query"],
                        "expected": expected,
                        "predicted": predicted,
                        "errors": errs,
                    }
                )
        total = len(cases)
        results[name] = {
            "total": total,
            "exact": exact,
            "exact_pct": exact / total if total else 0.0,
            "available_f1": round(_f1(available), 4),
            "excluded_f1": round(_f1(excluded), 4),
            "calorie_accuracy": round(calorie_hits / (total * 3), 4) if total else 0.0,
            "errors": dict(taxonomy),
            "failures": failures,
        }
    return results


def format_results_table(results: dict[str, dict]) -> str:
    """ASCII-aligned table, same layout the report historically used."""
    lines = [
        "Approach         Exact     Available F1  Excluded F1  Calorie accuracy",
        "---------------- --------- ------------- ------------ ----------------",
    ]
    for name, metrics in results.items():
        lines.append(
            f"{name:<16} {metrics['exact']:>3}/{metrics['total']:<5}  {metrics['available_f1']:>11.3f}  "
            f"{metrics['excluded_f1']:>10.3f}  {metrics['calorie_accuracy']:>15.3f}"
        )
        errors = ", ".join(f"{key}={value}" for key, value in sorted(metrics["errors"].items())) or "none"
        lines.append(f"  errors: {errors}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="show each mismatching query")
    parser.add_argument("--output-json", type=Path, default=None, help="persist the metrics table as JSON")
    args = parser.parse_args()

    results = run_experiment()
    print(f"Pipeline 1: {results[next(iter(results))]['total']} queries (annotated + stress)")
    print(format_results_table(results))

    if args.verbose:
        for name, metrics in results.items():
            for failure in metrics["failures"]:
                print(f"  {name} {failure['dataset']}/{failure['id']}: {failure['query']}")
                print(f"    errors={failure['errors']}; expected={failure['expected']}; predicted={failure['predicted']}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[info] wrote {args.output_json}")


if __name__ == "__main__":
    main()
