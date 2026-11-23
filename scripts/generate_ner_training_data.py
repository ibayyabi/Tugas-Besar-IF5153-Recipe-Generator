"""Generate synthetic pantry queries with ingredient spans for NER training.

Recipes from the RecipeNLG sample (``NER`` field) supply real ingredient names;
templates turn them into natural-language queries with calorie/exclusion
clauses. The matching input is the *masked* context Pipeline 1 actually feeds
to its matcher (calorie and exclusion spans blanked by the same production
functions), so train-time and inference-time text distributions match.

Output: JSONL rows ``{"text", "spans", "canonical", "query"}`` where ``spans``
are character offsets of ingredient words inside the masked ``text``. Labels
are aligned to model subtokens at training time.

Usage:
    python scripts/generate_ner_training_data.py --count 15000 --seed 42
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline1.constraint_extractor import (
    exclusion_spans,
    extract_calorie_constraints,
    normalize_query_text,
)
from src.pipeline1.query_parser import _blank, load_vocabulary
from src.preprocessing.ingredient_normalizer import Vocabulary, normalize_ingredient

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "samples" / "recipes_sample.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "ner_training"

TOKEN = re.compile(r"[a-z0-9]+")

CLAUSE_TEMPLATES = [
    "I have {ings}.",
    "I've got {ings}.",
    "I have {ings} in my pantry.",
    "I have {ings} on hand.",
    "i have {ings} at home.",
    "Currently I have {ings}.",
]
CONSTRAINT_TEMPLATES = [
    "",
    "Around {n} calories.",
    "around {n} calories",
    "About {n} kcal.",
    "~{n} kcal",
    "Under {n} calories.",
    "under {n} calories",
    "Max {n} kcal.",
    "less than {n} calories",
    "Between {a} and {b} calories.",
    "At least {a} calories.",
    "No {x}.",
    "no {x}",
    "Without {x}.",
    "No {x} or {y}.",
    "without {x} and {y}",
    "Around {n} calories, no {x}.",
    "Under {n} calories without {x}.",
]


def _join_names(names: list[str], rng: random.Random) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} {rng.choice(['and', 'plus', 'with'])} {names[1]}"
    return ", ".join(names[:-1]) + f" {rng.choice(['and', 'plus'])} {names[-1]}"


def _maybe_pluralize(name: str, rng: random.Random) -> str:
    if rng.random() < 0.30 and not name.endswith("s"):
        head, _, rest = name.rpartition(" ")
        head = head + "s" if not rest else head
        return f"{head} {rest}".strip() if rest else head
    return name


def _maybe_typo(word: str, rng: random.Random) -> str:
    if len(word) >= 4 and rng.random() < 0.10:
        cut = rng.randrange(1, len(word) - 1)
        return word[:cut] + word[cut + 1] + word[cut] + word[cut + 2 :]
    return word


def _surface(name: str, rng: random.Random) -> str:
    """Render a canonical ingredient as a plausible query surface form."""
    words = [_maybe_typo(w, rng) for w in _maybe_pluralize(name, rng).split()]
    return " ".join(words)


def _canonical_pool(vocab: Vocabulary, recipes: list[list[str]]) -> list[str]:
    seen: dict[str, None] = {}
    for names in recipes:
        for name in names:
            canonical = vocab.canonical(normalize_ingredient(name))
            if canonical:
                seen.setdefault(canonical, None)
    return sorted(seen)


def _word_spans(clause: str, names: list[str]) -> list[tuple[int, int]]:
    """Char spans of every expected ingredient word inside the normalized clause."""
    expected = [word for name in names for word in name.split()]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in expected:
        match = re.compile(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])").search(clause, cursor)
        if match is None:  # typo augmentation changed the surface; search all remaining words
            match = re.compile(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])").search(clause)
            if match is None:
                raise ValueError(f"ingredient word {word!r} not found in clause {clause!r}")
        spans.append((match.start(), match.end()))
        cursor = match.end()
    return spans


def _ingredient_spans(clause: str, names: list[str]) -> list[list[int]]:
    """One char span per ingredient PHRASE (multi-word names stay one entity).

    Per-ingredient grouping is essential: after normalization, consecutive
    different ingredients are also separated by a single space, so a gap-based
    merge would fuse ``chicken potato`` into one entity and corrupt BIO labels.
    """
    word_spans = _word_spans(clause, names)
    phrases: list[list[int]] = []
    cursor = 0
    for name in names:
        width = len(name.split())
        first, last = word_spans[cursor], word_spans[cursor + width - 1]
        phrases.append([first[0], last[1]])
        cursor += width
    return phrases


def generate_one(names: list[str], excluded_pool: list[str], rng: random.Random, vocab: Vocabulary) -> dict:
    surfaces = [_surface(name, rng) for name in names]
    clause = rng.choice(CLAUSE_TEMPLATES).format(ings=_join_names(surfaces, rng))

    excluded = [n for n in rng.sample(excluded_pool, k=min(2, len(excluded_pool))) if n not in names]
    template = rng.choice(CONSTRAINT_TEMPLATES)
    constraint = template.format(
        n=rng.choice(range(200, 800, 25)),
        a=rng.choice(range(250, 500, 50)),
        b=rng.choice(range(500, 800, 50)),
        x=excluded[0] if excluded else "cheese",
        y=excluded[1] if len(excluded) > 1 else "milk",
    )
    query = f"{clause} {constraint}".strip()

    # Production masking, verbatim from query_parser.constrain_query.
    text, boundaries = normalize_query_text(query)
    _, calorie_spans = extract_calorie_constraints(text)
    masked = _blank(text, calorie_spans)
    masked_boundaries = _blank(boundaries, calorie_spans)
    spans = exclusion_spans(masked, masked_boundaries)
    available_context = _blank(masked, spans)

    clause_text = normalize_query_text(clause)[0]
    canonical = [vocab.canonical(normalize_ingredient(s)) or s for s in surfaces]
    return {
        "text": available_context,
        "spans": _ingredient_spans(clause_text, surfaces),
        "canonical": canonical,
        "query": query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=15000, help="training rows to generate")
    parser.add_argument("--dev-rows", type=int, default=500, help="held-out rows for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus", type=Path, default=SAMPLE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    vocab = load_vocabulary()

    recipes: list[list[str]] = []
    with args.corpus.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            names = ast.literal_eval(row["NER"])
            if isinstance(names, list):
                recipes.append([str(n).strip().lower() for n in names if str(n).strip()])

    pool = _canonical_pool(vocab, recipes)
    print(f"canonical ingredient pool: {len(pool)} | recipes: {len(recipes)}")

    rows: list[dict] = []
    attempts = 0
    while len(rows) < args.count + args.dev_rows and attempts < (args.count + args.dev_rows) * 3:
        attempts += 1
        recipe = rng.choice(recipes)
        usable = [n for n in recipe if vocab.canonical(normalize_ingredient(n))]
        if len(usable) < 2:
            continue
        k = rng.randint(2, min(6, len(usable)))
        names = rng.sample(usable, k=k)
        try:
            rows.append(generate_one(names, pool, rng, vocab))
        except ValueError:
            continue

    rng.shuffle(rows)
    dev, train = rows[: args.dev_rows], rows[args.dev_rows :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in (("train", train), ("dev", dev)):
        out_path = args.output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for row in split_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{out_path}: {len(split_rows)} rows")

    sample = train[0]
    print("sample text:", sample["text"])
    print("sample spans:", sample["spans"], "->", [sample["text"][s:e] for s, e in sample["spans"]])


if __name__ == "__main__":
    main()
