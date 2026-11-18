"""Shared ingredient normalization and table loading.

Used by the dataset tooling only (coverage report, synthetic query generation).
Pipeline code (`src/preprocessing/`) will re-implement the same rules; keep the
two in sync through data/processed/ingredient_aliases.csv.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def singularize(word: str) -> str:
    if len(word) < 4 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, fold the last token's plural."""
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    tokens = text.split()
    if not tokens:
        return ""
    tokens[-1] = singularize(tokens[-1])
    return " ".join(tokens)


class Resolver:
    def __init__(self, processed: Path = PROCESSED) -> None:
        self.baseline = {normalize(r["ingredient"]) for r in read_rows(processed / "calorie_baseline.csv")}
        self.supplement = {normalize(r["ingredient"]) for r in read_rows(processed / "supplement_calories.csv")}
        self.zero = {normalize(r["ingredient"]) for r in read_rows(processed / "zero_calorie_ingredients.csv")}
        self.aliases = {
            normalize(r["alias"]): normalize(r["canonical"])
            for r in read_rows(processed / "ingredient_aliases.csv")
        }
        self.known = self.baseline | self.supplement | self.zero

    def resolve(self, term: str) -> str | None:
        key = normalize(term)
        if not key:
            return None
        key = self.aliases.get(key, key)
        if key in self.known:
            return key
        tokens = key.split()
        for start in range(1, len(tokens)):  # drop leading qualifiers
            candidate = " ".join(tokens[start:])
            candidate = self.aliases.get(candidate, candidate)
            if candidate in self.known:
                return candidate
        return None
