"""Canonical ingredient normalization shared by every pipeline.

Two collaborators live here on purpose:

* ``Vocabulary`` — the closed query-time vocabulary used by Pipeline 1. Only
  ingredients with a calorie treatment in ``data/processed/`` plus the curated
  alias table can be recognised; longest surface phrase wins.
* ``IngredientNormalizer`` — the offline normalizer used by corpus preprocessing
  and Pipeline 2 (alias resolution, suffix-trim scanning, recipe vocabulary).

Both canonicalize ingredient strings through ``normalize_ingredient`` (fold +
head-noun singularization). ``normalize_text`` stays a pure query-time fold and
must not singularize: it feeds whole user queries, including trailing exclusion
clauses such as "no cheese" that would corrupt under naive stemming. ``~`` is
kept there because the calorie constraint grammar reads ``~500`` as an
approximate target. Rules mirror ``scripts/ingredient_tables.py``; keep the two
in sync through ``data/processed/ingredient_aliases.csv``.
"""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DATA_DIR = DEFAULT_PROCESSED_DIR
DEFAULT_VOCAB_PATH = ROOT / "artifacts" / "ingredient_vocabulary.pkl"

CALORIE_TABLES = ("calorie_baseline.csv", "supplement_calories.csv", "zero_calorie_ingredients.csv")
STAPLE_TABLE = "pantry_staples.csv"
ALIAS_TABLE = "ingredient_aliases.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def normalize_text(text: str) -> str:
    """Lowercase, drop apostrophes, collapse everything that is not a word or ~."""
    text = text.lower().replace("\u2019", "'").replace("'", "")
    text = re.sub(r"[^a-z0-9~]+", " ", text)
    return text.strip()


def singularize(word: str) -> str:
    # suffix rules, not WordNet lemmatization; the closed vocabulary and
    # alias table hide the gap. Swap in NLTK when the vocabulary goes open-domain.
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


def normalize_ingredient(name: str) -> str:
    """Normalize an ingredient phrase; only the head noun is plural-folded."""
    tokens = normalize_text(name).split()
    if not tokens:
        return ""
    tokens[-1] = singularize(tokens[-1])
    return " ".join(tokens)


class Vocabulary:
    """Surface form -> canonical ingredient, longest match wins."""

    def __init__(self, surface_to_canonical: dict[str, str]) -> None:
        self.surface_to_canonical = surface_to_canonical
        self.max_tokens = max((len(s.split()) for s in surface_to_canonical), default=1)

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR) -> "Vocabulary":
        canonical = set()
        for table in CALORIE_TABLES + (STAPLE_TABLE,):
            canonical |= {normalize_ingredient(row["ingredient"]) for row in read_csv_rows(data_dir / table)}
        aliases = {
            normalize_ingredient(row["alias"]): normalize_ingredient(row["canonical"])
            for row in read_csv_rows(data_dir / ALIAS_TABLE)
        }
        surface = {name: aliases.get(name, name) for name in canonical}
        surface.update(aliases)
        return cls(surface)

    def lookup(self, key: str) -> str | None:
        return self.surface_to_canonical.get(key)

    def canonical(self, surface: str) -> str | None:
        return self.lookup(normalize_ingredient(surface))


class IngredientNormalizer:
    """Offline alias/qualifier resolution for corpus preprocessing and Pipeline 2."""

    def __init__(self, processed_dir: Path | str = DEFAULT_PROCESSED_DIR) -> None:
        self.processed_dir = Path(processed_dir)
        self.baseline = {
            normalize_ingredient(r["ingredient"])
            for r in read_csv_rows(self.processed_dir / "calorie_baseline.csv")
        }
        self.supplement = {
            normalize_ingredient(r["ingredient"])
            for r in read_csv_rows(self.processed_dir / "supplement_calories.csv")
        }
        self.zero = {
            normalize_ingredient(r["ingredient"])
            for r in read_csv_rows(self.processed_dir / "zero_calorie_ingredients.csv")
        }
        self.aliases = {
            normalize_ingredient(r["alias"]): normalize_ingredient(r["canonical"])
            for r in read_csv_rows(self.processed_dir / ALIAS_TABLE)
        }
        self.known = self.baseline | self.supplement | self.zero

    def resolve(self, term: str) -> str | None:
        key = normalize_ingredient(term)
        if not key:
            return None
        key = self.aliases.get(key, key)
        if key in self.known:
            return key
        tokens = key.split()
        for start in range(1, len(tokens)):
            candidate = " ".join(tokens[start:])
            candidate = self.aliases.get(candidate, candidate)
            if candidate in self.known:
                return candidate
        return None

    def canonicalize(self, term: str, drop_qualifiers: bool = True) -> str:
        """Normalize, alias-map, and (optionally) drop leading qualifiers.

        Same fallback contract as :meth:`normalize_ingredient`: returns the
        alias-mapped normalized form when nothing known matches.
        """
        key = normalize_ingredient(term)
        if not key:
            return ""
        mapped = self.aliases.get(key, key)
        if not self.known or mapped in self.known or not drop_qualifiers:
            return mapped
        tokens = key.split()
        for start in range(1, len(tokens)):
            candidate = " ".join(tokens[start:])
            candidate_mapped = self.aliases.get(candidate, candidate)
            if candidate_mapped in self.known:
                return candidate_mapped
        return mapped

    def canonicalize_list(
        self,
        terms: Iterable[str],
        drop_qualifiers: bool = True,
    ) -> list[str]:
        """Deduplicated, order-preserving list of canonical ingredient names."""
        seen: dict[str, None] = {}
        for term in terms:
            canonical = self.canonicalize(term, drop_qualifiers=drop_qualifiers)
            if canonical:
                seen.setdefault(canonical, None)
        return list(seen)

    def normalize_ingredient(self, term: str) -> str:
        return self.canonicalize(term, drop_qualifiers=True)

    def extract_canonical_ingredients(
        self,
        raw_terms: list[str],
        known_only: bool = False,
    ) -> list[str]:
        seen: dict[str, None] = {}
        for term in raw_terms:
            if not isinstance(term, str):
                continue
            canonical = self.resolve(term) if known_only else self.normalize_ingredient(term)
            if canonical:
                seen.setdefault(canonical, None)
        return list(seen)

    def build_vocabulary(self, recipe_ingredient_lists: list[list[str]]) -> list[str]:
        vocab: set[str] = set(self.known)
        for ingredients in recipe_ingredient_lists:
            for item in ingredients:
                if item:
                    vocab.add(item)
        return sorted(vocab)

    @staticmethod
    def save_vocabulary(vocabulary: list[str], path: Path | str = DEFAULT_VOCAB_PATH) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            pickle.dump(vocabulary, fh)
        return out_path

    @staticmethod
    def load_vocabulary(path: Path | str = DEFAULT_VOCAB_PATH) -> list[str]:
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
