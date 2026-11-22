"""Fuzzy-fallback approaches: production exact matches first, then similarity recovery.

Moved here from ``evaluation/experiment.py`` so the CLI harness, the tests, and
the UI share one implementation. Both variants keep the production exact-match
spans untouched and only try to rescue uncovered tokens.
"""

from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from dataclasses import dataclass

from src.pipeline1.approaches.base import ApproachTrace, IngredientApproach, IngredientMatch
from src.pipeline1.approaches.rule import RuleApproach
from src.preprocessing.ingredient_normalizer import Vocabulary, singularize

TOKEN = re.compile(r"[a-z0-9]+")


def _char_ngrams(value: str) -> Counter[str]:
    return Counter(value[i : i + n] for n in (2, 3, 4) for i in range(len(value) - n + 1))


@dataclass(frozen=True)
class _FuzzySpec:
    method: str  # difflib | char-ngram
    threshold: float


class FuzzyApproach(IngredientApproach):
    """Exact vocabulary scan, then similarity-match uncovered single tokens."""

    name = ""  # instance-level: difflib-0.82 / char-ngram-0.60
    description = ""

    difflib_name = "difflib-0.82"
    difflib_description = "Production matcher plus difflib.SequenceMatcher fallback for unmatched tokens (threshold 0.82)."
    ngram_name = "char-ngram-0.60"
    ngram_description = "Production matcher plus character 2-4-gram cosine fallback for unmatched tokens (threshold 0.60)."

    def __init__(self, spec: _FuzzySpec) -> None:
        self.spec = spec
        self.name = f"{spec.method}-{spec.threshold:.2f}"
        self.description = (
            self.difflib_description if spec.method == "difflib" else self.ngram_description
        )
        self._exact = RuleApproach()
        self._grams: dict[str, Counter[str]] | None = None
        self._norms: dict[str, float] | None = None

    @classmethod
    def for_name(cls, name: str) -> "FuzzyApproach":
        if name == cls.difflib_name:
            return cls(_FuzzySpec("difflib", 0.82))
        if name == cls.ngram_name:
            return cls(_FuzzySpec("char-ngram", 0.60))
        raise KeyError(f"unknown fuzzy approach: {name!r}")

    def _score_token(self, token: str, keys: list[tuple[str, str]]) -> tuple[float, str, str]:
        if self.spec.method == "difflib":
            scored = ((difflib.SequenceMatcher(None, token, key).ratio(), key, value) for key, value in keys)
            return max(scored, default=(0.0, "", ""))
        if self._grams is None:
            self._grams = {key: _char_ngrams(key) for key, _ in keys}
            self._norms = {
                key: math.sqrt(sum(n * n for n in vector.values())) for key, vector in self._grams.items()
            }
        left = _char_ngrams(token)
        left_norm = math.sqrt(sum(n * n for n in left.values()))
        scored = (
            (
                sum(count * self._grams[key].get(gram, 0) for gram, count in left.items())
                / (left_norm * self._norms[key])
                if left_norm
                else 0.0,
                key,
                value,
            )
            for key, value in keys
        )
        return max(scored, default=(0.0, "", ""))

    def trace(self, text: str, vocab: Vocabulary) -> ApproachTrace:
        exact = self._exact.trace(text, vocab)
        matches = list(exact.matches)
        covered = [(m.start, m.end) for m in matches]
        keys = self._single_word_keys(vocab)
        seen = {m.canonical: None for m in matches}
        unmatched: list[str] = []
        for token_match in TOKEN.finditer(text):
            start, end = token_match.span()
            if any(start < hit_end and end > hit_start for hit_start, hit_end in covered):
                continue
            token = singularize(token_match.group())
            unmatched.append(token)
            score, _, canonical = self._score_token(token, keys)
            if canonical and score >= self.spec.threshold and canonical not in seen:
                matches.append(
                    IngredientMatch(
                        start=start,
                        end=end,
                        surface=token_match.group(),
                        canonical=canonical,
                        method=f"fuzzy-{self.spec.method}",
                        score=round(score, 4),
                    )
                )
                seen[canonical] = None
        return ApproachTrace(
            approach=self.name,
            text=text,
            matches=matches,
            unmatched_tokens=unmatched,
            extras={"threshold": self.spec.threshold, "fallback_method": self.spec.method},
        )

    def _single_word_keys(self, vocab: Vocabulary) -> list[tuple[str, str]]:
        return [(key, value) for key, value in vocab.surface_to_canonical.items() if " " not in key]
