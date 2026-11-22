"""Approach contract and evidence records for Pipeline 1 matching."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.preprocessing.ingredient_normalizer import Vocabulary


@dataclass
class IngredientMatch:
    """One ingredient mention found in the (masked) matching text."""

    start: int
    end: int
    surface: str
    canonical: str
    method: str  # vocabulary | compound-split | difflib | char-ngram | ner
    score: float | None = None  # similarity or model confidence, when applicable


@dataclass
class ApproachTrace:
    """Intermediate evidence for one matching context; rendered by the UI."""

    approach: str
    text: str  # the exact text the approach matched on (already masked)
    matches: list[IngredientMatch] = field(default_factory=list)
    unmatched_tokens: list[str] = field(default_factory=list)
    uncanonicalized: list[dict[str, object]] = field(default_factory=list)  # NER-only OOV hits
    extras: dict[str, object] = field(default_factory=dict)  # approach-specific evidence

    @property
    def ingredients(self) -> list[str]:
        seen: dict[str, None] = {}
        for match in self.matches:
            seen.setdefault(match.canonical, None)
        return list(seen)


class IngredientApproach:
    """Strategy interface; subclasses are stateless unless they load a model."""

    name: str = ""
    description: str = ""

    def extract(self, text: str, vocab: Vocabulary) -> list[str]:
        return self.trace(text, vocab).ingredients

    def trace(self, text: str, vocab: Vocabulary) -> ApproachTrace:
        raise NotImplementedError
