"""Vocabulary-scanning approaches: production rule matcher and the exact-surface ablation."""

from __future__ import annotations

import re

from src.pipeline1.approaches.base import ApproachTrace, IngredientApproach, IngredientMatch
from src.preprocessing.ingredient_normalizer import Vocabulary, singularize

TOKEN = re.compile(r"[a-z0-9]+")


class RuleApproach(IngredientApproach):
    """Production longest-match scan: aliases, plural folding, compound splitting."""

    name = "rule"
    description = "Longest-match vocabulary scan with aliases, plural folding, and compound splitting (production baseline)."

    def trace(self, text: str, vocab: Vocabulary) -> ApproachTrace:
        tokens = [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]
        matches: list[IngredientMatch] = []
        index = 0
        while index < len(tokens):
            remaining = len(tokens) - index
            for size in range(min(vocab.max_tokens, remaining), 0, -1):
                window = [token for token, _, _ in tokens[index : index + size]]
                window[-1] = singularize(window[-1])
                canonical = vocab.lookup(" ".join(window))
                if canonical:
                    matches.append(
                        IngredientMatch(
                            start=tokens[index][1],
                            end=tokens[index + size - 1][2],
                            surface=" ".join(token for token, _, _ in tokens[index : index + size]),
                            canonical=canonical,
                            method="vocabulary",
                        )
                    )
                    index += size
                    break
            else:
                token, start, end = tokens[index]
                joined = _split_compound(token, vocab)
                if joined:
                    matches.append(
                        IngredientMatch(
                            start=start, end=end, surface=token, canonical=joined, method="compound-split"
                        )
                    )
                index += 1
        matched_chars = [(m.start, m.end) for m in matches]
        unmatched = [
            token
            for token, start, end in tokens
            if not any(start < hit_end and end > hit_start for hit_start, hit_end in matched_chars)
        ]
        return ApproachTrace(approach=self.name, text=text, matches=matches, unmatched_tokens=unmatched)


class SurfaceExactApproach(RuleApproach):
    """Ablation: exact vocabulary surfaces only — no plural folding or compound recovery."""

    name = "surface-exact"
    description = "Ablation: exact vocabulary surface matching; no plural folding, aliases, or compound splitting."

    def trace(self, text: str, vocab: Vocabulary) -> ApproachTrace:
        tokens = [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]
        matches: list[IngredientMatch] = []
        index = 0
        while index < len(tokens):
            remaining = len(tokens) - index
            for size in range(min(vocab.max_tokens, remaining), 0, -1):
                key = " ".join(token for token, _, _ in tokens[index : index + size])
                canonical = vocab.lookup(key)
                if canonical:
                    matches.append(
                        IngredientMatch(
                            start=tokens[index][1],
                            end=tokens[index + size - 1][2],
                            surface=key,
                            canonical=canonical,
                            method="vocabulary",
                        )
                    )
                    index += size
                    break
            else:
                index += 1
        matched_chars = [(m.start, m.end) for m in matches]
        unmatched = [
            token
            for token, start, end in tokens
            if not any(start < hit_end and end > hit_start for hit_start, hit_end in matched_chars)
        ]
        return ApproachTrace(approach=self.name, text=text, matches=matches, unmatched_tokens=unmatched)


def _split_compound(token: str, vocab: Vocabulary) -> str | None:
    """``sweetpotato`` -> ``sweet potato``: a joined spelling the alias table
    does not spell out. Split only when both halves are vocabulary words."""
    folded = singularize(token)
    for cut in range(1, len(folded)):
        canonical = vocab.lookup(f"{folded[:cut]} {singularize(folded[cut:])}")
        if canonical:
            return canonical
    return None
