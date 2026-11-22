"""Small rendering utilities shared across the Streamlit tabs."""

from __future__ import annotations

import html
from typing import Iterable

# PRD Section 24 default weights — mirrored for display only; the reranker owns the truth.
COLOR_CALORIE = "#f9d77e"  # amber: calorie constraint spans
COLOR_EXCLUSION = "#f6a6a6"  # red: exclusion clauses
COLOR_INGREDIENT = "#bfe8bf"  # green: matched ingredient mentions
COLOR_NER = "#9fd3f0"  # blue: NER-detected mentions


def _escape(text: str) -> str:
    return html.escape(text)


def highlight_spans(text: str, spans: list[tuple[int, int]], color: str, label: str) -> str:
    """Wrap ``spans`` of ``text`` in a colored inline mark (HTML-escaped)."""
    if not spans:
        return _escape(text)
    pieces: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        if start < cursor or end > len(text):
            continue
        pieces.append(_escape(text[cursor:start]))
        pieces.append(f'<span style="background:{color};border-radius:3px;padding:0 2px" title="{label}">{_escape(text[start:end])}</span>')
        cursor = end
    pieces.append(_escape(text[cursor:]))
    return "".join(pieces)


def highlight_matches(text: str, matches: Iterable, base_color: str = COLOR_INGREDIENT) -> str:
    """Highlight approach matches; NER/fuzzy matches get their own tint + tooltip."""
    ordered = sorted(matches, key=lambda m: (m.start, m.end))
    pieces: list[str] = []
    cursor = 0
    for match in ordered:
        if match.start < cursor or match.end > len(text):
            continue
        pieces.append(_escape(text[cursor:match.start]))
        title = f"{match.method}: {match.surface} -> {match.canonical}"
        if match.score is not None:
            title += f" ({match.score:.2f})"
        color = COLOR_NER if match.method == "ner" else base_color
        pieces.append(
            f'<span style="background:{color};border-radius:3px;padding:0 2px" title="{_escape(title)}">'
            f"{_escape(text[match.start:match.end])}</span>"
        )
        cursor = match.end
    pieces.append(_escape(text[cursor:]))
    return "".join(pieces) if ordered else _escape(text)


def score_components(ranked, config) -> dict[str, float]:
    """Weighted contributions behind ``final_score`` (PRD §24) for charting."""
    coverage = ranked.coverage
    calorie = ranked.calorie
    return {
        f"α·tfidf ({config.alpha})": round(config.alpha * ranked.tfidf_score, 4),
        f"β·coverage ({config.beta})": round(config.beta * coverage.weighted_coverage, 4),
        f"γ·calorie ({config.gamma})": round(
            config.gamma * (calorie.score if calorie.estimated_calories is not None else config.unknown_calorie_score), 4
        ),
        f"−δ·missing ({config.delta})": round(-config.delta * coverage.missing_penalty, 4),
    }
