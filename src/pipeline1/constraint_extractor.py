"""Pipeline 1 — calorie and exclusion constraint extraction (PRD §12.5–12.6).

Both extractors work on text that has already gone through
``normalize_text`` (lowercase, punctuation collapsed to spaces, apostrophes
removed), so ``don't want`` arrives as ``dont want``.
"""

from __future__ import annotations

import re

_NUMBER = r"(\d+(?:\.\d+)?)"
_UNIT = r"(?:\s*(?:calories?|kcal|cal))?"

# Order matters: a range must win over its own endpoints, and an upper bound
# ("no more than 700") must win over the bare comparator ("more than 700").
CALORIE_RANGE = re.compile(
    # "&" and "," collapse to spaces in normalize_text, so the connector is optional.
    rf"\bbetween\s+{_NUMBER}\s+(?:(?:and|or|n)\s+)?{_NUMBER}{_UNIT}\b"
)
CALORIE_MAX = re.compile(
    rf"\b(?:no\s+more\s+than|not\s+more\s+than|at\s+most|up\s+to|less\s+than|under|below|max(?:imum)?)\s+{_NUMBER}{_UNIT}\b"
)
CALORIE_MIN = re.compile(
    rf"\b(?:at\s+least|no\s+less\s+than|greater\s+than|more\s+than|above|over|minimum|min)\s+{_NUMBER}{_UNIT}\b"
)
CALORIE_TARGET = re.compile(
    rf"(?<!\w)(?:(?:around|about|approximately|roughly|close\s+to|nearly)\s+|~\s*){_NUMBER}\s*{_UNIT}"
)

# "no more than" / "no less than" are calorie bounds, not exclusions; calorie
# spans are blanked before exclusion detection, so any remaining "no" is a cue.
EXCLUSION_CUE = re.compile(
    r"\b(?:without|excluding|exclude|avoiding|avoid|dont\s+want|do\s+not\s+want"
    r"|doesnt\s+want|does\s+not\s+want|didnt\s+want|did\s+not\s+want|no)\b"
)
CLAUSE_END = re.compile(r"[.,;!?]")


def _number(raw: str) -> int | float:
    return float(raw) if "." in raw else int(raw)


def extract_calorie_constraints(text: str) -> tuple[dict[str, int | float | None], list[tuple[int, int]]]:
    """Return the calorie fields and the character spans they occupy."""
    fields: dict[str, int | float | None] = {
        "calorie_target": None,
        "calorie_min": None,
        "calorie_max": None,
    }
    taken: list[tuple[int, int]] = []

    def free(match: re.Match) -> bool:
        return not any(match.start() < end and start < match.end() for start, end in taken)

    for match in CALORIE_RANGE.finditer(text):
        if free(match):
            taken.append(match.span())
            fields["calorie_min"], fields["calorie_max"] = _number(match[1]), _number(match[2])

    for field, pattern in (
        ("calorie_max", CALORIE_MAX),
        ("calorie_min", CALORIE_MIN),
        ("calorie_target", CALORIE_TARGET),
    ):
        for match in pattern.finditer(text):
            if fields[field] is None and free(match):
                taken.append(match.span())
                fields[field] = _number(match[1])
                break

    return fields, taken


def exclusion_spans(masked_text: str) -> list[tuple[int, int]]:
    """Character spans that follow an exclusion cue, up to the next cue/period.

    ``no milk or butter`` yields one span containing ``milk or butter``;
    ``no butter and no cream`` yields two.
    """
    cues = list(EXCLUSION_CUE.finditer(masked_text))
    spans: list[tuple[int, int]] = []
    for index, cue in enumerate(cues):
        end = len(masked_text)
        clause = CLAUSE_END.search(masked_text, cue.end())
        if clause:
            end = clause.start()
        if index + 1 < len(cues) and cues[index + 1].start() < end:
            end = cues[index + 1].start()
        spans.append((cue.end(), end))
    return spans
