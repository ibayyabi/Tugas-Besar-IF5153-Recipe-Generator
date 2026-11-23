"""Pipeline 1 — calorie and exclusion constraint extraction (PRD §12.5–12.6).

``normalize_query_text`` folds a query while keeping what the constraint grammar
needs: decimal points and numeric signs survive, and a parallel boundary string
records clause punctuation and hyphenation without leaking markers into the
ingredient matcher.
"""

from __future__ import annotations

import re

_CALORIE_UNIT = r"(?:calories?|kcal|cal)"
# A bare number after "under/over/around" is only a calorie constraint when it
# is not followed by an obvious time/volume/count unit ("under 20 minutes",
# "over 2 cups", "no more than 3 eggs").
_NON_CALORIE_UNIT = (
    r"(?:minutes?|mins?|hours?|hrs?|seconds?|secs?|days?"
    r"|cups?|tablespoons?|tbsp|teaspoons?|tsp"
    r"|grams?|kilograms?|kg|g|milligrams?|mg|milliliters?|ml|liters?|l"
    r"|ounces?|oz|pounds?|lbs?|slices?|pieces?|servings?"
    r"|eggs?|cloves?|cans?|packages?|pkg|bunches?|sprigs?|heads?|stalks?"
    r"|people|persons?|dollars?|am|pm|oclock|degrees?"
    r"|large|medium|small|whole)"
)
_NUMBER = r"(-?\d+(?:\.\d+)?)"
_UNIT = rf"(?:\s*{_CALORIE_UNIT})?"
_NON_CALORIE_AFTER = re.compile(rf"\s*{_NON_CALORIE_UNIT}\b")

# Order matters: a range must win over its own endpoints, and an upper bound
# ("no more than 700") must win over the bare comparator ("more than 700").
CALORIE_RANGE = re.compile(
    # "&" and "," collapse to spaces in normalize_query_text, so the connector is optional.
    rf"\b(?:between|from)\s+{_NUMBER}\s+(?:(?:and|or|n|to|through)\s+)?{_NUMBER}{_UNIT}\b"
)
# "400 to 600 calories" without a leading "between"/"from".
CALORIE_RANGE_TO = re.compile(rf"(?<!\w){_NUMBER}\s+(?:to|through)\s+{_NUMBER}{_UNIT}\b")
CALORIE_MAX = re.compile(
    rf"\b(?:no\s+more\s+than|not\s+more\s+than|no\s+greater\s+than|not\s+greater\s+than"
    rf"|less\s+than\s+or\s+equal\s+to|at\s+most|up\s+to|not\s+exceeding"
    rf"|less\s+than|under|below|max(?:imum)?)\s+{_NUMBER}{_UNIT}\b"
)
CALORIE_MIN = re.compile(
    rf"\b(?:at\s+least|no\s+less\s+than|not\s+less\s+than|no\s+smaller\s+than"
    rf"|greater\s+than\s+or\s+equal\s+to|greater\s+than|more\s+than|above|over|minimum|min)"
    rf"\s+{_NUMBER}{_UNIT}\b"
)
CALORIE_TARGET = re.compile(
    rf"(?<!\w)(?:(?:around|about|approximately|roughly|close\s+to|nearly)\s+|~\s*)"
    rf"{_NUMBER}{_UNIT}\b"
)
# Suffix forms are unambiguous only when the calorie unit is spelled out:
# "500 calories max", "500 kcal or fewer".
CALORIE_MAX_SUFFIX = re.compile(
    rf"(?<!\w){_NUMBER}\s*{_CALORIE_UNIT}\s*(?:max(?:imum)?|or\s+(?:less|fewer|under)|at\s+most)\b"
)
CALORIE_MIN_SUFFIX = re.compile(
    rf"(?<!\w){_NUMBER}\s*{_CALORIE_UNIT}\s*(?:min(?:imum)?|or\s+(?:more|over|greater)|at\s+least)\b"
)

# "no more than" / "no less than" are calorie bounds, not exclusions; calorie
# spans are blanked before exclusion detection, so any remaining "no" is a cue.
EXCLUSION_CUE = re.compile(
    r"\b(?:without|excluding|exclude|excluded|avoiding|avoid"
    r"|dont\s+want|do\s+not\s+want|doesnt\s+want|does\s+not\s+want|didnt\s+want|did\s+not\s+want"
    r"|dont\s+eat|do\s+not\s+eat|doesnt\s+eat|does\s+not\s+eat|didnt\s+eat|did\s+not\s+eat"
    r"|dont\s+have|do\s+not\s+have|doesnt\s+have|does\s+not\s+have|didnt\s+have|did\s+not\s+have"
    r"|cant\s+eat|cannot\s+eat|can\s+not\s+eat|allergic\s+to|except(?:\s+for)?"
    r"|skip|skipping|no)\b"
)
CLAUSE_END = re.compile(r"[.,;!?]")
CLAUSE_MARKS = ".,;!?"
HYPHEN_MARKS = "-–—"
# "no cheese and I have potato": the availability clause starts here.
TRANSITION_BREAK = re.compile(
    r"\b(?:but|however|then|and\s+(?=(?:(?:i|we|you)(?:ve)?|there)\s+"
    r"(?:also\s+)?(?:have|has|got|need|want|like|is|are)))\b"
)
FREE_SUFFIX = re.compile(r"\bfree\b")
# "no more than" / "no less than" are bounds, never exclusion cues.
BOUND_PHRASE = re.compile(r"\s+(?:more|less|greater|smaller)\s+than\b")


def _is_digit(char: str) -> bool:
    return "0" <= char <= "9"


def normalize_query_text(text: str) -> tuple[str, str]:
    """Fold a query into ``(text, boundaries)`` for constraint extraction.

    ``text`` equals ``normalize_text`` for ordinary input, but keeps decimal
    points and numeric signs so ``20.5`` and ``-500`` are not split. Every
    character in ``boundaries`` aligns with ``text``: ``|`` marks a clause
    punctuation run, ``-`` marks a hyphen, and a space marks everything else.
    """
    lowered = text.lower().replace("\u2019", "'").replace("'", "")
    chars: list[str] = []
    marks: list[str] = []
    pending = False
    pending_clause = False
    pending_hyphen = False
    for index, char in enumerate(lowered):
        previous = lowered[index - 1] if index else ""
        following = lowered[index + 1] if index + 1 < len(lowered) else ""
        allowed = ("a" <= char <= "z") or _is_digit(char) or char == "~"
        if char == "." and _is_digit(previous) and _is_digit(following):
            allowed = True
        if (
            char == "-"
            and _is_digit(following)
            and not (previous.isascii() and previous.isalnum())
        ):
            allowed = True
        if allowed:
            if chars and pending:
                chars.append(" ")
                marks.append("|" if pending_clause else "-" if pending_hyphen else " ")
            chars.append(char)
            marks.append(" ")
            pending = pending_clause = pending_hyphen = False
        else:
            pending = True
            if char in CLAUSE_MARKS:
                pending_clause = True
            elif char in HYPHEN_MARKS:
                pending_hyphen = True
    return "".join(chars), "".join(marks)


def _number(raw: str) -> int | float | None:
    value = float(raw) if "." in raw else int(raw)
    return value if value > 0 else None


def extract_calorie_constraints(text: str) -> tuple[dict[str, int | float | None], list[tuple[int, int]]]:
    """Return the calorie fields and the character spans they occupy.

    Non-positive values (``0``, ``-500``) are consumed as spans but left unset:
    they are not meaningful calorie constraints.
    """
    fields: dict[str, int | float | None] = {
        "calorie_target": None,
        "calorie_min": None,
        "calorie_max": None,
    }
    taken: list[tuple[int, int]] = []

    def free(match: re.Match) -> bool:
        return not any(match.start() < end and start < match.end() for start, end in taken)

    def is_non_calorie(match: re.Match) -> bool:
        """True for "under 20 minutes" / "no more than 3 eggs" style bounds."""
        return _NON_CALORIE_AFTER.match(text, match.end()) is not None

    for pattern in (CALORIE_RANGE, CALORIE_RANGE_TO):
        for match in pattern.finditer(text):
            if not free(match) or is_non_calorie(match):
                continue
            taken.append(match.span())
            low, high = _number(match[1]), _number(match[2])
            if low is not None and high is not None:
                fields["calorie_min"], fields["calorie_max"] = (
                    (low, high) if low <= high else (high, low)
                )

    for field, patterns in (
        ("calorie_max", (CALORIE_MAX, CALORIE_MAX_SUFFIX)),
        ("calorie_min", (CALORIE_MIN, CALORIE_MIN_SUFFIX)),
        ("calorie_target", (CALORIE_TARGET,)),
    ):
        candidates = sorted(
            (match for pattern in patterns for match in pattern.finditer(text)),
            key=lambda match: match.start(),
        )
        for match in candidates:
            if not free(match) or is_non_calorie(match):
                continue
            taken.append(match.span())
            value = _number(match[1])
            if fields[field] is None and value is not None:
                fields[field] = value

    return fields, taken


def exclusion_spans(masked_text: str, boundaries: str | None = None) -> list[tuple[int, int]]:
    """Character spans that follow an exclusion cue, up to the next clause.

    ``no milk or butter`` yields one span containing ``milk or butter``;
    ``no butter and no cream`` yields two; ``no cheese, I have potato`` stops at
    the comma. ``boundaries`` is the parallel string from
    :func:`normalize_query_text`; without it the legacy period/comma scan runs.
    """
    cues = list(EXCLUSION_CUE.finditer(masked_text))
    spans: list[tuple[int, int]] = []
    for index, cue in enumerate(cues):
        # "no-bake" is a compound adjective, not an exclusion directive.
        if (
            cue.group() == "no"
            and boundaries is not None
            and cue.end() < len(boundaries)
            and boundaries[cue.end()] == "-"
        ):
            continue
        # "no more than 3 eggs" is a bound, not an exclusion of eggs.
        if cue.group() == "no" and BOUND_PHRASE.match(masked_text, cue.end()):
            continue

        end = len(masked_text)
        if boundaries is not None:
            clause = boundaries.find("|", cue.end())
            if clause != -1:
                end = clause
        else:
            clause = CLAUSE_END.search(masked_text, cue.end())
            if clause:
                end = clause.start()

        transition = TRANSITION_BREAK.search(masked_text, cue.end())
        if transition and transition.start() < end:
            end = transition.start()
        if index + 1 < len(cues) and cues[index + 1].start() < end:
            end = cues[index + 1].start()
        spans.append((cue.end(), end))

    if boundaries is not None:
        # "cheese-free" / "sugar free": exclude the token before "free".
        for match in FREE_SUFFIX.finditer(masked_text):
            separator = match.start() - 1
            if separator < 0 or boundaries[separator] == "|":
                continue
            if masked_text[match.end() :].lstrip().startswith("range"):
                continue
            start = masked_text.rfind(" ", 0, separator) + 1
            if start < separator:
                spans.append((start, separator))

    spans.sort()
    return spans
