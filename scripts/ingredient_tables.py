"""Dataset tooling normalization — the single source lives in src/preprocessing.

Re-exports the same rules Pipeline 1-3 use (fold + head-noun singularization,
alias resolution) so dataset tooling can no longer drift from pipeline code.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.ingredient_normalizer import (  # noqa: E402
    IngredientNormalizer as Resolver,
    normalize_ingredient as normalize,
    read_csv_rows as read_rows,
    singularize,
)

__all__ = ["PROCESSED", "Resolver", "normalize", "read_rows", "singularize"]
