"""Encode the processed recipe corpus with a sentence-transformer (Pipeline 2 dense approach).

Reads ``data/processed/recipes.parquet`` (same corpus as the TF-IDF approach),
encodes each recipe's canonical ingredient text, and stores L2-normalized
vectors plus a metadata sidecar under ``artifacts/``.

Usage:
    python scripts/encode_dense_corpus.py                       # default model + corpus
    python scripts/encode_dense_corpus.py --model all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CORPUS = REPO_ROOT / "data" / "processed" / "recipes.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts"


def encode_corpus(corpus_path: Path, model_name: str, out_dir: Path, batch_size: int = 128) -> None:
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    corpus = pd.read_parquet(corpus_path)

    def to_text(row) -> str:
        text = str(row.get("canonical_text") or "").strip()
        if not text:
            text = " ".join(row["canonical_ingredients"])
        return text

    texts = [to_text(row) for _, row in corpus.iterrows()]
    print(f"encoding {len(texts)} recipes with {model_name}")

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "dense_vectors.npy", vectors)
    meta = {
        "model": model_name,
        "corpus_path": str(corpus_path),
        "column": "canonical_text",
        "normalized": True,
        "dim": int(vectors.shape[1]),
        "n_recipes": int(vectors.shape[0]),
    }
    (out_dir / "dense_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {vectors.shape} vectors to {out_dir / 'dense_vectors.npy'}")
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    encode_corpus(args.corpus, args.model, args.out_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
