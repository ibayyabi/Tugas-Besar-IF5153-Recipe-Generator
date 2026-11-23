"""BERT-based token-classification approach for Pipeline 1 ingredient extraction.

Fine-tuned DistilBERT trained on synthetic pantry queries (see
``scripts/generate_ner_training_data.py`` and ``scripts/train_ner.py``). The
model sees the same masked matching text the rule-based approaches see:
calorie and exclusion spans are already blanked by the query parser.

Transformers/torch are imported lazily so the lexical approaches keep working
without the deep-learning stack installed.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline1.approaches.base import ApproachTrace, IngredientApproach, IngredientMatch
from src.preprocessing.ingredient_normalizer import Vocabulary, normalize_ingredient

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "ner_model"
# DistilBERT's positional limit; training used short queries, so this only
# stops long inputs from being silently cut at 96 tokens.
MAX_LENGTH = 512


def _pick_device(torch) -> "torch.device":
    """CUDA (NVIDIA) > MPS (Apple silicon) > CPU — portable by default."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NerApproach(IngredientApproach):
    """Token classification (B-ING/I-ING/O) with vocabulary canonicalization."""

    name = "ner"
    description = "Fine-tuned DistilBERT token classification over the query, canonicalized through the closed vocabulary."

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._tokenizer = None
        self._model = None
        self._device = None

    @property
    def available(self) -> bool:
        return (self.model_dir / "config.json").exists()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.available:
            raise FileNotFoundError(
                f"NER model artifacts not found at {self.model_dir}; run scripts/train_ner.py first"
            )
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        # eager attention is required for output_attentions (SDPA returns none)
        self._model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir, attn_implementation="eager"
        )
        self._device = _pick_device(torch)
        self._model.to(self._device)
        self._model.eval()

    def _predict(self, text: str):
        """Tokenize + tag one text; returns (offsets, tag_ids, probabilities)."""
        import torch

        self._load()
        encoding = self._tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = self._model(
                input_ids=encoding["input_ids"].to(self._device),
                attention_mask=encoding["attention_mask"].to(self._device),
            ).logits[0]
        probs = torch.softmax(logits, dim=-1)
        tag_ids = probs.argmax(dim=-1).tolist()
        confidences = probs.max(dim=-1).values.tolist()
        return offsets, tag_ids, confidences

    def trace(self, text: str, vocab: Vocabulary) -> ApproachTrace:
        offsets, tag_ids, confidences = self._predict(text)
        id2label = self._model.config.id2label

        token_evidence: list[dict[str, object]] = []
        matches: list[IngredientMatch] = []
        uncanonicalized: list[dict[str, object]] = []
        unmatched: list[str] = []

        open_start = open_end = None
        open_probs: list[float] = []

        def close_span() -> None:
            nonlocal open_start, open_end, open_probs
            if open_start is None:
                return
            surface = text[open_start:open_end]
            canonical = vocab.canonical(normalize_ingredient(surface))
            confidence = sum(open_probs) / len(open_probs) if open_probs else None
            if canonical:
                matches.append(
                    IngredientMatch(
                        start=open_start,
                        end=open_end,
                        surface=surface,
                        canonical=canonical,
                        method="ner",
                        score=round(confidence, 4) if confidence is not None else None,
                    )
                )
            else:
                uncanonicalized.append(
                    {"surface": surface, "confidence": round(confidence, 4) if confidence else None}
                )
            open_start = open_end = None
            open_probs = []

        for (start, end), tag_id, conf in zip(offsets, tag_ids, confidences):
            label = id2label[tag_id]
            token = text[start:end]
            if end <= start:  # special tokens (CLS/SEP) carry (0, 0) offsets
                token_evidence.append({"token": token, "tag": label, "prob": round(conf, 4)})
                continue
            token_evidence.append(
                {"token": token, "tag": label, "prob": round(conf, 4), "start": start, "end": end}
            )
            if label.endswith("B-ING"):
                close_span()
                open_start, open_end, open_probs = start, end, [conf]
            elif label.endswith("I-ING") and open_start is not None:
                open_end = end
                open_probs.append(conf)
            elif label.endswith("I-ING"):
                open_start, open_end, open_probs = start, end, [conf]
            else:
                close_span()
                if token:
                    unmatched.append(token)
        close_span()

        seen: set[str] = set()
        ordered_matches: list[IngredientMatch] = []
        for match in matches:
            if match.canonical not in seen:
                ordered_matches.append(match)
                seen.add(match.canonical)
        return ApproachTrace(
            approach=self.name,
            text=text,
            matches=ordered_matches,
            unmatched_tokens=unmatched,
            uncanonicalized=uncanonicalized,
            extras={"tokens": token_evidence, "model_dir": str(self.model_dir)},
        )

    def attention(self, text: str) -> dict[str, object]:
        """Last-layer attention averaged over heads, for the UI heatmap."""
        import torch

        self._load()
        encoding = self._tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            outputs = self._model(
                input_ids=encoding["input_ids"].to(self._device),
                attention_mask=encoding["attention_mask"].to(self._device),
                output_attentions=True,
            )
        # last layer, mean over heads -> [seq, seq]
        matrix = outputs.attentions[-1][0].mean(dim=0).cpu().tolist()
        tokens = self._tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        return {
            "tokens": tokens,
            "matrix": matrix,
            "offsets": offsets,
            "layers": len(outputs.attentions),
            "heads": self._model.config.num_attention_heads,
        }
