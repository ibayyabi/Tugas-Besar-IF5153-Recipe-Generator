"""Fine-tune DistilBERT for pantry-ingredient token classification (Pipeline 1 NER approach).

Trains on the synthetic masked-context queries produced by
``scripts/generate_ner_training_data.py`` (labels: O / B-ING / I-ING). Best
checkpoint by dev span-F1 is saved to ``artifacts/ner_model/`` for the
``NerApproach`` registry entry and the Streamlit UI.

Defaults target Apple-silicon MPS; falls back to CPU automatically.

Usage:
    python scripts/train_ner.py                       # full run (15k rows, 3 epochs)
    python scripts/train_ner.py --max-samples 200     # smoke run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TRAIN = REPO_ROOT / "artifacts" / "ner_training" / "train.jsonl"
DEFAULT_DEV = REPO_ROOT / "artifacts" / "ner_training" / "dev.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "ner_model"

LABELS = ["O", "B-ING", "I-ING"]
MAX_LENGTH = 96


def _align_labels(text: str, spans: list[list[int]], offsets: list[tuple[int, int]]) -> list[int]:
    """BIO labels per subtoken; -100 for special tokens.

    ``spans`` from the generator are per-ingredient PHRASE spans (multi-word
    names arrive as one span), so no gap-based merging happens here: merging on
    single-space gaps would fuse consecutive distinct ingredients after text
    normalization and corrupt the labels.
    """
    labels: list[int] = []
    for start, end in offsets:
        if end <= start:
            labels.append(-100)
            continue
        entity = next((m for m in spans if start < m[1] and end > m[0]), None)
        if entity is None:
            labels.append(0)
        elif start == entity[0]:
            labels.append(1)  # B-ING
        else:
            labels.append(2)  # I-ING
    return labels


def _load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    return rows


def _encode(rows: list[dict], tokenizer) -> list[dict[str, list[int]]]:
    encoded = []
    for row in rows:
        item = tokenizer(
            row["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        labels = _align_labels(row["text"], row["spans"], item["offset_mapping"])
        encoded.append(
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": labels,
                "spans": row["spans"],
            }
        )
    return encoded


def _collate(batch: list[dict[str, list[int]]], tokenizer):
    model_inputs = tokenizer.pad(
        [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
        return_tensors="pt",
    )
    width = model_inputs["input_ids"].shape[1]
    labels = np.full((len(batch), width), -100, dtype=np.int64)
    for i, example in enumerate(batch):
        labels[i, : len(example["labels"])] = example["labels"]
    model_inputs["labels"] = __import__("torch").from_numpy(labels)
    spans = [example["spans"] for example in batch]
    return model_inputs, spans


def _decode_spans(label_ids: list[int], offsets: list[tuple[int, int]]) -> list[list[int]]:
    """BIO decode over real (non-special) subtoken offsets -> char spans."""
    spans: list[list[int]] = []
    open_start: int | None = None
    open_end = 0
    for label, (start, end) in zip(label_ids, offsets):
        if end <= start:
            continue
        if label == 1:  # B-ING
            if open_start is not None:
                spans.append([open_start, open_end])
            open_start, open_end = start, end
        elif label == 2:  # I-ING
            if open_start is None:
                open_start, open_end = start, end
            else:
                open_end = end
        else:
            if open_start is not None:
                spans.append([open_start, open_end])
                open_start = None
    if open_start is not None:
        spans.append([open_start, open_end])
    return spans


def evaluate(model, rows: list[dict[str, list[int]]], tokenizer, device, batch_size: int) -> dict[str, float]:
    import torch

    model.eval()
    token_tp = token_fp = token_fn = 0
    span_tp = span_fp = span_fn = 0
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch, _ = _collate(rows[i : i + batch_size], tokenizer)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1).cpu()

            for j, example in enumerate(rows[i : i + batch_size]):
                offsets = example["offsets"]
                pred_ids = preds[j][: len(offsets)].tolist()
                pred_labels = [p for p, off in zip(pred_ids, offsets) if off[1] > off[0]]
                pred_spans = _decode_spans(
                    pred_labels,
                    [off for off in offsets if off[1] > off[0]],
                )
                gold_labels = [l for l, off in zip(example["labels"], offsets) if off[1] > off[0] and l != -100]
                gold_spans = example["spans"]

                token_tp += sum(1 for p, g in zip(pred_labels, gold_labels) if p != 0 and g != 0 and p == g)
                token_fp += sum(1 for p, g in zip(pred_labels, gold_labels) if p != 0 and g == 0)
                token_fn += sum(1 for p, g in zip(pred_labels, gold_labels) if p == 0 and g != 0)

                pred_set = {tuple(s) for s in pred_spans}
                gold_set = {tuple(s) for s in gold_spans}
                span_tp += len(pred_set & gold_set)
                span_fp += len(pred_set - gold_set)
                span_fn += len(gold_set - pred_set)

    def _f1(tp: int, fp: int, fn: int) -> float:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "token_f1": _f1(token_tp, token_fp, token_fn),
        "span_f1": _f1(span_tp, span_fp, span_fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-samples", type=int, default=None, help="cap train rows (smoke runs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from src.pipeline1.approaches.ner import _pick_device

    device = _pick_device(torch)
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    id2label = {0: "O", 1: "B-ING", 2: "I-ING"}
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=3, id2label=id2label, label2id={v: k for k, v in id2label.items()}
    ).to(device)

    train_rows = _load_rows(args.train, args.max_samples)
    dev_rows = _load_rows(args.dev, None)
    print(f"train rows: {len(train_rows)} | dev rows: {len(dev_rows)}")

    # Keep per-example offsets so evaluation can decode predicted spans.
    train_encoded = _encode(train_rows, tokenizer)
    dev_encoded = _encode(dev_rows, tokenizer)
    for example, row in zip(dev_encoded, dev_rows):
        enc = tokenizer(row["text"], truncation=True, max_length=MAX_LENGTH, return_offsets_mapping=True)
        example["offsets"] = enc["offset_mapping"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(train_encoded) // args.batch_size + 1) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    best_span_f1 = -1.0
    order = list(range(len(train_encoded)))
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        running = 0.0
        for step, start in enumerate(range(0, len(order), args.batch_size)):
            batch_idx = order[start : start + args.batch_size]
            batch, _ = _collate([train_encoded[i] for i in batch_idx], tokenizer)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} step {step}: loss={loss.item():.4f}", flush=True)
        metrics = evaluate(model, dev_encoded, tokenizer, device, args.batch_size)
        print(
            f"epoch {epoch + 1} done | train_loss={running / (len(order) / args.batch_size + 1):.4f} "
            f"| dev token_f1={metrics['token_f1']:.4f} span_f1={metrics['span_f1']:.4f}",
            flush=True,
        )
        if metrics["span_f1"] > best_span_f1:
            best_span_f1 = metrics["span_f1"]
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            (args.output_dir / "eval_report.json").write_text(
                json.dumps({"epoch": epoch + 1, **metrics}, indent=2), encoding="utf-8"
            )
            print(f"  saved new best (span_f1={best_span_f1:.4f}) to {args.output_dir}")

    print(f"done; best dev span_f1={best_span_f1:.4f}")


if __name__ == "__main__":
    main()
