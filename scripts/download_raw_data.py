#!/usr/bin/env python3
"""Fetch every external dataset CookFromPantry needs (stdlib only).

Dataset inventory and rationale: data/README.md.

    python scripts/download_raw_data.py --calories          # USDA FoodData Central (~7 MB)
    python scripts/download_raw_data.py --recipes sample    # ~7 MB random sample
    python scripts/download_raw_data.py --recipes full      # 2.3 GB full RecipeNLG corpus
    python scripts/download_raw_data.py --all               # calories + sample
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_RECIPES = ROOT / "data" / "raw" / "recipenlg"
RAW_CALORIES = ROOT / "data" / "raw" / "calories"
SAMPLES = ROOT / "data" / "samples"

# RecipeNLG original columns: title, ingredients, directions, link, source, NER.
# HuggingFace mirrors are used because recipenlg.cs.put.poznan.pl has been
# unreachable since 2025 and Kaggle needs an authenticated API token.
RECIPE_MIRRORS = [
    "https://huggingface.co/datasets/Mahimas/recipenlg/resolve/main/RecipeNLG.csv",
    "https://huggingface.co/datasets/SandhyaKilari/RecipeNLG_dataset/resolve/main/RecipeNLG_dataset.csv",
]
RECIPE_COLUMNS = ["title", "ingredients", "directions", "link", "source", "NER"]

# USDA FoodData Central, public domain (CC0).
# SR Legacy only: 7,793 foods, the last complete generic-ingredient release.
# Foundation Foods switched to a per-sample schema without food_nutrient.csv.
FDC_FILES = {
    "sr_legacy": "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip",
}

# The corpus is ordered by source, so a prefix sample would be single-source.
# With the full file on disk we take a uniform random sample; without it we fall
# back to equal-size byte windows spread across the whole file.
SAMPLE_ROWS = 10_000
SAMPLE_SEED = 42
SAMPLE_CHUNKS = 12
SAMPLE_CHUNK_BYTES = 800_000


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "cookfrompantry-dataset-prep/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        seen = 0
        t0 = time.time()
        with dest.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                seen += len(chunk)
                if total:
                    print(f"\r  {dest.name}: {seen / total:6.1%} ({seen >> 20} / {total >> 20} MB)", end="")
                else:
                    print(f"\r  {dest.name}: {seen >> 20} MB", end="")
        print(f"\r  {dest.name}: {seen >> 20} MB in {time.time() - t0:.0f}s")


def _range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "cookfrompantry-dataset-prep/1.0",
            "Range": f"bytes={start}-{end}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _corpus_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "cookfrompantry-dataset-prep/1.0", "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content_range = resp.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        return int(resp.headers["Content-Length"])


def _parse_recipe_row(row: list[str], keep: list[int]) -> list[str] | None:
    """Return the six RecipeNLG columns, or None for a truncated/corrupt row."""
    if len(row) <= max(keep):
        return None
    try:
        ast.literal_eval(row[keep[1]])  # ingredients
        ast.literal_eval(row[keep[-1]])  # NER
    except (ValueError, SyntaxError, TypeError):
        return None
    return [row[i] for i in keep]


def _normalize_recipes(chunks: list[bytes], dest: Path) -> int:
    """Drop the unnamed pandas index column, keep the six RecipeNLG columns."""
    header: list[str] | None = None
    keep: list[int] = []
    written = 0
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RECIPE_COLUMNS)
        for index, raw in enumerate(chunks):
            rows = list(csv.reader(io.StringIO(raw.decode("utf-8", errors="ignore"), newline="")))
            if header is None:
                header = rows.pop(0)
                keep = [header.index(c) for c in RECIPE_COLUMNS]
            else:
                rows.pop(0)  # starts inside a previous record
            if not raw.endswith(b"\n"):
                rows.pop()  # last record is cut off
            for row in rows:
                record = _parse_recipe_row(row, keep)
                if record:
                    writer.writerow(record)
                    written += 1
    return written


def _write_sample(records: list[list[str]], dest: Path) -> int:
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RECIPE_COLUMNS)
        writer.writerows(records)
    return len(records)


def _sample_local(path: Path, count: int, seed: int) -> list[list[str]]:
    """Reservoir sample `count` valid records in corpus order (one pass over the file)."""
    rng = random.Random(seed)
    picked: list[tuple[int, list[str]]] = []
    keep: list[int] = []
    seen = 0
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        keep = [header.index(c) for c in RECIPE_COLUMNS]
        for row in reader:
            if len(row) <= max(keep):  # truncated row
                continue
            if len(picked) < count:
                picked.append((seen, row))
            else:
                slot = rng.randrange(seen + 1)
                if slot < count:
                    picked[slot] = (seen, row)
            seen += 1
    picked.sort(key=lambda item: item[0])
    records = []
    for _, row in picked:
        record = _parse_recipe_row(row, keep)
        if record:
            records.append(record)
    print(f"[info] {seen} corpus rows scanned, {len(records)} sampled")
    return records


def fetch_recipe_sample(force: bool = False) -> Path:
    dest = SAMPLES / "recipes_sample.csv"
    if dest.exists() and not force:
        print(f"[skip] {dest.relative_to(ROOT)} exists")
        return dest
    SAMPLES.mkdir(parents=True, exist_ok=True)
    full = RAW_RECIPES / "RecipeNLG.csv"
    if full.exists():
        print(f"[recipes] sampling {SAMPLE_ROWS} rows from local {full.relative_to(ROOT)}")
        n = _write_sample(_sample_local(full, SAMPLE_ROWS, SAMPLE_SEED), dest)
        print(f"[ok] {dest.relative_to(ROOT)}: {n} recipes")
        return dest
    last_err: Exception | None = None
    for url in RECIPE_MIRRORS:
        try:
            size = _corpus_size(url)
            print(f"[recipes] sampling {SAMPLE_CHUNKS} windows of {SAMPLE_CHUNK_BYTES >> 10} KB from {size / 1e9:.2f} GB")
            chunks = []
            for i in range(SAMPLE_CHUNKS):
                start = i * size // SAMPLE_CHUNKS
                blob = _range(url, start, min(start + SAMPLE_CHUNK_BYTES - 1, size - 1))
                if len(blob) < SAMPLE_CHUNK_BYTES // 2:
                    raise RuntimeError(f"short range response at byte {start}")
                chunks.append(blob)
            n = _normalize_recipes(chunks, dest)  # network fallback
            print(f"[ok] {dest.relative_to(ROOT)}: {n} recipes")
            return dest
        except Exception as err:  # noqa: BLE001 - try the next mirror
            last_err = err
            print(f"[warn] mirror failed: {err}")
    raise SystemExit(f"all RecipeNLG mirrors failed: {last_err}")


def fetch_recipe_full(force: bool = False) -> Path:
    dest = RAW_RECIPES / "RecipeNLG.csv"
    if dest.exists() and not force:
        print(f"[skip] {dest.relative_to(ROOT)} exists ({dest.stat().st_size >> 20} MB)")
        return dest
    RAW_RECIPES.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.part")
    last_err: Exception | None = None
    for url in RECIPE_MIRRORS:
        try:
            print(f"[recipes] downloading 2.3 GB corpus from {url}")
            _download(url, tmp)
            tmp.replace(dest)
            with dest.open(encoding="utf-8", newline="") as fh:
                header = next(csv.reader(fh))
            missing = [c for c in RECIPE_COLUMNS if c not in header]
            if missing:
                raise SystemExit(f"{dest.name}: missing columns {missing}, got {header}")
            print(f"[ok] {dest.relative_to(ROOT)} ({dest.stat().st_size >> 20} MB)")
            return dest
        except SystemExit:
            raise
        except Exception as err:  # noqa: BLE001
            last_err = err
            print(f"[warn] mirror failed: {err}")
    tmp.unlink(missing_ok=True)
    raise SystemExit(f"all RecipeNLG mirrors failed: {last_err}")


def fetch_calories(force: bool = False) -> None:
    RAW_CALORIES.mkdir(parents=True, exist_ok=True)
    for name, url in FDC_FILES.items():
        target = RAW_CALORIES / name
        if target.is_dir() and any(target.glob("food.csv")) and not force:
            print(f"[skip] data/raw/calories/{name} exists")
            continue
        print(f"[calories] {name}")
        archive = RAW_CALORIES / f"{name}.zip"
        _download(url, archive)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                rel = Path(member).parts[1:]  # strip the archive's top folder
                if not rel or member.endswith("/"):
                    continue
                out = target.joinpath(*rel)
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, out.open("wb") as dst:
                    dst.write(src.read())
        archive.unlink()
        print(f"[ok] data/raw/calories/{name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipes", choices=["none", "sample", "full"], default="none")
    ap.add_argument("--calories", action="store_true", help="USDA FoodData Central CSVs")
    ap.add_argument("--all", action="store_true", help="calories + recipe sample")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.all:
        args.calories, args.recipes = True, "sample"
    if args.calories:
        fetch_calories(args.force)
    if args.recipes == "sample":
        fetch_recipe_sample(args.force)
    elif args.recipes == "full":
        fetch_recipe_full(args.force)
    if not args.calories and args.recipes == "none":
        ap.print_help(sys.stderr)


if __name__ == "__main__":
    main()
