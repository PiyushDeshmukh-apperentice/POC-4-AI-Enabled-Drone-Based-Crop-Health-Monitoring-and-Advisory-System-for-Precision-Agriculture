#!/usr/bin/env python3
r"""
01_inventory.py  —  Phase 1: Inventory & Manifest Builder
=============================================================
Walks test_1000_images\, extracts metadata from filenames and folder names,
reads actual pixel dimensions from each file, and writes/extends manifest.csv.

Usage:
    python pipeline/01_inventory.py
    python pipeline/01_inventory.py --root test_1000_images --manifest manifest.csv

Safely rerunnable: existing rows (keyed by file_path) are preserved; only new
files are appended.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
missing = []
try:
    import rasterio
except ImportError:
    missing.append("rasterio")
try:
    from PIL import Image
except ImportError:
    missing.append("Pillow")

if missing:
    print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
    print("        Run:  pip install -r pipeline/requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants / patterns
# ---------------------------------------------------------------------------
# Folder name: F1_01_2025_07_15_1716_Nursery_Stage
FOLDER_RE = re.compile(
    r"^(?P<field>[A-Za-z0-9]+_\d+)"        # F1_01
    r"_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})"
    r"_(?P<time>\d{4})(?:\(\d+\))?"         # 1716 or 1716(1) — duplicate session
    r"_(?P<stage>.+)$"                       # Nursery_Stage
)


# File name: 20250715172855_0001_D.JPG  /  _MS_G.TIF  etc.
FILE_RE = re.compile(
    r"^(?P<timestamp>\d{14})_(?P<seq>\d{4})_(?P<variant>D|MS_G|MS_R|MS_RE|MS_NIR)"
    r"\.(?P<ext>JPG|TIF)$",
    re.IGNORECASE,
)

MANIFEST_SOURCE_COLS = [
    "capture_id", "timestamp", "sequence", "field_plot", "growth_stage",
    "variant", "file_path", "width_px", "height_px", "file_size_bytes",
]
TILE_EXTRA_COLS = [
    "tile_index", "tile_row", "tile_col",
    "tile_x_offset", "tile_y_offset",
    "tile_width", "tile_height",
    "source_capture_id", "tile_file_path",
]
MANIFEST_COLS = MANIFEST_SOURCE_COLS + TILE_EXTRA_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_folder(folder_name: str) -> dict | None:
    m = FOLDER_RE.match(folder_name)
    if not m:
        return None
    return {"field_plot": m.group("field"), "growth_stage": m.group("stage")}


def parse_filename(filename: str) -> dict | None:
    m = FILE_RE.match(filename)
    if not m:
        return None
    return {
        "timestamp":  m.group("timestamp"),
        "sequence":   m.group("seq"),
        "variant":    m.group("variant").upper(),
        "capture_id": f"{m.group('timestamp')}_{m.group('seq')}",
    }


def read_dimensions(filepath: Path, variant: str) -> tuple[int, int]:
    """Return (width, height) for a file. TIF via rasterio, JPG via Pillow."""
    try:
        if variant == "D":
            with Image.open(filepath) as img:
                return img.width, img.height
        else:
            with rasterio.open(filepath) as ds:
                return ds.width, ds.height
    except Exception as exc:
        print(f"  [WARN] Could not read dimensions for {filepath.name}: {exc}")
        return 0, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 1: Build manifest.csv")
    ap.add_argument("--root",     default="test_1000_images", help="Root download dir")
    ap.add_argument("--manifest", default="manifest.csv",     help="Output CSV path")
    args = ap.parse_args()

    root_dir     = Path(args.root).resolve()
    manifest_path = Path(args.manifest).resolve()

    if not root_dir.exists():
        print(f"[ERROR] Root directory not found: {root_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load existing manifest rows (for idempotency)
    # ------------------------------------------------------------------
    existing_paths: set[str] = set()
    existing_rows:  list[dict] = []

    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                existing_paths.add(row["file_path"])
        print(f"[INFO] Loaded {len(existing_rows)} existing rows from {manifest_path.name}")

    # ------------------------------------------------------------------
    # Walk the directory tree
    # ------------------------------------------------------------------
    new_rows:    list[dict] = []
    skipped:     int = 0
    unrecognised: list[str] = []

    for file_path in sorted(root_dir.rglob("*")):
        if not file_path.is_file():
            continue

        rel_str = str(file_path)

        # Already in manifest?
        if rel_str in existing_paths:
            skipped += 1
            continue

        # Parse filename
        file_meta = parse_filename(file_path.name)
        if file_meta is None:
            unrecognised.append(rel_str)
            continue

        # Parse parent folder(s) for stage — walk up until a folder matches
        folder_meta = None
        for parent in file_path.parents:
            folder_meta = parse_folder(parent.name)
            if folder_meta:
                break

        if folder_meta is None:
            print(f"  [WARN] Could not parse stage from path: {file_path}")
            folder_meta = {"field_plot": "unknown", "growth_stage": "unknown"}

        # Read pixel dimensions
        w, h = read_dimensions(file_path, file_meta["variant"])

        row = {
            "capture_id":      file_meta["capture_id"],
            "timestamp":       file_meta["timestamp"],
            "sequence":        file_meta["sequence"],
            "field_plot":      folder_meta["field_plot"],
            "growth_stage":    folder_meta["growth_stage"],
            "variant":         file_meta["variant"],
            "file_path":       rel_str,
            "width_px":        w,
            "height_px":       h,
            "file_size_bytes": file_path.stat().st_size,
        }
        new_rows.append(row)
        print(f"  [OK] {file_meta['variant']:8s}  {file_path.name}  {w}x{h}")

    # ------------------------------------------------------------------
    # Write merged manifest
    # ------------------------------------------------------------------
    all_rows = existing_rows + new_rows

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(all_rows)

    # ------------------------------------------------------------------
    # Print dimension summary per variant
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("MANIFEST SUMMARY")
    print("=" * 65)
    print(f"  Total rows written : {len(all_rows)}")
    print(f"  Newly added        : {len(new_rows)}")
    print(f"  Skipped (existing) : {skipped}")

    if unrecognised:
        print(f"\n  Unrecognised files ({len(unrecognised)}) — not added to manifest:")
        for u in unrecognised:
            print(f"    {u}")

    print("\n  Pixel dimensions by variant (from newly scanned files):")
    print(f"  {'Variant':<10} {'Width':>8} {'Height':>8}  {'Count':>6}")
    print(f"  {'-'*10} {'-'*8} {'-'*8}  {'-'*6}")

    dims_by_variant: dict[str, list[tuple[int,int]]] = defaultdict(list)
    for row in new_rows:
        dims_by_variant[row["variant"]].append((int(row["width_px"]), int(row["height_px"])))

    for variant in sorted(dims_by_variant):
        dims  = dims_by_variant[variant]
        unique = set(dims)
        if len(unique) == 1:
            w, h = unique.pop()
            print(f"  {variant:<10} {w:>8} {h:>8}  {len(dims):>6}")
        else:
            print(f"  {variant:<10} MULTIPLE DIMENSIONS FOUND:")
            for (w, h), count in sorted(
                {d: dims.count(d) for d in unique}.items()
            ):
                print(f"    {w}x{h} — {count} files")

    print(f"\n  Manifest written to: {manifest_path}\n")


if __name__ == "__main__":
    main()
