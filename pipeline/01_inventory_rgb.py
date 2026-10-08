#!/usr/bin/env python3
"""
01_inventory_rgb.py - RGB-only inventory (growth-stage classification)

Walks the root dir, keeps only *_D.JPG files, parses field/session/stage from
the parent folder name, reads pixel dimensions, writes manifest_rgb.csv.
Paths are stored RELATIVE to --root. Safely rerunnable (existing rows kept).

Usage:
    python 01_inventory_rgb.py --root test_1000_images --manifest manifest_rgb.csv
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Missing Pillow:  pip install Pillow")

# Folder: F1_01_2025_07_15_1716_Nursery_Stage  (or ..._1716(1)_...)
FOLDER_RE = re.compile(
    r"^(?P<field>[A-Za-z0-9]+_\d+)"
    r"_(?P<y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})"
    r"_(?P<time>\d{4})(?:\(\d+\))?"
    r"_(?P<stage>.+)$"
)
# File: 20250715172855_0001_D.JPG
FILE_RE = re.compile(r"^(?P<ts>\d{14})_(?P<seq>\d{4})_D\.JPG$", re.IGNORECASE)

COLS = ["capture_id", "timestamp", "sequence", "field_plot", "session", "date",
        "growth_stage", "file_path", "width_px", "height_px", "file_size_bytes"]


def parse_folder(path: Path, root: Path):
    for parent in path.parents:
        m = FOLDER_RE.match(parent.name)
        if m:
            return {
                "field_plot": m["field"],
                "session": parent.name,
                "date": f"{m['y']}-{m['m']}-{m['d']}",
                "growth_stage": m["stage"],
            }
        if parent == root:
            break
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="test_1000_images")
    ap.add_argument("--manifest", default="manifest_rgb.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest = Path(args.manifest).resolve()
    if not root.exists():
        sys.exit(f"Root not found: {root}")

    rows, seen = [], set()
    if manifest.exists():
        with open(manifest, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                seen.add(r["file_path"])
        print(f"[INFO] Loaded {len(rows)} existing rows")

    new, bad_folder = 0, 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        m = FILE_RE.match(p.name)
        if not m:
            continue
        rel = p.relative_to(root).as_posix()
        if rel in seen:
            continue
        meta = parse_folder(p, root)
        if meta is None:
            bad_folder += 1
            meta = {"field_plot": "unknown", "session": "unknown",
                    "date": "", "growth_stage": "unknown"}
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as e:
            print(f"[WARN] unreadable {p.name}: {e}")
            w, h = 0, 0
        rows.append({
            "capture_id": f"{m['ts']}_{m['seq']}", "timestamp": m["ts"],
            "sequence": m["seq"], **meta, "file_path": rel,
            "width_px": w, "height_px": h, "file_size_bytes": p.stat().st_size,
        })
        seen.add(rel)
        new += 1

    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nRows total: {len(rows)}   new: {new}   unparsed folders: {bad_folder}")
    print("Dimensions :", dict(Counter((r["width_px"], r["height_px"]) for r in rows)))
    print("By field   :", dict(Counter(r["field_plot"] for r in rows)))
    print("By stage   :", dict(Counter(r["growth_stage"] for r in rows)))
    print(f"Manifest   : {manifest}")


if __name__ == "__main__":
    main()
