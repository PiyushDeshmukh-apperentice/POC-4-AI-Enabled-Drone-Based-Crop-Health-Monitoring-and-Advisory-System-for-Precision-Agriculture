#!/usr/bin/env python3
"""
05_validate.py  —  Phase 5: Validation
=========================================
Verifies the pipeline outputs by:
  1. Checking tile counts match expected grid for each stream.
  2. Spot-checking N random tiles per stream: opens them, checks dimensions,
     checks they are not blank (variance > 0).
  3. Printing a summary table: rows = growth stages, columns = streams.
  4. Writing validation_report.txt.

Usage:
    python pipeline/05_validate.py
    python pipeline/05_validate.py --tiles tiles --manifest manifest.csv --sample 10
    python pipeline/05_validate.py --tile-size 1024 --overlap 0.25
"""

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
missing = []
try:
    import numpy as np
except ImportError:
    missing.append("numpy")
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
# Helpers
# ---------------------------------------------------------------------------

def expected_tile_count(width: int, height: int, tile_size: int, overlap: float) -> int:
    """Compute expected number of tiles for given image dimensions."""
    step  = max(1, int(tile_size * (1.0 - overlap)))
    n_col = math.ceil((width  - 1) / step) + 1 if width  > tile_size else 1
    n_row = math.ceil((height - 1) / step) + 1 if height > tile_size else 1
    # More precisely: count starts from 0 and steps until edge
    x, col_count = 0, 0
    while x < width:
        col_count += 1
        if x + tile_size >= width:
            break
        x += step
    y, row_count = 0, 0
    while y < height:
        row_count += 1
        if y + tile_size >= height:
            break
        y += step
    return col_count * row_count


def open_tile(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Open a tile file and return (array, (width, height))."""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png"):
        with Image.open(path) as img:
            arr = np.array(img)
            return arr, (img.width, img.height)
    else:
        with rasterio.open(path) as ds:
            arr = ds.read()
            return arr, (ds.width, ds.height)


def is_blank(arr: np.ndarray) -> bool:
    """Return True if all pixels are the same value (blank/zeroed tile)."""
    flat = arr.flatten().astype(float)
    return float(np.nanvar(flat)) == 0.0


# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 5: Validate pipeline outputs")
    ap.add_argument("--tiles",     default="tiles",        help="Tiles root directory")
    ap.add_argument("--organized", default="organized",    help="Organised files root")
    ap.add_argument("--manifest",  default="manifest.csv", help="Manifest CSV")
    ap.add_argument("--report",    default="validation_report.txt", help="Output report path")
    ap.add_argument("--tile-size", type=int, default=1024, help="Tile size used in Phase 4")
    ap.add_argument("--overlap",   type=float, default=0.25, help="Overlap fraction used in Phase 4")
    ap.add_argument("--sample",    type=int, default=10,   help="Tiles to spot-check per stream")
    ap.add_argument("--seed",      type=int, default=42,   help="Random seed for spot-check")
    args = ap.parse_args()

    random.seed(args.seed)

    tiles_dir     = Path(args.tiles).resolve()
    manifest_path = Path(args.manifest).resolve()
    report_path   = Path(args.report).resolve()

    lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    log(f"IMPaCT-UAV-MsRGB Pipeline Validation Report")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Tile size: {args.tile_size}px  Overlap: {args.overlap*100:.0f}%")
    log("=" * 70)

    # ------------------------------------------------------------------
    # Load manifest and split into source rows vs tile rows
    # ------------------------------------------------------------------
    all_rows   = load_manifest(manifest_path)
    if not all_rows:
        log("[ERROR] Manifest not found or empty. Run phases 1–4 first.")
        sys.exit(1)

    source_rows = [r for r in all_rows if not r.get("tile_file_path", "").strip()]
    tile_rows   = [r for r in all_rows if r.get("tile_file_path", "").strip()]

    log(f"\nManifest: {len(source_rows)} source files, {len(tile_rows)} tile entries")

    # Build a lookup: source file path → dimensions
    source_dims: dict[str, tuple[int, int]] = {}
    for r in source_rows:
        try:
            source_dims[r["file_path"]] = (int(r["width_px"]), int(r["height_px"]))
        except (ValueError, KeyError):
            pass

    # ------------------------------------------------------------------
    # Section 1: Tile count verification
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log("SECTION 1: TILE COUNT VERIFICATION")
    log("-" * 70)

    # Group tile rows by (stream, sub_band, stage, source capture)
    # stream = derived from tile_file_path structure
    count_errors: list[str] = []

    def derive_stream(tile_path_str: str) -> str:
        """Guess stream from path."""
        p = Path(tile_path_str)
        parts = p.parts
        try:
            tiles_idx = next(i for i, pt in enumerate(parts) if pt == "tiles")
            return parts[tiles_idx + 1] if tiles_idx + 1 < len(parts) else "unknown"
        except StopIteration:
            return "unknown"

    # Group tiles by (stream, source_capture_id)
    tiles_by_source: dict[tuple, list[dict]] = defaultdict(list)
    for row in tile_rows:
        stream     = derive_stream(row["tile_file_path"])
        capture_id = row.get("source_capture_id", row.get("capture_id", ""))
        tiles_by_source[(stream, capture_id)].append(row)

    stream_tile_counts: dict[str, list[int]] = defaultdict(list)

    for (stream, capture_id), t_rows in tiles_by_source.items():
        actual_count = len(t_rows)
        stream_tile_counts[stream].append(actual_count)

        # Find source dims for this capture
        src_path = t_rows[0].get("file_path", "")
        dims     = source_dims.get(src_path)
        if dims and dims[0] > 0 and dims[1] > 0:
            expected = expected_tile_count(dims[0], dims[1], args.tile_size, args.overlap)
            ok       = actual_count == expected
            status   = "OK" if ok else "MISMATCH"
            if not ok:
                msg = (f"  [{status}] {stream}/{capture_id}: "
                       f"actual={actual_count}  expected={expected}  "
                       f"({dims[0]}x{dims[1]})")
                count_errors.append(msg)
                log(msg)

    if not count_errors:
        log("  All tile counts match expected grid. [OK]")

    log(f"\n  Tile count per image by stream:")
    log(f"  {'Stream':<16} {'Images':>7} {'Avg tiles':>10} {'Min':>6} {'Max':>6}")
    log(f"  {'-'*16} {'-'*7} {'-'*10} {'-'*6} {'-'*6}")
    for stream, counts in sorted(stream_tile_counts.items()):
        avg = sum(counts) / len(counts) if counts else 0
        mn  = min(counts) if counts else 0
        mx  = max(counts) if counts else 0
        log(f"  {stream:<16} {len(counts):>7} {avg:>10.1f} {mn:>6} {mx:>6}")

    # ------------------------------------------------------------------
    # Section 2: Spot-check
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log(f"SECTION 2: SPOT-CHECK ({args.sample} random tiles per stream)")
    log("-" * 70)

    # Collect all tile files actually on disk per stream
    stream_files: dict[str, list[Path]] = defaultdict(list)
    if tiles_dir.exists():
        for f in tiles_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
                stream_name = f.parts[len(tiles_dir.parts)] if len(f.parts) > len(tiles_dir.parts) else "unknown"
                stream_files[stream_name].append(f)

    spot_errors:  list[str] = []
    spot_summary: dict[str, dict] = {}

    for stream_name, all_files in sorted(stream_files.items()):
        sample_size = min(args.sample, len(all_files))
        sample      = random.sample(all_files, sample_size)
        ok_count    = 0
        blank_count = 0
        error_count = 0

        for tile_path in sample:
            try:
                arr, (w, h) = open_tile(tile_path)
                if is_blank(arr):
                    msg = f"  [BLANK] {tile_path}"
                    spot_errors.append(msg)
                    log(msg)
                    blank_count += 1
                else:
                    ok_count += 1
            except Exception as exc:
                msg = f"  [ERROR] {tile_path}: {exc}"
                spot_errors.append(msg)
                log(msg)
                error_count += 1

        spot_summary[stream_name] = {
            "total":  len(all_files),
            "sample": sample_size,
            "ok":     ok_count,
            "blank":  blank_count,
            "error":  error_count,
        }

    log(f"\n  {'Stream':<16} {'Total':>7} {'Sampled':>8} {'OK':>5} {'Blank':>6} {'Error':>6}")
    log(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*5} {'-'*6} {'-'*6}")
    for stream_name, s in sorted(spot_summary.items()):
        log(f"  {stream_name:<16} {s['total']:>7} {s['sample']:>8} {s['ok']:>5} {s['blank']:>6} {s['error']:>6}")

    # ------------------------------------------------------------------
    # Section 3: Stage × Stream tile count table
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log("SECTION 3: FINAL COUNTS BY GROWTH STAGE AND STREAM")
    log("-" * 70)

    # Count tiles by (growth_stage, stream)
    stage_stream_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    all_stages:  set[str] = set()
    all_streams: set[str] = set()

    for row in tile_rows:
        stage  = row.get("growth_stage", "unknown")
        stream = derive_stream(row["tile_file_path"])
        stage_stream_counts[stage][stream] += 1
        all_stages.add(stage)
        all_streams.add(stream)

    sorted_stages  = sorted(all_stages)
    sorted_streams = sorted(all_streams)

    header = f"  {'Stage':<30}" + "".join(f" {s:>12}" for s in sorted_streams) + f" {'TOTAL':>8}"
    log(header)
    log("  " + "-" * (len(header) - 2))

    grand_totals = {s: 0 for s in sorted_streams}
    for stage in sorted_stages:
        row_str   = f"  {stage:<30}"
        row_total = 0
        for stream in sorted_streams:
            cnt = stage_stream_counts[stage].get(stream, 0)
            row_str += f" {cnt:>12}"
            row_total         += cnt
            grand_totals[stream] += cnt
        row_str += f" {row_total:>8}"
        log(row_str)

    total_str = f"  {'GRAND TOTAL':<30}" + "".join(f" {grand_totals[s]:>12}" for s in sorted_streams)
    total_str += f" {sum(grand_totals.values()):>8}"
    log("  " + "-" * (len(header) - 2))
    log(total_str)

    # ------------------------------------------------------------------
    # Overall result
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    total_issues = len(count_errors) + len(spot_errors)
    if total_issues == 0:
        log("RESULT: ALL CHECKS PASSED [OK]")
    else:
        log(f"RESULT: {total_issues} issue(s) found — review above for details.")
    log("=" * 70)

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[INFO] Report written to: {report_path}")


if __name__ == "__main__":
    main()
