#!/usr/bin/env python3
"""
04_tile.py  —  Phase 4: Tiling
================================
Tiles images from organized/ into CNN-ready tiles written to tiles/.
Each resolution class (RGB, MS bands, Indices) is tiled independently with
its own natural grid — the RGB tile count is NOT forced onto MS data.

Edge tiles are padded (reflect or constant) rather than dropped.

Usage:
    python pipeline/04_tile.py
    python pipeline/04_tile.py --tile-size 1024 --overlap 0.25 --pad-mode reflect
    python pipeline/04_tile.py --streams rgb ms_bands indices

Extends manifest.csv with one row per tile linking back to source capture.
Safely rerunnable — existing tile files are skipped.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from collections import defaultdict

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
    from rasterio.transform import Affine
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
# Constants
# ---------------------------------------------------------------------------
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
ALL_COLS = MANIFEST_SOURCE_COLS + TILE_EXTRA_COLS

STREAMS = {
    "rgb":       {"subdir": "rgb",                "ext": ".jpg",  "is_raster": False},
    "ms_bands":  {"subdir": "ms_bands",           "ext": ".tif",  "is_raster": True},
    "indices":   {"subdir": "indices",            "ext": ".tif",  "is_raster": True},
}

# Regex to extract variant from MS band folder path component
MS_BAND_RE = re.compile(r"(MS_G|MS_R|MS_RE|MS_NIR|NDVI|NDRE|GNDVI)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Tiling core
# ---------------------------------------------------------------------------

def compute_tile_grid(
    width: int, height: int, tile_size: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """
    Returns list of (x_start, y_start, x_end, y_end) for each tile.
    Uses overlap fraction to compute step size.
    Edge tiles may be smaller than tile_size before padding.
    """
    step = max(1, int(tile_size * (1.0 - overlap)))
    tiles = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            tiles.append((x, y, x_end, y_end))
            if x_end == width:
                break
            x += step
        if y_end == height:
            break
        y += step
    return tiles


def pad_array(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
    pad_mode: str,
    is_channels_first: bool = False,
) -> np.ndarray:
    """Pad a 2D or 3D array to (target_h, target_w) using pad_mode."""
    if arr.ndim == 2:
        h, w = arr.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_mode == "reflect" and (h < 2 or w < 2):
            pad_mode = "edge"
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode=pad_mode)
    elif is_channels_first:  # (bands, h, w)
        _, h, w = arr.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_mode == "reflect" and (h < 2 or w < 2):
            pad_mode = "edge"
        return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode=pad_mode)
    else:  # (h, w, channels)
        h, w, _ = arr.shape
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        if pad_mode == "reflect" and (h < 2 or w < 2):
            pad_mode = "edge"
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)


# ---------------------------------------------------------------------------
# RGB tiling (JPEG via Pillow)
# ---------------------------------------------------------------------------

def tile_rgb_file(
    src_path: Path,
    out_dir: Path,
    capture_id: str,
    tile_size: int,
    overlap: float,
    pad_mode: str,
) -> list[dict]:
    """Tile a JPEG RGB image. Returns list of tile metadata dicts."""
    with Image.open(src_path) as img:
        arr = np.array(img)  # HxWx3

    height, width = arr.shape[:2]
    grid = compute_tile_grid(width, height, tile_size, overlap)
    results = []

    for idx, (x0, y0, x1, y1) in enumerate(grid):
        row_num = idx // math.ceil(width / max(1, int(tile_size * (1 - overlap))))
        col_num = idx % math.ceil(width / max(1, int(tile_size * (1 - overlap))))

        out_name = f"{capture_id}_tile{idx:04d}.jpg"
        out_path = out_dir / out_name

        if out_path.exists():
            results.append(_tile_meta(capture_id, idx, row_num, col_num,
                                      x0, y0, tile_size, tile_size, out_path))
            continue

        crop = arr[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape[:2]

        if crop_h < tile_size or crop_w < tile_size:
            # Pad
            padded = pad_array(crop, tile_size, tile_size, pad_mode, is_channels_first=False)
        else:
            padded = crop

        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(padded).save(out_path, quality=95)
        results.append(_tile_meta(capture_id, idx, row_num, col_num,
                                  x0, y0, tile_size, tile_size, out_path))

    return results


# ---------------------------------------------------------------------------
# Raster tiling (GeoTIFF via rasterio)
# ---------------------------------------------------------------------------

def tile_raster_file(
    src_path: Path,
    out_dir: Path,
    capture_id: str,
    tile_size: int,
    overlap: float,
    pad_mode: str,
) -> list[dict]:
    """Tile a single or multi-band GeoTIFF. Returns list of tile metadata dicts."""
    with rasterio.open(src_path) as ds:
        width    = ds.width
        height   = ds.height
        profile  = ds.profile.copy()
        data     = ds.read()   # (bands, H, W)
        transform = ds.transform

    grid = compute_tile_grid(width, height, tile_size, overlap)

    cols_per_row = math.ceil(
        (width - 1) / max(1, int(tile_size * (1 - overlap))) + 1
    )
    results = []

    profile.update(
        width=tile_size,
        height=tile_size,
        driver="GTiff",
        compress="lzw",
    )

    for idx, (x0, y0, x1, y1) in enumerate(grid):
        row_num = idx // cols_per_row
        col_num = idx % cols_per_row

        out_name = f"{capture_id}_tile{idx:04d}.tif"
        out_path = out_dir / out_name

        if out_path.exists():
            results.append(_tile_meta(capture_id, idx, row_num, col_num,
                                      x0, y0, tile_size, tile_size, out_path))
            continue

        crop = data[:, y0:y1, x0:x1]
        crop_h = y1 - y0
        crop_w = x1 - x0

        if crop_h < tile_size or crop_w < tile_size:
            crop = pad_array(crop, tile_size, tile_size, pad_mode, is_channels_first=True)

        # Update geotransform for tile origin
        tile_transform = Affine(
            transform.a, transform.b, transform.c + x0 * transform.a,
            transform.d, transform.e, transform.f + y0 * transform.e,
        )
        tile_profile = profile.copy()
        tile_profile["transform"] = tile_transform

        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **tile_profile) as dst:
            dst.write(crop)

        results.append(_tile_meta(capture_id, idx, row_num, col_num,
                                  x0, y0, tile_size, tile_size, out_path))

    return results


def _tile_meta(capture_id, idx, row_num, col_num, x0, y0, tw, th, out_path) -> dict:
    return {
        "tile_index":       idx,
        "tile_row":         row_num,
        "tile_col":         col_num,
        "tile_x_offset":    x0,
        "tile_y_offset":    y0,
        "tile_width":       tw,
        "tile_height":      th,
        "source_capture_id": capture_id,
        "tile_file_path":   str(out_path),
    }


# ---------------------------------------------------------------------------
# Stream discovery
# ---------------------------------------------------------------------------

def discover_stream_files(
    organized_dir: Path, stream: str
) -> list[tuple[Path, str, str, str]]:
    """
    Returns list of (file_path, capture_id, stage, sub_band).
    sub_band is the intermediate folder for ms_bands and indices (e.g. 'MS_G', 'NDVI').
    """
    results = []
    stream_dir = organized_dir / STREAMS[stream]["subdir"]
    if not stream_dir.exists():
        return results

    if stream == "rgb":
        for stage_dir in sorted(stream_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            for f in sorted(stage_dir.glob("*.JPG")):
                stem = f.stem  # 20250715172855_0001_D
                parts = stem.rsplit("_", maxsplit=1)
                capture_id = "_".join(parts[0].rsplit("_", maxsplit=1)[:2]) if "_" in parts[0] else stem
                # Actually: stem = ts_seq_D, so capture_id = ts_seq
                m = re.match(r"^(\d{14}_\d{4})_D$", stem, re.IGNORECASE)
                if m:
                    capture_id = m.group(1)
                results.append((f, capture_id, stage_dir.name, "rgb"))

    elif stream in ("ms_bands", "indices"):
        for band_dir in sorted(stream_dir.iterdir()):
            if not band_dir.is_dir():
                continue
            sub_band = band_dir.name
            for stage_dir in sorted(band_dir.iterdir()):
                if not stage_dir.is_dir():
                    continue
                for f in sorted(stage_dir.glob("*.TIF")):
                    stem = f.stem
                    # stem like: 20250715172855_0001_MS_G  or  _NDVI
                    m = re.match(r"^(\d{14}_\d{4})_", stem)
                    capture_id = m.group(1) if m else stem
                    results.append((f, capture_id, stage_dir.name, sub_band))

    return results


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------

def load_existing_tile_paths(manifest_path: Path) -> set[str]:
    """Return set of tile_file_path values already in manifest."""
    paths = set()
    if not manifest_path.exists():
        return paths
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("tile_file_path", ""):
                paths.add(row["tile_file_path"])
    return paths


def load_source_rows(manifest_path: Path) -> list[dict]:
    """Return all non-tile source rows from manifest."""
    rows = []
    if not manifest_path.exists():
        return rows
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 4: Tile images for CNN training")
    ap.add_argument("--organized",  default="organized",     help="Organised input dir")
    ap.add_argument("--tiles",      default="tiles",         help="Output tiles dir")
    ap.add_argument("--manifest",   default="manifest.csv",  help="Manifest CSV to extend")
    ap.add_argument("--tile-size",  type=int, default=1024,  help="Tile size in pixels")
    ap.add_argument("--overlap",    type=float, default=0.25, help="Overlap fraction (0–1)")
    ap.add_argument("--pad-mode",   default="reflect",
                    choices=["reflect", "edge", "constant"],
                    help="Padding mode for edge tiles")
    ap.add_argument("--streams",    nargs="+",
                    choices=["rgb", "ms_bands", "indices"],
                    default=["rgb", "ms_bands", "indices"],
                    help="Which streams to tile")
    args = ap.parse_args()

    organized_dir = Path(args.organized).resolve()
    tiles_dir     = Path(args.tiles).resolve()
    manifest_path = Path(args.manifest).resolve()

    tile_size = args.tile_size
    overlap   = args.overlap
    pad_mode  = args.pad_mode

    print(f"[INFO] Tile size : {tile_size}px")
    print(f"[INFO] Overlap   : {overlap * 100:.0f}%  (step = {int(tile_size * (1-overlap))}px)")
    print(f"[INFO] Pad mode  : {pad_mode}")
    print(f"[INFO] Streams   : {args.streams}")

    # ------------------------------------------------------------------
    # Load existing state
    # ------------------------------------------------------------------
    existing_tile_paths = load_existing_tile_paths(manifest_path)
    source_rows         = load_source_rows(manifest_path)

    # Build lookup: file_path → source row dict
    source_by_path: dict[str, dict] = {r["file_path"]: r for r in source_rows
                                        if r.get("tile_file_path", "") == ""}

    new_tile_rows: list[dict] = []
    stream_counts: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Process each stream
    # ------------------------------------------------------------------
    for stream in args.streams:
        files = discover_stream_files(organized_dir, stream)
        print(f"\n[INFO] Stream '{stream}': {len(files)} source files found.")

        is_raster = STREAMS[stream]["is_raster"]
        tile_fn   = tile_raster_file if is_raster else tile_rgb_file

        per_image_counts: list[int] = []
        stream_counts[stream] = {"files": 0, "tiles_new": 0, "tiles_skip": 0}

        for (src_file, capture_id, stage, sub_band) in files:
            # Output directory: tiles/<stream>/<sub_band>/<stage>/  (or tiles/rgb/<stage>/)
            if stream == "rgb":
                out_dir = tiles_dir / "rgb" / stage
            else:
                out_dir = tiles_dir / stream / sub_band / stage

            tile_metas = tile_fn(
                src_path=src_file,
                out_dir=out_dir,
                capture_id=capture_id,
                tile_size=tile_size,
                overlap=overlap,
                pad_mode=pad_mode,
            )

            per_image_counts.append(len(tile_metas))
            stream_counts[stream]["files"] += 1

            # Build source base row for manifest extension
            src_base = source_by_path.get(str(src_file), {})

            for meta in tile_metas:
                tile_path_str = meta["tile_file_path"]
                if tile_path_str in existing_tile_paths:
                    stream_counts[stream]["tiles_skip"] += 1
                    continue

                full_row = {
                    "capture_id":      src_base.get("capture_id", capture_id),
                    "timestamp":       src_base.get("timestamp", ""),
                    "sequence":        src_base.get("sequence", ""),
                    "field_plot":      src_base.get("field_plot", ""),
                    "growth_stage":    src_base.get("growth_stage", stage),
                    "variant":         src_base.get("variant", sub_band),
                    "file_path":       src_base.get("file_path", str(src_file)),
                    "width_px":        src_base.get("width_px", tile_size),
                    "height_px":       src_base.get("height_px", tile_size),
                    "file_size_bytes": src_base.get("file_size_bytes", ""),
                    "tile_index":      meta["tile_index"],
                    "tile_row":        meta["tile_row"],
                    "tile_col":        meta["tile_col"],
                    "tile_x_offset":   meta["tile_x_offset"],
                    "tile_y_offset":   meta["tile_y_offset"],
                    "tile_width":      meta["tile_width"],
                    "tile_height":     meta["tile_height"],
                    "source_capture_id": meta["source_capture_id"],
                    "tile_file_path":  tile_path_str,
                }
                new_tile_rows.append(full_row)
                existing_tile_paths.add(tile_path_str)
                stream_counts[stream]["tiles_new"] += 1

        if per_image_counts:
            avg = sum(per_image_counts) / len(per_image_counts)
            mn  = min(per_image_counts)
            mx  = max(per_image_counts)
            print(f"  Tiles/image: avg={avg:.1f}  min={mn}  max={mx}")

    # ------------------------------------------------------------------
    # Append tile rows to manifest
    # ------------------------------------------------------------------
    if new_tile_rows:
        all_source = load_source_rows(manifest_path)
        all_rows = all_source + new_tile_rows
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[INFO] Manifest updated: {len(new_tile_rows)} tile rows added.")
    else:
        print("\n[INFO] No new tile rows to add to manifest.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TILING SUMMARY")
    print("=" * 65)
    print(f"  {'Stream':<14} {'Files':>6} {'New Tiles':>10} {'Skipped':>8}")
    print(f"  {'-'*14} {'-'*6} {'-'*10} {'-'*8}")
    for stream, c in stream_counts.items():
        print(f"  {stream:<14} {c['files']:>6} {c['tiles_new']:>10} {c['tiles_skip']:>8}")
    print(f"\n  Tile output: {tiles_dir}\n")


if __name__ == "__main__":
    main()
