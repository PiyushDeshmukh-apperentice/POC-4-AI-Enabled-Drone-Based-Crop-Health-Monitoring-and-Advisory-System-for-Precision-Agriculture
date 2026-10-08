#!/usr/bin/env python3
r"""
02_segregate.py  —  Phase 2: Segregate by Purpose
====================================================
Organises files from test_1000_images\ into:
  organized/rgb/<stage>/                    ← _D.JPG files
  organized/ms_bands/<band>/<stage>/        ← individual _MS_*.TIF files
  organized/ms_stack/<stage>/               ← 4-band stacked GeoTIFF (gated)

The ms_stack step requires the explicit --build-stack flag.  Before using it,
confirm that the dataset's Processed_Images.zip does NOT already contain
aligned orthomosaics (which would make stacking raw captures redundant).

Usage:
    python pipeline/02_segregate.py
    python pipeline/02_segregate.py --manifest manifest.csv --organized organized
    python pipeline/02_segregate.py --build-stack          # also build ms_stack

Safely rerunnable: existing output files are skipped.
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
missing = []
try:
    import rasterio
    import numpy as np
    from rasterio.transform import from_bounds
except ImportError:
    missing.append("rasterio / numpy")

if missing and "--build-stack" in sys.argv:
    print(f"[ERROR] Missing dependencies for stack building: {', '.join(missing)}")
    print("        Run:  pip install -r pipeline/requirements.txt")
    sys.exit(1)

# Variant → subdirectory name under ms_bands/
BAND_DIRS = {
    "MS_G":   "MS_G",
    "MS_R":   "MS_R",
    "MS_RE":  "MS_RE",
    "MS_NIR": "MS_NIR",
}

# Band order for the 4-band stack
STACK_ORDER = ["MS_G", "MS_R", "MS_RE", "MS_NIR"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_link_or_copy(src: Path, dst: Path) -> str:
    """
    Try to create a hardlink; fall back to copy.
    Returns 'link', 'copy', or 'skip'.
    """
    if dst.exists():
        return "skip"

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(src, dst)
        return "link"
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return "copy"


def build_stack(band_paths: dict[str, Path], out_path: Path) -> bool:
    """
    Combine 4 MS bands into a single 4-band GeoTIFF (G, R, RE, NIR).
    Preserves CRS and transform from the first available band.
    Returns True on success.
    """
    import rasterio
    import numpy as np
    from rasterio.enums import Resampling

    if out_path.exists():
        return False  # already done

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Open all bands
        datasets = {}
        for band_name in STACK_ORDER:
            bp = band_paths.get(band_name)
            if bp is None or not bp.exists():
                print(f"  [WARN] Stack skipped {out_path.name}: missing band {band_name}")
                return False
            datasets[band_name] = rasterio.open(bp)

        # Use first band's profile as template
        ref_ds = datasets[STACK_ORDER[0]]
        profile = ref_ds.profile.copy()
        profile.update(
            count=4,
            dtype="float32",
            driver="GTiff",
            compress="lzw",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for idx, band_name in enumerate(STACK_ORDER, start=1):
                ds = datasets[band_name]
                data = ds.read(1).astype("float32")
                dst.write(data, idx)
                dst.update_tags(idx, band=band_name)

        for ds in datasets.values():
            ds.close()

        return True

    except Exception as exc:
        print(f"  [ERROR] Stack failed for {out_path.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 2: Segregate files by purpose")
    ap.add_argument("--manifest",    default="manifest.csv", help="Input manifest CSV")
    ap.add_argument("--root",        default="test_1000_images", help="Root of raw downloads")
    ap.add_argument("--organized",   default="organized",    help="Output organised directory")
    ap.add_argument("--build-stack", action="store_true",
                    help="Also build 4-band ms_stack GeoTIFFs (see note below)")
    args = ap.parse_args()

    manifest_path  = Path(args.manifest).resolve()
    root_dir       = Path(args.root).resolve()
    organized_dir  = Path(args.organized).resolve()

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("        Run 01_inventory.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Stack gate warning
    # ------------------------------------------------------------------
    if args.build_stack:
        print("=" * 65)
        print("NOTE: --build-stack is enabled.")
        print("      Before using stacks for training, confirm whether the")
        print("      Processed_Images.zip in the dataset root already contains")
        print("      Pix4D-aligned orthomosaics. If yes, those are better inputs")
        print("      (pre-aligned) and duplicating this work is unnecessary.")
        print("=" * 65)
    else:
        print("[INFO] ms_stack step SKIPPED (run with --build-stack to enable).")
        print("       First confirm: does Processed_Images.zip contain pre-aligned")
        print("       Pix4D orthomosaics? If yes, use those instead of raw stacks.")

    # ------------------------------------------------------------------
    # Load manifest
    # ------------------------------------------------------------------
    rows: list[dict] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Only process source-file rows (not tile rows which have tile_index col)
        for row in reader:
            if row.get("tile_index", "") == "":
                rows.append(row)

    print(f"[INFO] Loaded {len(rows)} source file rows from manifest.")

    # ------------------------------------------------------------------
    # Group by capture_id for stacking
    # ------------------------------------------------------------------
    captures: dict[str, dict] = defaultdict(lambda: {
        "band_paths": {}, "stage": "", "field": ""
    })

    counters = {"rgb_link": 0, "rgb_copy": 0, "rgb_skip": 0,
                "ms_link": 0,  "ms_copy": 0,  "ms_skip": 0,
                "stack_built": 0, "stack_skip": 0, "stack_fail": 0}

    for row in rows:
        src_path   = Path(row["file_path"])
        variant    = row["variant"]
        stage      = row["growth_stage"]
        capture_id = row["capture_id"]

        # -- RGB --
        if variant == "D":
            dst = organized_dir / "rgb" / stage / src_path.name
            action = safe_link_or_copy(src_path, dst)
            counters[f"rgb_{action}"] += 1

        # -- MS bands --
        elif variant in BAND_DIRS:
            band_subdir = BAND_DIRS[variant]
            dst = organized_dir / "ms_bands" / band_subdir / stage / src_path.name
            action = safe_link_or_copy(src_path, dst)
            counters[f"ms_{action}"] += 1

            # Record for stacking
            captures[capture_id]["band_paths"][variant] = src_path
            captures[capture_id]["stage"]  = stage
            captures[capture_id]["field"]  = row["field_plot"]

    # -- ms_stack --
    if args.build_stack:
        print(f"[INFO] Building ms_stack for {len(captures)} captures...")
        for capture_id, cap_data in captures.items():
            stage      = cap_data["stage"]
            band_paths = cap_data["band_paths"]
            out_name   = f"{capture_id}_stack.TIF"
            out_path   = organized_dir / "ms_stack" / stage / out_name

            if out_path.exists():
                counters["stack_skip"] += 1
                continue

            ok = build_stack(band_paths, out_path)
            if ok:
                counters["stack_built"] += 1
                print(f"  [STACK] {out_name}")
            elif not out_path.exists():
                counters["stack_fail"] += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SEGREGATION SUMMARY")
    print("=" * 65)
    print(f"  RGB  — linked: {counters['rgb_link']:4d}  copied: {counters['rgb_copy']:4d}  skipped: {counters['rgb_skip']:4d}")
    print(f"  MS   — linked: {counters['ms_link']:4d}  copied: {counters['ms_copy']:4d}  skipped: {counters['ms_skip']:4d}")
    if args.build_stack:
        print(f"  Stack— built:  {counters['stack_built']:4d}  skipped: {counters['stack_skip']:4d}  failed: {counters['stack_fail']:4d}")
    print(f"\n  Output root: {organized_dir}\n")


if __name__ == "__main__":
    main()
