#!/usr/bin/env python3
"""
03_indices.py  —  Phase 3: Vegetation Index Computation
=========================================================
Computes NDVI, NDRE, GNDVI from aligned per-capture MS bands in organized/ms_bands/.
Operates on full-resolution images first, saves float32 GeoTIFF, THEN tiles
are generated separately in Phase 4.

Formulas:
  NDVI  = (NIR - R)  / (NIR + R)
  NDRE  = (NIR - RE) / (NIR + RE)
  GNDVI = (NIR - G)  / (NIR + G)

Output:
  organized/indices/NDVI/<stage>/<capture_id>_NDVI.TIF
  organized/indices/NDRE/<stage>/<capture_id>_NDRE.TIF
  organized/indices/GNDVI/<stage>/<capture_id>_GNDVI.TIF

Usage:
    python pipeline/03_indices.py
    python pipeline/03_indices.py --ms-bands organized/ms_bands --indices organized/indices

Safely rerunnable: existing index files are skipped unless --force is passed.
"""

import argparse
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

if missing:
    print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
    print("        Run:  pip install -r pipeline/requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Index definitions: name → (numerator_band, denominator_bands)
#   NDVI  = (NIR - R)  / (NIR + R)
#   NDRE  = (NIR - RE) / (NIR + RE)
#   GNDVI = (NIR - G)  / (NIR + G)
# ---------------------------------------------------------------------------
INDICES = {
    "NDVI":  ("MS_NIR", "MS_R"),
    "NDRE":  ("MS_NIR", "MS_RE"),
    "GNDVI": ("MS_NIR", "MS_G"),
}

# Band folder names inside organized/ms_bands/
BAND_FOLDERS = ["MS_G", "MS_R", "MS_RE", "MS_NIR"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_band(band_path: Path) -> tuple:
    """
    Load a single-band GeoTIFF as float32 ndarray.
    Returns (data_2d, profile_dict).
    """
    with rasterio.open(band_path) as ds:
        data    = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
    return data, profile


def safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """Normalised ratio with NaN where denominator is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom != 0, num / denom, np.nan)
    return result.astype(np.float32)


def write_index(data: np.ndarray, profile: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        driver="GTiff",
        compress="lzw",
        nodata=np.nan,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)


# ---------------------------------------------------------------------------
# Discover captures
# ---------------------------------------------------------------------------

def discover_captures(ms_bands_dir: Path) -> dict[str, dict]:
    """
    Walk ms_bands/<band>/<stage>/ and group files by capture_id.
    Returns { capture_id: { "stage": str, "bands": { band_name: Path } } }
    """
    captures: dict[str, dict] = defaultdict(lambda: {"stage": "", "bands": {}})

    for band_folder in BAND_FOLDERS:
        band_root = ms_bands_dir / band_folder
        if not band_root.exists():
            continue
        for stage_dir in band_root.iterdir():
            if not stage_dir.is_dir():
                continue
            stage = stage_dir.name
            for tif_file in sorted(stage_dir.glob("*.TIF")):
                # Filename: <timestamp>_<seq>_<band>.TIF
                stem = tif_file.stem  # e.g. 20250715172855_0001_MS_NIR
                parts = stem.rsplit("_", maxsplit=2)
                if len(parts) < 3:
                    continue
                capture_id = f"{parts[0]}_{parts[1]}"
                captures[capture_id]["stage"] = stage
                captures[capture_id]["bands"][band_folder] = tif_file

    return dict(captures)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 3: Compute vegetation indices")
    ap.add_argument("--ms-bands", default="organized/ms_bands",
                    help="Path to organised MS band directory")
    ap.add_argument("--indices",  default="organized/indices",
                    help="Output directory for index GeoTIFFs")
    ap.add_argument("--force",    action="store_true",
                    help="Recompute even if output file already exists")
    args = ap.parse_args()

    ms_bands_dir  = Path(args.ms_bands).resolve()
    indices_dir   = Path(args.indices).resolve()

    if not ms_bands_dir.exists():
        print(f"[ERROR] MS bands directory not found: {ms_bands_dir}")
        print("        Run 02_segregate.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Discover all captures with MS bands
    # ------------------------------------------------------------------
    print(f"[INFO] Scanning {ms_bands_dir} ...")
    captures = discover_captures(ms_bands_dir)
    print(f"[INFO] Found {len(captures)} unique captures with MS bands.")

    counters = {idx: {"built": 0, "skipped": 0, "failed": 0}
                for idx in INDICES}

    # ------------------------------------------------------------------
    # Process each capture × each index
    # ------------------------------------------------------------------
    for capture_id, cap_data in sorted(captures.items()):
        stage = cap_data["stage"]
        bands = cap_data["bands"]

        for idx_name, (nir_band, other_band) in INDICES.items():
            out_path = indices_dir / idx_name / stage / f"{capture_id}_{idx_name}.TIF"

            if out_path.exists() and not args.force:
                counters[idx_name]["skipped"] += 1
                continue

            # Check required bands are present
            if nir_band not in bands or other_band not in bands:
                missing_b = [b for b in [nir_band, other_band] if b not in bands]
                print(f"  [WARN] {capture_id} / {idx_name}: missing bands {missing_b} — skipping")
                counters[idx_name]["failed"] += 1
                continue

            try:
                nir_data,   profile = load_band(bands[nir_band])
                other_data, _       = load_band(bands[other_band])

                # Normalised difference
                numerator   = nir_data   - other_data
                denominator = nir_data   + other_data
                index_map   = safe_divide(numerator, denominator)

                write_index(index_map, profile, out_path)
                counters[idx_name]["built"] += 1
                print(f"  [OK] {idx_name}  {capture_id}  →  {out_path.name}")

            except Exception as exc:
                print(f"  [ERROR] {idx_name} / {capture_id}: {exc}")
                counters[idx_name]["failed"] += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VEGETATION INDEX SUMMARY")
    print("=" * 65)
    print(f"  {'Index':<8} {'Built':>6} {'Skipped':>8} {'Failed':>7}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*7}")
    for idx_name, c in counters.items():
        print(f"  {idx_name:<8} {c['built']:>6} {c['skipped']:>8} {c['failed']:>7}")
    total_built   = sum(c["built"]   for c in counters.values())
    total_skipped = sum(c["skipped"] for c in counters.values())
    total_failed  = sum(c["failed"]  for c in counters.values())
    print(f"  {'TOTAL':<8} {total_built:>6} {total_skipped:>8} {total_failed:>7}")
    print(f"\n  Output: {indices_dir}\n")


if __name__ == "__main__":
    main()
