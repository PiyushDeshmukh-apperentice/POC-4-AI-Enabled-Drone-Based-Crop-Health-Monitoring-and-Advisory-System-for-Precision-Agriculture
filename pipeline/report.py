#!/usr/bin/env python3
"""
report.py — Pipeline Production Report
Counts source images, outputs produced (organized, indices, tiles) per stage and variant.
"""
import csv
import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime

ROOT         = Path(__file__).parent.parent   # e:\paddy_test
MANIFEST     = ROOT / "manifest.csv"
ORGANIZED    = ROOT / "organized"
TILES        = ROOT / "tiles"
REPORT_OUT   = ROOT / "pipeline_report.txt"

SEP  = "=" * 72
SEP2 = "-" * 72

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

# ─────────────────────────────────────────────
# 1. Read manifest (source rows only)
# ─────────────────────────────────────────────
source_rows = []
if MANIFEST.exists():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("tile_file_path", "").strip():
                source_rows.append(row)

# Group source rows: stage → variant → count
src_by_stage_variant = defaultdict(lambda: defaultdict(int))
src_by_stage_variant_dims = defaultdict(lambda: defaultdict(set))
for row in source_rows:
    stage   = row["growth_stage"]
    variant = row["variant"]
    src_by_stage_variant[stage][variant] += 1
    src_by_stage_variant_dims[stage][variant].add(
        f"{row['width_px']}x{row['height_px']}"
    )

all_stages   = sorted(src_by_stage_variant.keys())
all_variants = ["D", "MS_G", "MS_R", "MS_RE", "MS_NIR"]

# ─────────────────────────────────────────────
# 2. Count organized/ files
# ─────────────────────────────────────────────
# organized/rgb/<stage>/               → variant = D
# organized/ms_bands/<band>/<stage>/   → variant = band
# organized/indices/<idx>/<stage>/     → variant = idx

org_by_stage_variant = defaultdict(lambda: defaultdict(int))

rgb_dir = ORGANIZED / "rgb"
if rgb_dir.exists():
    for stage_dir in rgb_dir.iterdir():
        if stage_dir.is_dir():
            cnt = sum(1 for f in stage_dir.iterdir() if f.is_file())
            org_by_stage_variant[stage_dir.name]["D"] += cnt

ms_dir = ORGANIZED / "ms_bands"
if ms_dir.exists():
    for band_dir in ms_dir.iterdir():
        if band_dir.is_dir():
            for stage_dir in band_dir.iterdir():
                if stage_dir.is_dir():
                    cnt = sum(1 for f in stage_dir.iterdir() if f.is_file())
                    org_by_stage_variant[stage_dir.name][band_dir.name] += cnt

idx_dir = ORGANIZED / "indices"
idx_by_stage_index = defaultdict(lambda: defaultdict(int))
if idx_dir.exists():
    for index_dir in idx_dir.iterdir():
        if index_dir.is_dir():
            for stage_dir in index_dir.iterdir():
                if stage_dir.is_dir():
                    cnt = sum(1 for f in stage_dir.iterdir() if f.is_file())
                    idx_by_stage_index[stage_dir.name][index_dir.name] += cnt

# ─────────────────────────────────────────────
# 3. Count tiles/ files
# ─────────────────────────────────────────────
# tiles/rgb/<stage>/
# tiles/ms_bands/<band>/<stage>/
# tiles/indices/<idx>/<stage>/

tile_by_stage_stream = defaultdict(lambda: defaultdict(int))

tile_rgb = TILES / "rgb"
if tile_rgb.exists():
    for stage_dir in tile_rgb.iterdir():
        if stage_dir.is_dir():
            cnt = sum(1 for f in stage_dir.rglob("*") if f.is_file())
            tile_by_stage_stream[stage_dir.name]["rgb (D)"] += cnt

tile_ms = TILES / "ms_bands"
if tile_ms.exists():
    for band_dir in tile_ms.iterdir():
        if band_dir.is_dir():
            for stage_dir in band_dir.iterdir():
                if stage_dir.is_dir():
                    cnt = sum(1 for f in stage_dir.rglob("*") if f.is_file())
                    tile_by_stage_stream[stage_dir.name][f"ms_bands ({band_dir.name})"] += cnt

tile_idx = TILES / "indices"
if tile_idx.exists():
    for idx_name_dir in tile_idx.iterdir():
        if idx_name_dir.is_dir():
            for stage_dir in idx_name_dir.iterdir():
                if stage_dir.is_dir():
                    cnt = sum(1 for f in stage_dir.rglob("*") if f.is_file())
                    tile_by_stage_stream[stage_dir.name][f"indices ({idx_name_dir.name})"] += cnt

# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────
log(SEP)
log(f"  IMPaCT-UAV-MsRGB  -  Pipeline Production Report")
log(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(SEP)

# ── SECTION 1: Source images ─────────────────
log()
log("SECTION 1 - SOURCE IMAGES (downloaded, in test_1000_images/)")
log(SEP2)
log(f"  {'Stage':<30} {'Variant':<12} {'Count':>6}  {'Dimensions'}")
log(f"  {'-'*30} {'-'*12} {'-'*6}  {'-'*20}")

total_src = 0
for stage in all_stages:
    first = True
    for variant in all_variants:
        cnt = src_by_stage_variant[stage].get(variant, 0)
        if cnt == 0:
            continue
        dims = ", ".join(sorted(src_by_stage_variant_dims[stage][variant]))
        stage_label = stage if first else ""
        log(f"  {stage_label:<30} {variant:<12} {cnt:>6}  {dims}")
        first = False
        total_src += cnt
log(f"  {'':<30} {'TOTAL':<12} {total_src:>6}")

# ── SECTION 2: Organised files ───────────────
log()
log("SECTION 2 - ORGANISED FILES (hardlinks/copies in organized/)")
log(SEP2)
log(f"  Phase 2: RGB + MS bands   (organized/rgb/ and organized/ms_bands/)")
log()
log(f"  {'Stage':<30} {'Variant':<12} {'Source':>8}  {'Organized':>10}  {'Match?':>6}")
log(f"  {'-'*30} {'-'*12} {'-'*8}  {'-'*10}  {'-'*6}")

for stage in all_stages:
    first = True
    for variant in all_variants:
        src_cnt = src_by_stage_variant[stage].get(variant, 0)
        org_cnt = org_by_stage_variant[stage].get(variant, 0)
        match   = "[OK]" if src_cnt == org_cnt else f"MISMATCH ({org_cnt})"
        if src_cnt == 0 and org_cnt == 0:
            continue
        stage_label = stage if first else ""
        log(f"  {stage_label:<30} {variant:<12} {src_cnt:>8}  {org_cnt:>10}  {match:>6}")
        first = False

# ── SECTION 3: Vegetation indices ────────────
log()
log("SECTION 3 - VEGETATION INDICES (organized/indices/)")
log(SEP2)
all_indices = sorted({idx for s in idx_by_stage_index.values() for idx in s})
# Also count expected: one index per capture (capture = one set of 4 MS bands)
captures_per_stage = defaultdict(set)
for row in source_rows:
    if row["variant"] != "D":
        captures_per_stage[row["growth_stage"]].add(row["capture_id"])

log(f"  {'Stage':<30} {'Index':<8} {'Expected':>10}  {'Produced':>10}  {'Match?':>6}")
log(f"  {'-'*30} {'-'*8} {'-'*10}  {'-'*10}  {'-'*6}")

for stage in all_stages:
    expected_per_idx = len(captures_per_stage[stage])
    first = True
    for idx in ["NDVI", "NDRE", "GNDVI"]:
        produced = idx_by_stage_index[stage].get(idx, 0)
        match    = "[OK]" if produced == expected_per_idx else f"MISMATCH"
        stage_label = stage if first else ""
        log(f"  {stage_label:<30} {idx:<8} {expected_per_idx:>10}  {produced:>10}  {match:>6}")
        first = False

# ── SECTION 4: Tiles ─────────────────────────
log()
log("SECTION 4 - TILES (tiles/)")
log(SEP2)

if not any(tile_by_stage_stream.values()):
    log("  No tiles found. Run:  python pipeline\\04_tile.py")
else:
    all_tile_streams = sorted({s for st in tile_by_stage_stream.values() for s in st})
    log(f"  {'Stage':<30} {'Stream':<26} {'Tiles':>8}  {'Per Image':>10}")
    log(f"  {'-'*30} {'-'*26} {'-'*8}  {'-'*10}")
    for stage in sorted(tile_by_stage_stream.keys()):
        first = True
        for stream in all_tile_streams:
            cnt = tile_by_stage_stream[stage].get(stream, 0)
            if cnt == 0:
                continue
            # Determine source count for this stream
            if "rgb" in stream.lower() or "(D)" in stream:
                n_src = src_by_stage_variant[stage].get("D", 1)
            elif "indices" in stream.lower():
                n_src = src_by_stage_variant[stage].get("MS_G", 1)
            else:
                band = stream.split("(")[-1].rstrip(")")
                n_src = src_by_stage_variant[stage].get(band, 1) or 1
            per_img = f"{cnt/n_src:.1f}" if n_src else "-"
            stage_label = stage if first else ""
            log(f"  {stage_label:<30} {stream:<26} {cnt:>8}  {per_img:>10}")
            first = False

# ── GRAND TOTALS ─────────────────────────────
log()
log(SEP)
log("  GRAND TOTALS")
log(SEP2)

total_org = sum(
    cnt
    for stage_data in org_by_stage_variant.values()
    for cnt in stage_data.values()
)
total_idx = sum(
    cnt
    for stage_data in idx_by_stage_index.values()
    for cnt in stage_data.values()
)
total_tiles = sum(
    cnt
    for stage_data in tile_by_stage_stream.values()
    for cnt in stage_data.values()
)
log(f"  Source files (downloaded)     : {total_src}")
log(f"  Organised files (Phase 2)     : {total_org}")
log(f"  Index maps produced (Phase 3) : {total_idx}")
log(f"  Tiles produced (Phase 4)      : {total_tiles if total_tiles else 'Phase 4 not run yet'}")
log()
log(f"  Report saved to: {REPORT_OUT}")
log(SEP)

# Write to file
with open(REPORT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
