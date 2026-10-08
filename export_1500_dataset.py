"""
export_1500_dataset.py

Exports a zip-ready dataset package containing:
- source_tif_images/: 45 full-resolution source captures (9 per growth stage class) in standard TIFF format.
- tiled_images/: All 35 cropped tiles per source capture (315 tiles per class, 1,575 total tiles).
- export_manifest.csv: Complete mapping of all 1,575 tiles to their source TIF file, crop offsets, and class.
- README.md: Comprehensive dataset documentation.
"""

import os
import shutil
import csv
from pathlib import Path
from PIL import Image
from collections import defaultdict

ORGANIZED_RGB = Path(r"e:\paddy_test\organized\rgb")
TILES_RGB = Path(r"e:\paddy_test\tiles\rgb")
EXPORT_DIR = Path(r"e:\paddy_test\paddy_dataset_export_1500")

STAGES = [
    "Nursery_Stage",
    "Vegetative_Stage",
    "Booting_Stage",
    "Flowering_Stage",
    "Mature_Stage"
]

CAPTURES_PER_CLASS = 9
TILES_PER_CAPTURE = 35


def main():
    print(f"Initializing dataset export to: {EXPORT_DIR}")
    
    source_tif_dir = EXPORT_DIR / "source_tif_images"
    tiled_dir = EXPORT_DIR / "tiled_images"
    
    # Clean export dir subfolders
    if source_tif_dir.exists():
        shutil.rmtree(source_tif_dir)
    if tiled_dir.exists():
        shutil.rmtree(tiled_dir)

    source_tif_dir.mkdir(parents=True, exist_ok=True)
    tiled_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    summary_stats = {}

    for stage in STAGES:
        print(f"\nProcessing stage: {stage}...")
        stage_src_tif_dir = source_tif_dir / stage
        stage_tile_dir = tiled_dir / stage
        stage_src_tif_dir.mkdir(parents=True, exist_ok=True)
        stage_tile_dir.mkdir(parents=True, exist_ok=True)

        stage_tiles_src_dir = TILES_RGB / stage
        stage_org_rgb_dir = ORGANIZED_RGB / stage

        # Group available tiles by capture_id
        capture_to_tiles = defaultdict(list)
        for tile_file in sorted(stage_tiles_src_dir.glob("*.jpg")):
            cid = tile_file.name.split("_tile")[0]
            capture_to_tiles[cid].append(tile_file)

        print(f"  Found {len(capture_to_tiles)} source captures in {stage} with existing tiles.")
        
        # Sort capture IDs for determinism
        sorted_captures = sorted(capture_to_tiles.keys())
        
        # Select 9 complete captures that have all 35 tiles
        selected_captures = []
        for cid in sorted_captures:
            tiles_for_cid = sorted(capture_to_tiles[cid])
            if len(tiles_for_cid) == TILES_PER_CAPTURE:
                selected_captures.append(cid)
            if len(selected_captures) == CAPTURES_PER_CLASS:
                break

        print(f"  Selected {len(selected_captures)} source captures (all {TILES_PER_CAPTURE} tiles each).")
        
        # 1. Convert and save full-resolution source captures to standard TIFF
        tif_saved_count = 0
        for cid in selected_captures:
            src_jpg_path = stage_org_rgb_dir / f"{cid}_D.JPG"
            if not src_jpg_path.exists():
                alt_matches = list(stage_org_rgb_dir.glob(f"{cid}*"))
                if alt_matches:
                    src_jpg_path = alt_matches[0]
                else:
                    raise FileNotFoundError(f"Could not find source JPG for capture {cid} in {stage_org_rgb_dir}")

            dst_tif_path = stage_src_tif_dir / f"{cid}.tif"
            
            # Save using standard baseline TIFF (compatible with Windows Photo Viewer, Photoshop, Paint, QGIS, etc.)
            with Image.open(src_jpg_path) as img:
                img.save(dst_tif_path, format="TIFF", compression="tiff_lzw")
            tif_saved_count += 1

        # 2. Copy ALL 35 tiles for each of the 9 captures (315 tiles per class)
        tiles_copied_count = 0
        for cid in selected_captures:
            tiles_for_cid = sorted(capture_to_tiles[cid])
            for tile_path in tiles_for_cid:
                dst_tile_path = stage_tile_dir / tile_path.name
                shutil.copy2(tile_path, dst_tile_path)
                tiles_copied_count += 1

                # Extract tile metadata
                tile_name = tile_path.name
                tile_idx_str = tile_name.split("_tile")[-1].split(".")[0]
                tile_idx = int(tile_idx_str) if tile_idx_str.isdigit() else -1

                manifest_rows.append({
                    "class": stage,
                    "tile_filename": tile_name,
                    "tile_relative_path": f"tiled_images/{stage}/{tile_name}",
                    "source_tif_filename": f"{cid}.tif",
                    "source_tif_relative_path": f"source_tif_images/{stage}/{cid}.tif",
                    "source_capture_id": cid,
                    "tile_index": tile_idx,
                })

        summary_stats[stage] = {
            "source_tif_count": tif_saved_count,
            "tiled_images_count": tiles_copied_count
        }

    # 3. Write manifest CSV
    manifest_csv_path = EXPORT_DIR / "export_manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class", "tile_filename", "tile_relative_path",
            "source_tif_filename", "source_tif_relative_path",
            "source_capture_id", "tile_index"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nSaved export manifest to {manifest_csv_path} ({len(manifest_rows)} rows)")

    # 4. Write README.md
    readme_path = EXPORT_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        tot_tif = sum(s["source_tif_count"] for s in summary_stats.values())
        tot_tiles = sum(s["tiled_images_count"] for s in summary_stats.values())

        f.write("# Paddy Crop Multi-Stage Dataset Export (Source TIFs + Full 35-Tile Sets)\n\n")
        f.write("This zip-ready dataset bundle contains full-resolution source images in standard TIFF format and their complete 35-tile sets balanced across all 5 paddy growth stages.\n\n")
        f.write("## Dataset Structure\n\n")
        f.write("```\n")
        f.write("paddy_dataset_export_1500/\n")
        f.write("├── README.md                      <- Dataset description and breakdown\n")
        f.write("├── export_manifest.csv            <- Full mapping of every tile to its source TIF\n")
        f.write("├── source_tif_images/             <- Full-resolution source images (Standard TIFF format)\n")
        for stage in STAGES:
            f.write(f"│   ├── {stage}/          <- {summary_stats[stage]['source_tif_count']} source .tif files\n")
        f.write("└── tiled_images/                  <- Cropped CNN-ready tiles (1024x1024 px)\n")
        for stage in STAGES:
            f.write(f"    ├── {stage}/          <- {summary_stats[stage]['tiled_images_count']} tile images (9 captures x 35 tiles)\n")
        f.write("```\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Growth Stage (Class) | Source TIF Images | Tiled Images (1024x1024) | Tiles Per Source Image |\n")
        f.write("|---|---|---|---|\n")
        for stage, stats in summary_stats.items():
            f.write(f"| **{stage}** | {stats['source_tif_count']} | {stats['tiled_images_count']:,} | 35 tiles/image |\n")
        f.write(f"| **TOTAL** | **{tot_tif}** | **{tot_tiles:,}** | **35 tiles/image** |\n\n")
        f.write("## Tiling Specifications\n")
        f.write("- **Full Image Resolution**: 5280 x 3956 pixels (20MP)\n")
        f.write("- **Tile Size**: 1024 x 1024 pixels\n")
        f.write("- **Overlap**: 25% (step stride 768 px)\n")
        f.write("- **Grid Layout**: 7 columns x 5 rows = 35 tiles per full image (complete set for all captures)\n")
        f.write("- **Mapping**: Each tile links directly to its source TIF in `export_manifest.csv`\n")

    print(f"Saved README to {readme_path}")
    print(f"\n=======================================================")
    print(f"Export Completed Successfully!")
    print(f"  Total Source TIFs: {tot_tif} (9 per class)")
    print(f"  Total Tiles:       {tot_tiles} (315 per class, all 35 tiles/image)")
    print(f"  Location:          {EXPORT_DIR}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
