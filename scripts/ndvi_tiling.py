import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset_multispectral/Nursery_Stage"
OUTPUT_DIR   = "ndvi_output/Nursery_Stage"
N_TILES      = 6          # change to 3 for Rucha's version
GRID_ROWS    = 2          # grid layout: 2 rows x 3 cols = 6 tiles
GRID_COLS    = 3          # change to 1x3 for Rucha's 3-tile version
CSV_OUTPUT   = "ndvi_output/ndvi_results_nursery_6tiles.csv"

# Paddy-specific NDVI thresholds (from literature)
def classify(ndvi):
    if ndvi < 0.15:
        return "No Vegetation"
    elif ndvi < 0.21:
        return "Severe Stress"
    elif ndvi < 0.35:
        return "Nursery / Sparse"
    elif ndvi < 0.50:
        return "Moderate Growth"
    else:
        return "Healthy / Dense"

CATEGORY_COLORS = {
    "No Vegetation" : "#d73027",
    "Severe Stress"  : "#fc8d59",
    "Nursery / Sparse": "#fee08b",
    "Moderate Growth": "#91cf60",
    "Healthy / Dense": "#1a9850",
}
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)

nir_dir = os.path.join(DATASET_DIR, "MS_NIR")
red_dir = os.path.join(DATASET_DIR, "MS_R")

# Match pairs by image number (e.g. 0001, 0002 ...)
nir_files = sorted(os.listdir(nir_dir))
red_files = sorted(os.listdir(red_dir))

print(f"Found {len(nir_files)} NIR and {len(red_files)} Red images\n")

csv_rows = []

for nir_fname, red_fname in zip(nir_files, red_files):
    img_id = nir_fname.split("_")[1]   # e.g. 0001
    print(f"Processing image {img_id} ...")

    nir_path = os.path.join(nir_dir, nir_fname)
    red_path = os.path.join(red_dir, red_fname)

    with rasterio.open(nir_path) as nir_src:
        nir = nir_src.read(1).astype(np.float32)

    with rasterio.open(red_path) as red_src:
        red = red_src.read(1).astype(np.float32)

    # ── Compute NDVI ──────────────────────────────────────────────────────────
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = np.where(
        (nir + red) == 0,
        np.nan,
        (nir - red) / (nir + red)
    )
    ndvi = np.clip(ndvi, -1, 1)

    H, W = ndvi.shape

    # ── Split into grid tiles (GRID_ROWS x GRID_COLS) ─────────────────────────
    row_edges = np.linspace(0, H, GRID_ROWS + 1, dtype=int)
    col_edges = np.linspace(0, W, GRID_COLS + 1, dtype=int)

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(5 * GRID_COLS, 4 * GRID_ROWS))
    fig.suptitle(f"Image {img_id} — {N_TILES} NDVI Tiles ({GRID_ROWS}×{GRID_COLS} grid) — Nursery Stage", fontsize=13)

    t = 0
    im = None
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            r0, r1 = row_edges[r], row_edges[r + 1]
            c0, c1 = col_edges[c], col_edges[c + 1]
            tile = ndvi[r0:r1, c0:c1]

            mean_ndvi = float(np.nanmean(tile))
            category  = classify(mean_ndvi)

            # save each tiled NDVI image as PNG
            tile_fname = f"img{img_id}_tile{t+1:02d}_ndvi.png"
            tile_path = os.path.join(OUTPUT_DIR, tile_fname)

            plt.imsave(
                tile_path,
                tile,
                cmap="RdYlGn",
                vmin=-1,
                vmax=1
            )           

            # Plot tile
            ax = axes[r][c] if GRID_ROWS > 1 else axes[c]
            im = ax.imshow(tile, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_title(f"Tile {t+1}\nNDVI: {mean_ndvi:.3f}\n{category}", fontsize=9)
            ax.axis("off")

            csv_rows.append({
                "image_id"  : img_id,
                "tile"      : t + 1,
                "mean_ndvi" : round(mean_ndvi, 4),
                "category"  : category,
            })
            t += 1

    plt.colorbar(im, ax=axes[GRID_ROWS-1][GRID_COLS-1] if GRID_ROWS > 1 else axes[-1], fraction=0.046, pad=0.04, label="NDVI")
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, f"img{img_id}_ndvi_tiles.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {fig_path}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "tile", "mean_ndvi", "category"])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\nCSV saved → {CSV_OUTPUT}")

# ── Summary ───────────────────────────────────────────────────────────────────
all_ndvi = [r["mean_ndvi"] for r in csv_rows]
print(f"\n{'─'*45}")
print(f"  Nursery Stage — {N_TILES}-tile Summary")
print(f"{'─'*45}")
print(f"  Total tiles   : {len(all_ndvi)}")
print(f"  Mean NDVI     : {np.mean(all_ndvi):.4f}")
print(f"  Min NDVI      : {np.min(all_ndvi):.4f}")
print(f"  Max NDVI      : {np.max(all_ndvi):.4f}")
print(f"{'─'*45}")

# Category breakdown
from collections import Counter
counts = Counter(r["category"] for r in csv_rows)
for cat, n in counts.items():
    print(f"  {cat:20s}: {n} tiles")
print(f"{'─'*45}")
