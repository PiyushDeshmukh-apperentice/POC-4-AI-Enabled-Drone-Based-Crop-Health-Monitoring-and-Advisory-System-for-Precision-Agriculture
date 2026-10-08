import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

MANIFEST  = Path("../manifest.csv")
OUT_DIR   = Path("splits")
SEED      = 42
VAL_FRAC  = 0.18

BUCKET_MAP = {
    "Nursery_Stage":    "Seedling",
    "Vegetative_Stage": "Vegetative",
    "Booting_Stage":    "Flowering",
    "Flowering_Stage":  "Flowering",
    "Mature_Stage":     "Mature",
    "Mature_stage":     "Mature",
}

def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    source_rows = {r["capture_id"]: r for r in all_rows
                   if not r.get("tile_file_path") and r.get("capture_id")}
    tile_rows   = [r for r in all_rows if r.get("tile_file_path")]

    print(f"Source: {len(source_rows)}  Tiles: {len(tile_rows)}")

    f1_tiles, f2_tiles = [], []
    for r in tile_rows:
        bucket = BUCKET_MAP.get(r["growth_stage"])
        if not bucket:
            continue
        r["bucket"] = bucket
        src = source_rows.get(r.get("source_capture_id", ""))
        if not src:
            continue
        fp = src.get("field_plot", "")
        if fp.startswith("F1"):
            f1_tiles.append(r)
        elif fp.startswith("F2"):
            f2_tiles.append(r)

    print(f"F1 tiles: {len(f1_tiles)}  F2 tiles (test): {len(f2_tiles)}")

    # --- GroupShuffleSplit on F1 by source_capture_id ---
    groups = np.array([r["source_capture_id"] for r in f1_tiles])
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    train_idx, val_idx = next(gss.split(f1_tiles, groups=groups))

    train = [f1_tiles[i] for i in train_idx]
    val   = [f1_tiles[i] for i in val_idx]
    test  = f2_tiles

    # Verify no image leaks between train and val
    train_caps = {r["source_capture_id"] for r in train}
    val_caps   = {r["source_capture_id"] for r in val}
    leaks = train_caps & val_caps
    print(f"Leaking captures between train/val: {len(leaks)}  (must be 0)")

    fields = ["tile_file_path", "bucket", "growth_stage", "source_capture_id"]
    for split_name, rows in [("train", train), ("val", val), ("test", test)]:
        out_path = OUT_DIR / f"{split_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{k: r[k] for k in fields} for r in rows])

        counts = defaultdict(int)
        for r in rows:
            counts[r["bucket"]] += 1
        print(f"{split_name:6s}: {len(rows):6d} tiles  |  {dict(sorted(counts.items()))}")

    print(f"\nSplits -> {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
