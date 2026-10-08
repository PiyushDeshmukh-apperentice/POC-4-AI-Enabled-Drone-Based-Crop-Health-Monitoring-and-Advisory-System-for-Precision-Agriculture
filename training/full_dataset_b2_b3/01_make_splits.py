import os
import random
import csv
from pathlib import Path

# Paths
IMG_ROOT = Path(r"e:\paddy_test\test_500_rgb_images")
OUT_DIR = Path(r"e:\paddy_test\training\full_dataset_b2_b3\splits")

# Class mapping
BUCKET_MAP = {
    "Nursery_Stage": "Seedling",
    "Vegetative_Stage": "Vegetative",
    "Booting_Stage": "Flowering",
    "Flowering_Stage": "Flowering",
    "Mature_Stage": "Mature",
    "Mature_stage": "Mature",
}

def get_class_from_path(p: Path) -> str:
    for part in p.parts:
        for b_key, b_val in BUCKET_MAP.items():
            if b_key in part:
                return b_val
    return None

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {IMG_ROOT} for _D.JPG files...")
    all_images = list(IMG_ROOT.rglob("*_D.JPG"))
    print(f"Found {len(all_images)} images.")
    
    dataset = []
    for img in all_images:
        cls = get_class_from_path(img)
        if cls:
            dataset.append({
                "file_path": str(img.resolve()),
                "class_name": cls,
                "capture_id": img.stem.replace("_D", "")
            })
            
    print(f"Mapped {len(dataset)} images to valid classes.")
    
    # Shuffle randomly
    random.seed(42)
    random.shuffle(dataset)
    
    # 70/15/15 Split
    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_set = dataset[:n_train]
    val_set = dataset[n_train:n_train+n_val]
    test_set = dataset[n_train+n_val:]
    
    def save_split(split_name, data):
        out_file = OUT_DIR / f"{split_name}.csv"
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "class_name", "capture_id"])
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved {len(data)} rows to {out_file}")

    save_split("train", train_set)
    save_split("val", val_set)
    save_split("test", test_set)
    
    print("Dataset splitting complete.")

if __name__ == "__main__":
    main()
