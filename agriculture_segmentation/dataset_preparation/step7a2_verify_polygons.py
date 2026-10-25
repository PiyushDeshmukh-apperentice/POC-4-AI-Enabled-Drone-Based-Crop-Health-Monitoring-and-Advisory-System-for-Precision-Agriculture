from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import pandas as pd


# ============================================================
# STEP 7A-2
# Verify YOLO polygons against original segmentation masks
# ============================================================

import os

PROJECT_ROOT = Path(os.environ.get(
    "CDAC_DATA_ROOT",
    Path.home() / "OneDrive" / "Desktop" / "CDAC"
)).expanduser()

MODEL_ROOT = (
    PROJECT_ROOT / "step6c3_model_dataset_v2"
)

PILOT_ROOT = (
    PROJECT_ROOT
    / "YOLOv8"
    / "step7a_polygon_pilot"
)

LABEL_DIR = PILOT_ROOT / "labels"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "YOLOv8"
    / "step7a2_polygon_verification"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_IMAGES = (
    MODEL_ROOT / "train" / "images"
)

TRAIN_MASKS = (
    MODEL_ROOT / "train" / "masks"
)


# ============================================================
# YOLO classes
# ============================================================

# Original common mask:
# 0 = Background
# 1 = Nutrition
# 2 = Weed
#
# YOLO:
# 0 = Nutrition
# 1 = Weed

ORIGINAL_TO_YOLO = {
    1: 0,
    2: 1,
}


# ============================================================
# IoU function
# ============================================================

def calculate_iou(original, reconstructed):

    intersection = np.logical_and(
        original,
        reconstructed
    ).sum()

    union = np.logical_or(
        original,
        reconstructed
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# Read YOLO polygon label
# ============================================================

def reconstruct_class_masks(
    label_path,
    width,
    height
):

    nutrition_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    weed_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]


    for line in lines:

        parts = line.split()

        if len(parts) < 7:
            continue

        class_id = int(parts[0])

        coordinates = list(
            map(float, parts[1:])
        )

        points = []

        for i in range(
            0,
            len(coordinates),
            2
        ):

            x_norm = coordinates[i]
            y_norm = coordinates[i + 1]

            x = round(
                x_norm * (width - 1)
            )

            y = round(
                y_norm * (height - 1)
            )

            points.append(
                [x, y]
            )

        if len(points) < 3:
            continue

        polygon = np.array(
            points,
            dtype=np.int32
        )

        if class_id == 0:

            cv2.fillPoly(
                nutrition_mask,
                [polygon],
                1
            )

        elif class_id == 1:

            cv2.fillPoly(
                weed_mask,
                [polygon],
                1
            )


    return nutrition_mask, weed_mask


# ============================================================
# Select the same 12 pilot images
# ============================================================

av_images = sorted(
    [
        p for p in TRAIN_IMAGES.iterdir()
        if p.is_file()
        and p.stem.startswith("AV_")
    ]
)[:6]

wr_images = sorted(
    [
        p for p in TRAIN_IMAGES.iterdir()
        if p.is_file()
        and p.stem.startswith("WR_")
    ]
)[:6]

selected_images = av_images + wr_images


# ============================================================
# Verification
# ============================================================

results = []

print("=" * 70)
print("STEP 7A-2 - YOLO POLYGON ROUND-TRIP VERIFICATION")
print("=" * 70)

for image_path in selected_images:

    mask_path = (
        TRAIN_MASKS
        / f"{image_path.stem}.png"
    )

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    print(
        f"\nChecking: {image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    original_mask = np.array(
        Image.open(mask_path)
    )

    height, width = image.shape[:2]

    reconstructed_nutrition, reconstructed_weed = (
        reconstruct_class_masks(
            label_path,
            width,
            height
        )
    )

    original_nutrition = (
        original_mask == 1
    )

    original_weed = (
        original_mask == 2
    )

    nutrition_iou = calculate_iou(
        original_nutrition,
        reconstructed_nutrition.astype(bool)
    )

    weed_iou = calculate_iou(
        original_weed,
        reconstructed_weed.astype(bool)
    )

    original_nutrition_pixels = (
        original_nutrition.sum()
    )

    reconstructed_nutrition_pixels = (
        reconstructed_nutrition.sum()
    )

    original_weed_pixels = (
        original_weed.sum()
    )

    reconstructed_weed_pixels = (
        reconstructed_weed.sum()
    )

    nutrition_difference = (
        reconstructed_nutrition_pixels
        - original_nutrition_pixels
    )

    weed_difference = (
        reconstructed_weed_pixels
        - original_weed_pixels
    )

    print(
        f"Nutrition IoU : {nutrition_iou:.6f}"
    )

    print(
        f"Weed IoU      : {weed_iou:.6f}"
    )

    print(
        f"Nutrition pixel difference: "
        f"{nutrition_difference}"
    )

    print(
        f"Weed pixel difference: "
        f"{weed_difference}"
    )

    results.append({
        "image": image_path.name,
        "source": (
            "AV"
            if image_path.stem.startswith("AV_")
            else "Weedy_Rice"
        ),
        "nutrition_iou": nutrition_iou,
        "weed_iou": weed_iou,
        "original_nutrition_pixels":
            original_nutrition_pixels,
        "reconstructed_nutrition_pixels":
            reconstructed_nutrition_pixels,
        "nutrition_pixel_difference":
            nutrition_difference,
        "original_weed_pixels":
            original_weed_pixels,
        "reconstructed_weed_pixels":
            reconstructed_weed_pixels,
        "weed_pixel_difference":
            weed_difference,
    })


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(results)

report_path = (
    OUTPUT_DIR
    / "step7a2_polygon_verification.csv"
)

results_df.to_csv(
    report_path,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(
    f"Images checked: {len(results_df)}"
)

print(
    f"Minimum Nutrition IoU: "
    f"{results_df['nutrition_iou'].min():.6f}"
)

print(
    f"Minimum Weed IoU: "
    f"{results_df['weed_iou'].min():.6f}"
)

print(
    f"Mean Nutrition IoU: "
    f"{results_df['nutrition_iou'].mean():.6f}"
)

print(
    f"Mean Weed IoU: "
    f"{results_df['weed_iou'].mean():.6f}"
)

print(
    f"\nReport:"
)

print(report_path)

print("\nSTEP 7A-2 COMPLETE")
print(
    "Do NOT generate the full YOLO dataset until "
    "these results are reviewed."
)