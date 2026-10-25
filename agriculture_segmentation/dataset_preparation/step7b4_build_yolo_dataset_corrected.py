from pathlib import Path
import csv
import shutil

from PIL import Image
import numpy as np
import cv2


# ============================================================
# STEP 7B-4 - BUILD CORRECTED YOLOv8 SEGMENTATION DATASET
# ============================================================

import os

PROJECT_ROOT = Path(os.environ.get(
    "CDAC_DATA_ROOT",
    Path.home() / "OneDrive" / "Desktop" / "CDAC"
)).expanduser()

SOURCE_DATASET = PROJECT_ROOT / "step6c3_model_dataset_v2"

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "YOLOv8"
    / "step7b4_yolo_dataset"
)

CLASS_NAMES = {
    1: "Nutrition_Deficiency",
    2: "Weed",
}

YOLO_CLASS_IDS = {
    1: 0,
    2: 1,
}

SPLITS = ["train", "val", "test"]


# ============================================================
# SAFETY CHECKS
# ============================================================

print("=" * 70)
print("STEP 7B-4 - BUILD CORRECTED YOLOv8 SEGMENTATION DATASET")
print("=" * 70)

print()
print("Checking source dataset...")

if not SOURCE_DATASET.exists():
    print("ERROR: Source dataset not found:")
    print(SOURCE_DATASET)
    raise SystemExit(1)

for split in SPLITS:
    image_dir = SOURCE_DATASET / split / "images"
    mask_dir = SOURCE_DATASET / split / "masks"

    if not image_dir.exists():
        print("ERROR: Missing image directory:", image_dir)
        raise SystemExit(1)

    if not mask_dir.exists():
        print("ERROR: Missing mask directory:", mask_dir)
        raise SystemExit(1)

print("Source dataset verified.")


# Do not overwrite an existing non-empty corrected dataset.
if OUTPUT_DATASET.exists() and any(OUTPUT_DATASET.iterdir()):
    print()
    print("ERROR: Output directory already exists and is not empty:")
    print(OUTPUT_DATASET)
    print()
    print("This script will NOT overwrite it.")
    raise SystemExit(1)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in SPLITS:
    (OUTPUT_DATASET / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DATASET / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


conversion_rows = []
error_rows = []

total_images = 0
total_polygons = 0
nutrition_polygons = 0
weed_polygons = 0


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):

    global total_images
    global total_polygons
    global nutrition_polygons
    global weed_polygons

    source_image_dir = SOURCE_DATASET / split / "images"
    source_mask_dir = SOURCE_DATASET / split / "masks"

    output_image_dir = OUTPUT_DATASET / "images" / split
    output_label_dir = OUTPUT_DATASET / "labels" / split

    image_files = sorted(
        [
            p for p in source_image_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    print()
    print("-" * 70)
    print(f"PROCESSING {split.upper()}")
    print("-" * 70)
    print("Source images:", len(image_files))

    split_images_created = 0
    split_labels_created = 0
    split_polygons = 0

    for index, image_path in enumerate(image_files, start=1):

        total_images += 1

        mask_path = source_mask_dir / (
            image_path.stem + ".png"
        )

        if not mask_path.exists():

            error_rows.append({
                "split": split,
                "image": image_path.name,
                "error": "Source mask not found",
            })

            continue

        try:

            # ------------------------------------------------
            # Read image
            # ------------------------------------------------

            image = Image.open(image_path)
            width, height = image.size

            # ------------------------------------------------
            # Read categorical mask
            # ------------------------------------------------

            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask, dtype=np.uint8)

            if mask_array.shape != (height, width):

                error_rows.append({
                    "split": split,
                    "image": image_path.name,
                    "error": (
                        f"Dimension mismatch: "
                        f"image={width}x{height}, "
                        f"mask="
                        f"{mask_array.shape[1]}x"
                        f"{mask_array.shape[0]}"
                    ),
                })

                continue

            yolo_lines = []

            image_polygon_count = 0
            image_nutrition_polygons = 0
            image_weed_polygons = 0
            ignored_zero_area = 0

            # ------------------------------------------------
            # Process Nutrition and Weed
            # ------------------------------------------------

            for source_class_id in [1, 2]:

                binary_mask = (
                    mask_array == source_class_id
                ).astype(np.uint8)

                contours, _ = cv2.findContours(
                    binary_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:

                    # ------------------------------------------------
                    # IMPORTANT:
                    # Ignore zero-area contours.
                    # These occur for isolated one-pixel annotations.
                    # The original mask is NOT changed.
                    # ------------------------------------------------

                    contour_area = cv2.contourArea(contour)

                    if contour_area <= 0:
                        ignored_zero_area += 1
                        continue

                    # Need at least 3 contour points.
                    if len(contour) < 3:
                        ignored_zero_area += 1
                        continue

                    # ------------------------------------------------
                    # Convert contour points to YOLO normalized format
                    # ------------------------------------------------

                    points = contour.reshape(-1, 2)

                    normalized_points = []

                    for x, y in points:

                        x_norm = float(x) / float(width)
                        y_norm = float(y) / float(height)

                        # Clamp against possible floating-point drift.
                        x_norm = max(0.0, min(1.0, x_norm))
                        y_norm = max(0.0, min(1.0, y_norm))

                        normalized_points.extend(
                            [x_norm, y_norm]
                        )

                    # YOLO segmentation needs at least 3 points.
                    if len(normalized_points) < 6:

                        ignored_zero_area += 1
                        continue

                    yolo_class_id = YOLO_CLASS_IDS[
                        source_class_id
                    ]

                    line = (
                        str(yolo_class_id)
                        + " "
                        + " ".join(
                            f"{value:.6f}"
                            for value in normalized_points
                        )
                    )

                    yolo_lines.append(line)

                    image_polygon_count += 1

                    if source_class_id == 1:
                        image_nutrition_polygons += 1
                    elif source_class_id == 2:
                        image_weed_polygons += 1

            # ------------------------------------------------
            # If no usable polygon exists, document exclusion.
            # ------------------------------------------------

            if len(yolo_lines) == 0:

                error_rows.append({
                    "split": split,
                    "image": image_path.name,
                    "error": (
                        "No usable YOLO polygons. "
                        "Original mask retained."
                    ),
                })

                continue

            # ------------------------------------------------
            # Copy image
            # ------------------------------------------------

            output_image_path = (
                output_image_dir / image_path.name
            )

            shutil.copy2(
                image_path,
                output_image_path
            )

            # ------------------------------------------------
            # Write YOLO label
            # ------------------------------------------------

            output_label_path = (
                output_label_dir
                / f"{image_path.stem}.txt"
            )

            with open(
                output_label_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(yolo_lines)
                )

                f.write("\n")

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            split_images_created += 1
            split_labels_created += 1
            split_polygons += image_polygon_count

            total_polygons += image_polygon_count
            nutrition_polygons += image_nutrition_polygons
            weed_polygons += image_weed_polygons

            conversion_rows.append({
                "split": split,
                "image": image_path.name,
                "polygons": image_polygon_count,
                "nutrition_polygons":
                    image_nutrition_polygons,
                "weed_polygons":
                    image_weed_polygons,
                "ignored_zero_area_contours":
                    ignored_zero_area,
            })

        except Exception as exc:

            error_rows.append({
                "split": split,
                "image": image_path.name,
                "error": repr(exc),
            })

        if index % 250 == 0:
            print(
                f"Processed {index}/{len(image_files)}"
            )

    print()
    print(f"{split} conversion complete")
    print("Images created :", split_images_created)
    print("Labels created :", split_labels_created)
    print("Polygons       :", split_polygons)

    return (
        len(image_files),
        split_images_created,
        split_labels_created,
        split_polygons,
    )


# ============================================================
# RUN ALL SPLITS
# ============================================================

split_results = {}

for split in SPLITS:
    split_results[split] = process_split(split)


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_text = f"""path: {OUTPUT_DATASET.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: Nutrition_Deficiency
  1: Weed
"""

with open(
    OUTPUT_DATASET / "data.yaml",
    "w",
    encoding="utf-8"
) as f:

    f.write(yaml_text)


# ============================================================
# SAVE CONVERSION REPORT
# ============================================================

report_path = (
    OUTPUT_DATASET
    / "step7b4_conversion_report.csv"
)

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "split",
        "image",
        "polygons",
        "nutrition_polygons",
        "weed_polygons",
        "ignored_zero_area_contours",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(conversion_rows)


# ============================================================
# SAVE ERROR REPORT
# ============================================================

error_path = (
    OUTPUT_DATASET
    / "step7b4_conversion_errors.csv"
)

with open(
    error_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "split",
        "image",
        "error",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(error_rows)


# ============================================================
# FINAL COUNTS
# ============================================================

print()
print("=" * 70)
print("STEP 7B-4 - FINAL SUMMARY")
print("=" * 70)

print(
    "Total source images processed :",
    total_images
)

print(
    "Total usable images created   :",
    sum(
        split_results[s][1]
        for s in SPLITS
    )
)

print(
    "Total polygons                :",
    total_polygons
)

print(
    "Nutrition polygons            :",
    nutrition_polygons
)

print(
    "Weed polygons                 :",
    weed_polygons
)

print(
    "Conversion exclusions/errors  :",
    len(error_rows)
)

print()
print("Expected source image counts:")

for split in SPLITS:

    expected = split_results[split][0]
    created = split_results[split][1]
    labels = split_results[split][2]

    print(
        f"{split:5s} | "
        f"expected={expected:4d} | "
        f"images={created:4d} | "
        f"labels={labels:4d}"
    )

print()
print("Output dataset:")
print(OUTPUT_DATASET)

print()
print("Conversion report:")
print(report_path)

print()
print("Error/exclusion report:")
print(error_path)

print()
print("data.yaml:")
print(OUTPUT_DATASET / "data.yaml")

print()
print("=" * 70)

if len(error_rows) == 1:

    print(
        "STEP 7B-4 COMPLETE - "
        "ONE DOCUMENTED YOLO EXCLUSION"
    )

    print()
    print(
        "This is the one-pixel Nutrition annotation "
        "that cannot form a usable polygon."
    )

elif len(error_rows) == 0:

    print(
        "STEP 7B-4 COMPLETE - "
        "NO CONVERSION ERRORS"
    )

else:

    print(
        "STEP 7B-4 COMPLETE - "
        "ERRORS FOUND"
    )

print("=" * 70)