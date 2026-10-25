from pathlib import Path
import csv

from PIL import Image
import numpy as np
import cv2


# ============================================================
# STEP 7B-2 - SCAN ALL MODEL-DATASET MASKS
# Find regions that may not produce usable YOLO polygons.
# READ-ONLY: no files are modified.
# ============================================================

import os

PROJECT_ROOT = Path(os.environ.get(
    "CDAC_DATA_ROOT",
    Path.home() / "OneDrive" / "Desktop" / "CDAC"
)).expanduser()

MODEL_DATASET = PROJECT_ROOT / "step6c3_model_dataset_v2"

OUTPUT_DIR = PROJECT_ROOT / "YOLOv8" / "step7b2_tiny_region_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "step7b2_tiny_region_analysis.csv"
OUTPUT_REPORT = OUTPUT_DIR / "step7b2_tiny_region_report.txt"


print("=" * 70)
print("STEP 7B-2 - SCAN ALL MODEL-DATASET MASKS")
print("=" * 70)

all_rows = []

total_images = 0
total_masks = 0
nutrition_images = 0
weed_images = 0

nutrition_regions = 0
weed_regions = 0

nutrition_zero_area_contours = 0
weed_zero_area_contours = 0

nutrition_one_pixel_components = 0
weed_one_pixel_components = 0

errors = 0


# ------------------------------------------------------------
# Process train / val / test
# ------------------------------------------------------------

for split in ["train", "val", "test"]:

    image_dir = MODEL_DATASET / split / "images"
    mask_dir = MODEL_DATASET / split / "masks"

    print()
    print("-" * 70)
    print(f"PROCESSING {split.upper()}")
    print("-" * 70)

    image_files = sorted(
        [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    print("Images:", len(image_files))

    for index, image_path in enumerate(image_files, start=1):

        total_images += 1

        mask_path = mask_dir / f"{image_path.stem}.png"

        if not mask_path.exists():
            print("ERROR - mask missing:", image_path.name)
            errors += 1
            continue

        total_masks += 1

        try:
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask, dtype=np.uint8)

            # ------------------------------------------------
            # Nutrition class = 1
            # Weed class = 2
            # ------------------------------------------------

            nutrition_mask = (
                mask_array == 1
            ).astype(np.uint8)

            weed_mask = (
                mask_array == 2
            ).astype(np.uint8)

            nutrition_pixels = int(np.sum(nutrition_mask))
            weed_pixels = int(np.sum(weed_mask))

            if nutrition_pixels > 0:
                nutrition_images += 1

            if weed_pixels > 0:
                weed_images += 1

            # ------------------------------------------------
            # Connected components
            # ------------------------------------------------

            nutrition_components, _, nutrition_stats, _ = (
                cv2.connectedComponentsWithStats(
                    nutrition_mask,
                    connectivity=8
                )
            )

            weed_components, _, weed_stats, _ = (
                cv2.connectedComponentsWithStats(
                    weed_mask,
                    connectivity=8
                )
            )

            nutrition_component_areas = (
                nutrition_stats[1:, cv2.CC_STAT_AREA]
                if nutrition_components > 1
                else np.array([], dtype=np.int32)
            )

            weed_component_areas = (
                weed_stats[1:, cv2.CC_STAT_AREA]
                if weed_components > 1
                else np.array([], dtype=np.int32)
            )

            nutrition_component_count = len(
                nutrition_component_areas
            )

            weed_component_count = len(
                weed_component_areas
            )

            nutrition_regions += nutrition_component_count
            weed_regions += weed_component_count

            nutrition_one_pixel = int(
                np.sum(nutrition_component_areas == 1)
            )

            weed_one_pixel = int(
                np.sum(weed_component_areas == 1)
            )

            nutrition_one_pixel_components += nutrition_one_pixel
            weed_one_pixel_components += weed_one_pixel

            # ------------------------------------------------
            # OpenCV external contours
            # ------------------------------------------------

            nutrition_contours, _ = cv2.findContours(
                nutrition_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            weed_contours, _ = cv2.findContours(
                weed_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            nutrition_contour_areas = [
                cv2.contourArea(c)
                for c in nutrition_contours
            ]

            weed_contour_areas = [
                cv2.contourArea(c)
                for c in weed_contours
            ]

            nutrition_zero = sum(
                1 for a in nutrition_contour_areas
                if a <= 0
            )

            weed_zero = sum(
                1 for a in weed_contour_areas
                if a <= 0
            )

            nutrition_zero_area_contours += nutrition_zero
            weed_zero_area_contours += weed_zero

            # ------------------------------------------------
            # Record only images with potentially problematic
            # very small components.
            # ------------------------------------------------

            if (
                nutrition_one_pixel > 0
                or weed_one_pixel > 0
                or nutrition_zero > 0
                or weed_zero > 0
            ):

                all_rows.append({
                    "split": split,
                    "image": image_path.name,
                    "nutrition_pixels": nutrition_pixels,
                    "weed_pixels": weed_pixels,
                    "nutrition_regions": nutrition_component_count,
                    "weed_regions": weed_component_count,
                    "nutrition_one_pixel_components":
                        nutrition_one_pixel,
                    "weed_one_pixel_components":
                        weed_one_pixel,
                    "nutrition_zero_area_contours":
                        nutrition_zero,
                    "weed_zero_area_contours":
                        weed_zero,
                })

            if index % 250 == 0:
                print(f"Processed {index}/{len(image_files)}")


        except Exception as exc:

            errors += 1

            print(
                "ERROR:",
                image_path.name,
                "|",
                repr(exc)
            )


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

fieldnames = [
    "split",
    "image",
    "nutrition_pixels",
    "weed_pixels",
    "nutrition_regions",
    "weed_regions",
    "nutrition_one_pixel_components",
    "weed_one_pixel_components",
    "nutrition_zero_area_contours",
    "weed_zero_area_contours",
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_rows)


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 7B-2 - TINY REGION ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total images scanned: {total_images}\n"
    )

    f.write(
        f"Total masks found: {total_masks}\n"
    )

    f.write(
        f"Errors: {errors}\n\n"
    )

    f.write(
        f"Images with Nutrition pixels: "
        f"{nutrition_images}\n"
    )

    f.write(
        f"Images with Weed pixels: "
        f"{weed_images}\n\n"
    )

    f.write(
        f"Nutrition connected regions: "
        f"{nutrition_regions}\n"
    )

    f.write(
        f"Weed connected regions: "
        f"{weed_regions}\n\n"
    )

    f.write(
        f"Nutrition 1-pixel components: "
        f"{nutrition_one_pixel_components}\n"
    )

    f.write(
        f"Weed 1-pixel components: "
        f"{weed_one_pixel_components}\n\n"
    )

    f.write(
        f"Nutrition zero-area contours: "
        f"{nutrition_zero_area_contours}\n"
    )

    f.write(
        f"Weed zero-area contours: "
        f"{weed_zero_area_contours}\n\n"
    )

    f.write(
        f"Potentially problematic images recorded: "
        f"{len(all_rows)}\n"
    )

    f.write(
        f"\nCSV: {OUTPUT_CSV}\n"
    )


# ------------------------------------------------------------
# Final terminal output
# ------------------------------------------------------------

print()
print("=" * 70)
print("STEP 7B-2 COMPLETE")
print("=" * 70)

print("Total images scanned :", total_images)
print("Total masks found    :", total_masks)
print("Errors               :", errors)

print()
print("Nutrition images     :", nutrition_images)
print("Weed images          :", weed_images)

print()
print("Nutrition regions    :", nutrition_regions)
print("Weed regions         :", weed_regions)

print()
print(
    "Nutrition 1-pixel components :",
    nutrition_one_pixel_components
)

print(
    "Weed 1-pixel components      :",
    weed_one_pixel_components
)

print()
print(
    "Nutrition zero-area contours :",
    nutrition_zero_area_contours
)

print(
    "Weed zero-area contours      :",
    weed_zero_area_contours
)

print()
print(
    "Potentially problematic images:",
    len(all_rows)
)

print()
print("CSV:")
print(OUTPUT_CSV)

print()
print("Report:")
print(OUTPUT_REPORT)

print()
print("No source files were modified.")
print("=" * 70)