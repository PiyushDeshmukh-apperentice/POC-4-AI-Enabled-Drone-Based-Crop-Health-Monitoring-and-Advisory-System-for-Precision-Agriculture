from pathlib import Path
import cv2
import random

# ============================================================
# STEP 7B-6
# FINAL VISUAL VERIFICATION OF YOLOv8-SEG DATASET
# ============================================================

import os

DATA_ROOT = Path(os.environ.get(
    "CDAC_DATA_ROOT",
    Path.home() / "OneDrive" / "Desktop" / "CDAC"
)).expanduser()

YOLO_ROOT = DATA_ROOT / "YOLOv8" / "step7b4_yolo_dataset"
OUTPUT_ROOT = DATA_ROOT / "YOLOv8" / "step7b6_visual_verification"

CLASS_NAMES = {
    0: "Nutrition_Deficiency",
    1: "Weed"
}

# Number of examples to create from each split
SAMPLES_PER_SPLIT = 5

random.seed(42)

# ------------------------------------------------------------
# Read YOLO segmentation label
# ------------------------------------------------------------

def read_yolo_segmentation(label_path):
    polygons = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 7:
                continue

            class_id = int(parts[0])

            coordinates = list(map(float, parts[1:]))

            points = []

            for i in range(0, len(coordinates), 2):
                x = coordinates[i]
                y = coordinates[i + 1]
                points.append((x, y))

            polygons.append((class_id, points))

    return polygons


# ------------------------------------------------------------
# Draw polygons
# ------------------------------------------------------------

def draw_polygons(image, polygons):
    output = image.copy()

    for class_id, points in polygons:

        height, width = output.shape[:2]

        pixel_points = []

        for x, y in points:
            px = int(round(x * (width - 1)))
            py = int(round(y * (height - 1)))

            pixel_points.append([px, py])

        if len(pixel_points) < 3:
            continue

        pts = __import__("numpy").array(pixel_points, dtype="int32")

        # Draw polygon boundary
        cv2.polylines(
            output,
            [pts],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2
        )

        # Put class name near polygon
        x0, y0 = pts[0]

        name = CLASS_NAMES.get(class_id, f"Class_{class_id}")

        cv2.putText(
            output,
            name,
            (int(x0), int(max(20, y0))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


# ------------------------------------------------------------
# Process each split
# ------------------------------------------------------------

for split in ["train", "val", "test"]:

    image_dir = YOLO_ROOT / "images" / split
    label_dir = YOLO_ROOT / "labels" / split

    output_dir = OUTPUT_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.JPG")) +
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.PNG"))
    )

    if len(image_files) == 0:
        print(f"{split}: no images found")
        continue

    # Select deterministic random examples
    rng = random.Random(42 + len(split))

    if len(image_files) <= SAMPLES_PER_SPLIT:
        selected = image_files
    else:
        selected = rng.sample(image_files, SAMPLES_PER_SPLIT)

    print()
    print("-" * 70)
    print(f"VISUAL VERIFICATION: {split.upper()}")
    print("-" * 70)

    for image_path in selected:

        label_path = label_dir / f"{image_path.stem}.txt"

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"ERROR reading image: {image_path.name}")
            continue

        polygons = read_yolo_segmentation(label_path)

        result = draw_polygons(image, polygons)

        output_path = output_dir / image_path.name

        cv2.imwrite(str(output_path), result)

        nutrition_count = sum(
            1 for class_id, _ in polygons if class_id == 0
        )

        weed_count = sum(
            1 for class_id, _ in polygons if class_id == 1
        )

        print(
            f"{image_path.name} | "
            f"Nutrition polygons: {nutrition_count} | "
            f"Weed polygons: {weed_count}"
        )

print()
print("=" * 70)
print("STEP 7B-6 VISUAL VERIFICATION FILES CREATED")
print("=" * 70)
print()
print(f"Output folder:")
print(OUTPUT_ROOT)
print()
print("Created examples from train / val / test.")
print("Open the generated images and visually inspect the polygon boundaries.")
print()
print("STEP 7B-6 SCRIPT COMPLETED")
print("=" * 70)