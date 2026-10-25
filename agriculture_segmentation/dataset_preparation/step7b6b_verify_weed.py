from pathlib import Path
import cv2
import numpy as np

import os

DATA_ROOT = Path(os.environ.get(
    "CDAC_DATA_ROOT",
    Path.home() / "OneDrive" / "Desktop" / "CDAC"
)).expanduser()

YOLO_ROOT = DATA_ROOT / "YOLOv8" / "step7b4_yolo_dataset"
OUTPUT_ROOT = DATA_ROOT / "YOLOv8" / "step7b6b_weed_visual_verification"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def read_yolo_label(label_path):
    polygons = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 7:
                continue

            class_id = int(parts[0])

            # We only want Weed = class 1
            if class_id != 1:
                continue

            coords = list(map(float, parts[1:]))

            points = []

            for i in range(0, len(coords), 2):
                points.append((coords[i], coords[i + 1]))

            if len(points) >= 3:
                polygons.append(points)

    return polygons


# Search all splits for images containing Weed polygons
weed_images = []

for split in ["train", "val", "test"]:

    image_dir = YOLO_ROOT / "images" / split
    label_dir = YOLO_ROOT / "labels" / split

    for image_path in sorted(image_dir.glob("*.jpg")):

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        polygons = read_yolo_label(label_path)

        if len(polygons) > 0:
            weed_images.append(
                (split, image_path, polygons)
            )


print("=" * 70)
print("STEP 7B-6B - TARGETED WEED VISUAL VERIFICATION")
print("=" * 70)

print()
print(f"Total images containing Weed polygons: {len(weed_images)}")
print()

# Select one from each split
selected = []

for split in ["train", "val", "test"]:

    candidates = [
        item for item in weed_images
        if item[0] == split
    ]

    if candidates:
        selected.append(candidates[0])


for split, image_path, polygons in selected:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"ERROR reading: {image_path.name}")
        continue

    height, width = image.shape[:2]

    for polygon in polygons:

        points = []

        for x, y in polygon:

            px = int(round(x * (width - 1)))
            py = int(round(y * (height - 1)))

            points.append([px, py])

        pts = np.array(points, dtype=np.int32)

        cv2.polylines(
            image,
            [pts],
            True,
            (0, 255, 255),
            2
        )

        x0, y0 = pts[0]

        cv2.putText(
            image,
            "Weed",
            (int(x0), int(max(20, y0))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    output_path = OUTPUT_ROOT / f"{split}_{image_path.name}"

    cv2.imwrite(str(output_path), image)

    print(
        f"{split.upper()} | "
        f"{image_path.name} | "
        f"Weed polygons: {len(polygons)}"
    )


print()
print("=" * 70)
print("WEED VISUAL VERIFICATION FILES CREATED")
print("=" * 70)
print()
print(OUTPUT_ROOT)
print()
print("Inspect the generated Weed images.")
print("=" * 70)