from pathlib import Path
import csv

# ============================================================
# STEP 7B-5 - YOLOv8 DATASET INTEGRITY VERIFICATION
# ============================================================

import os

PROJECT_ROOT = Path(os.environ.get(
    "CDAC_DATA_ROOT",
    Path.home() / "OneDrive" / "Desktop" / "CDAC"
)).expanduser()

YOLO_DATASET = (
    PROJECT_ROOT
    / "YOLOv8"
    / "step7b4_yolo_dataset"
)

EXPECTED_COUNTS = {
    "train": 2987,
    "val": 374,
    "test": 373,
}

EXPECTED_USABLE_COUNTS = {
    "train": 2986,
    "val": 374,
    "test": 373,
}

EXPECTED_EXCLUDED_IMAGES = 1

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

VALID_CLASS_IDS = {
    0,  # Nutrition_Deficiency
    1,  # Weed
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 7B-5 - YOLOv8 DATASET INTEGRITY VERIFICATION")
print("=" * 70)

if not YOLO_DATASET.exists():
    print()
    print("ERROR: YOLO dataset not found:")
    print(YOLO_DATASET)
    raise SystemExit(1)

print()
print("Dataset:")
print(YOLO_DATASET)


# ============================================================
# CHECK data.yaml
# ============================================================

print()
print("-" * 70)
print("CHECK 1 - data.yaml")
print("-" * 70)

yaml_path = YOLO_DATASET / "data.yaml"

yaml_ok = True

if not yaml_path.exists():

    print("FAIL: data.yaml does not exist.")
    yaml_ok = False

else:

    yaml_text = yaml_path.read_text(
        encoding="utf-8"
    )

    print(yaml_text)

    required_strings = [
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "0: Nutrition_Deficiency",
        "1: Weed",
    ]

    for required in required_strings:

        if required not in yaml_text:

            print(
                "FAIL: Missing:",
                required
            )

            yaml_ok = False

    if yaml_ok:
        print("PASS: data.yaml structure is correct.")


# ============================================================
# GLOBAL COUNTERS
# ============================================================

total_images = 0
total_labels = 0
total_polygons = 0

nutrition_polygons = 0
weed_polygons = 0

malformed_labels = []
orphan_labels = []
missing_labels = []
invalid_class_rows = []
invalid_coordinates = []
too_few_points = []

duplicate_image_stems = []

split_summary = {}


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

for split in ["train", "val", "test"]:

    print()
    print("-" * 70)
    print(f"CHECKING {split.upper()}")
    print("-" * 70)

    image_dir = (
        YOLO_DATASET
        / "images"
        / split
    )

    label_dir = (
        YOLO_DATASET
        / "labels"
        / split
    )

    if not image_dir.exists():

        print("FAIL: Image directory missing:")
        print(image_dir)
        continue

    if not label_dir.exists():

        print("FAIL: Label directory missing:")
        print(label_dir)
        continue

    image_files = sorted(
        [
            p for p in image_dir.iterdir()
            if p.suffix.lower()
            in VALID_IMAGE_EXTENSIONS
        ]
    )

    label_files = sorted(
        [
            p for p in label_dir.iterdir()
            if p.suffix.lower() == ".txt"
        ]
    )

    print("Images:", len(image_files))
    print("Labels:", len(label_files))

    expected_source = EXPECTED_COUNTS[split]
    expected_usable = EXPECTED_USABLE_COUNTS[split]

    if len(image_files) == expected_usable:
        print(
            f"PASS: usable image count = "
            f"{expected_usable}"
        )
    else:
        print(
            f"FAIL: expected {expected_usable} "
            f"usable images, found {len(image_files)}"
        )

    if len(label_files) == expected_usable:
        print(
            f"PASS: label count = "
            f"{expected_usable}"
        )
    else:
        print(
            f"FAIL: expected {expected_usable} "
            f"labels, found {len(label_files)}"
        )

    # --------------------------------------------------------
    # Image stems
    # --------------------------------------------------------

    image_stems = {
        p.stem
        for p in image_files
    }

    label_stems = {
        p.stem
        for p in label_files
    }

    # Missing labels
    missing = sorted(
        image_stems - label_stems
    )

    # Orphan labels
    orphan = sorted(
        label_stems - image_stems
    )

    if missing:

        print(
            "FAIL: Images without labels:",
            len(missing)
        )

        for stem in missing:
            missing_labels.append(
                f"{split}/{stem}"
            )

    else:

        print("PASS: Every image has a label.")

    if orphan:

        print(
            "FAIL: Orphan labels:",
            len(orphan)
        )

        for stem in orphan:
            orphan_labels.append(
                f"{split}/{stem}"
            )

    else:

        print("PASS: No orphan labels.")

    # --------------------------------------------------------
    # Duplicate stems
    # --------------------------------------------------------

    seen_stems = set()

    for image_path in image_files:

        if image_path.stem in seen_stems:

            duplicate_image_stems.append(
                f"{split}/{image_path.name}"
            )

        seen_stems.add(image_path.stem)

    # --------------------------------------------------------
    # Validate every label file
    # --------------------------------------------------------

    split_polygons = 0
    split_nutrition = 0
    split_weed = 0

    for label_path in label_files:

        try:

            text = label_path.read_text(
                encoding="utf-8"
            )

        except Exception as exc:

            malformed_labels.append(
                (
                    f"{split}/{label_path.name}",
                    f"Could not read file: {exc}"
                )
            )

            continue

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if len(lines) == 0:

            malformed_labels.append(
                (
                    f"{split}/{label_path.name}",
                    "Empty label file"
                )
            )

            continue

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            parts = line.split()

            # YOLO segmentation:
            # class_id x1 y1 x2 y2 ... xn yn
            if len(parts) < 7:

                malformed_labels.append(
                    (
                        f"{split}/{label_path.name}",
                        f"Line {line_number}: "
                        f"fewer than 3 points"
                    )
                )

                continue

            if (len(parts) - 1) % 2 != 0:

                malformed_labels.append(
                    (
                        f"{split}/{label_path.name}",
                        f"Line {line_number}: "
                        f"odd number of coordinates"
                    )
                )

                continue

            try:

                class_id = int(parts[0])

            except ValueError:

                malformed_labels.append(
                    (
                        f"{split}/{label_path.name}",
                        f"Line {line_number}: "
                        f"invalid class ID"
                    )
                )

                continue

            # ------------------------------------------------
            # Class ID check
            # ------------------------------------------------

            if class_id not in VALID_CLASS_IDS:

                invalid_class_rows.append(
                    (
                        f"{split}/{label_path.name}",
                        line_number,
                        class_id
                    )
                )

            # ------------------------------------------------
            # Coordinate check
            # ------------------------------------------------

            try:

                coordinates = [
                    float(value)
                    for value in parts[1:]
                ]

            except ValueError:

                malformed_labels.append(
                    (
                        f"{split}/{label_path.name}",
                        f"Line {line_number}: "
                        f"non-numeric coordinate"
                    )
                )

                continue

            for coordinate in coordinates:

                if (
                    coordinate < 0.0
                    or coordinate > 1.0
                ):

                    invalid_coordinates.append(
                        (
                            f"{split}/{label_path.name}",
                            line_number,
                            coordinate
                        )
                    )

            # ------------------------------------------------
            # Minimum 3 polygon points
            # ------------------------------------------------

            number_of_points = (
                len(coordinates) // 2
            )

            if number_of_points < 3:

                too_few_points.append(
                    (
                        f"{split}/{label_path.name}",
                        line_number,
                        number_of_points
                    )
                )

            # ------------------------------------------------
            # Count polygons
            # ------------------------------------------------

            split_polygons += 1

            if class_id == 0:
                split_nutrition += 1

            elif class_id == 1:
                split_weed += 1

    total_images += len(image_files)
    total_labels += len(label_files)

    total_polygons += split_polygons
    nutrition_polygons += split_nutrition
    weed_polygons += split_weed

    split_summary[split] = {
        "source_expected": expected_source,
        "usable_expected": expected_usable,
        "images": len(image_files),
        "labels": len(label_files),
        "polygons": split_polygons,
        "nutrition": split_nutrition,
        "weed": split_weed,
    }

    print()
    print("Polygon count:", split_polygons)
    print(
        "Nutrition polygons:",
        split_nutrition
    )
    print(
        "Weed polygons:",
        split_weed
    )


# ============================================================
# CHECK EXCLUSION REPORT
# ============================================================

print()
print("-" * 70)
print("CHECK 2 - DOCUMENTED EXCLUSION")
print("-" * 70)

error_report = (
    YOLO_DATASET
    / "step7b4_conversion_errors.csv"
)

exclusion_ok = False

if not error_report.exists():

    print("FAIL: Conversion error report missing.")

else:

    with open(
        error_report,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)
        rows = list(reader)

    print(
        "Documented exclusions/errors:",
        len(rows)
    )

    if len(rows) == EXPECTED_EXCLUDED_IMAGES:

        print(
            "PASS: Exactly one exclusion is documented."
        )

        print()
        print("Excluded image:")

        for row in rows:

            print(
                f"  split = {row.get('split')}"
            )

            print(
                f"  image = {row.get('image')}"
            )

            print(
                f"  reason = {row.get('error')}"
            )

        exclusion_ok = True

    else:

        print(
            "FAIL: Expected exactly "
            f"{EXPECTED_EXCLUDED_IMAGES} "
            f"exclusion(s)."
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("STEP 7B-5 - FINAL INTEGRITY RESULTS")
print("=" * 70)

checks = []


def record_check(name, passed):

    checks.append(
        (name, passed)
    )

    status = "PASS" if passed else "FAIL"

    print(
        f"{status}: {name}"
    )


record_check(
    "data.yaml",
    yaml_ok
)

for split in ["train", "val", "test"]:

    record_check(
        f"{split} image count",
        split_summary.get(
            split,
            {}
        ).get("images")
        == EXPECTED_USABLE_COUNTS[split]
    )

    record_check(
        f"{split} label count",
        split_summary.get(
            split,
            {}
        ).get("labels")
        == EXPECTED_USABLE_COUNTS[split]
    )


record_check(
    "No missing labels",
    len(missing_labels) == 0
)

record_check(
    "No orphan labels",
    len(orphan_labels) == 0
)

record_check(
    "No malformed labels",
    len(malformed_labels) == 0
)

record_check(
    "Only valid class IDs 0 and 1",
    len(invalid_class_rows) == 0
)

record_check(
    "All coordinates in [0, 1]",
    len(invalid_coordinates) == 0
)

record_check(
    "All polygons have at least 3 points",
    len(too_few_points) == 0
)

record_check(
    "No duplicate image stems",
    len(duplicate_image_stems) == 0
)

record_check(
    "Exactly one documented exclusion",
    exclusion_ok
)


# ============================================================
# PRINT FINAL STATISTICS
# ============================================================

print()
print("-" * 70)
print("FINAL DATASET STATISTICS")
print("-" * 70)

print(
    "Total images :",
    total_images
)

print(
    "Total labels :",
    total_labels
)

print(
    "Total polygons :",
    total_polygons
)

print(
    "Nutrition polygons :",
    nutrition_polygons
)

print(
    "Weed polygons :",
    weed_polygons
)

print()
print("Expected:")
print("Images  : 3733")
print("Labels  : 3733")
print("Polygons: 10477")


# ============================================================
# SAVE VERIFICATION REPORT
# ============================================================

verification_report = (
    YOLO_DATASET
    / "step7b5_integrity_report.txt"
)

with open(
    verification_report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 7B-5 - YOLOv8 DATASET INTEGRITY REPORT\n"
    )

    f.write("=" * 60 + "\n\n")

    for split in ["train", "val", "test"]:

        data = split_summary[split]

        f.write(
            f"{split.upper()}\n"
        )

        f.write(
            f"Images: {data['images']}\n"
        )

        f.write(
            f"Labels: {data['labels']}\n"
        )

        f.write(
            f"Polygons: {data['polygons']}\n"
        )

        f.write(
            f"Nutrition polygons: "
            f"{data['nutrition']}\n"
        )

        f.write(
            f"Weed polygons: "
            f"{data['weed']}\n\n"
        )

    f.write(
        f"Total images: {total_images}\n"
    )

    f.write(
        f"Total labels: {total_labels}\n"
    )

    f.write(
        f"Total polygons: {total_polygons}\n"
    )

    f.write(
        f"Nutrition polygons: "
        f"{nutrition_polygons}\n"
    )

    f.write(
        f"Weed polygons: "
        f"{weed_polygons}\n\n"
    )

    f.write(
        f"Missing labels: "
        f"{len(missing_labels)}\n"
    )

    f.write(
        f"Orphan labels: "
        f"{len(orphan_labels)}\n"
    )

    f.write(
        f"Malformed labels: "
        f"{len(malformed_labels)}\n"
    )

    f.write(
        f"Invalid class rows: "
        f"{len(invalid_class_rows)}\n"
    )

    f.write(
        f"Invalid coordinates: "
        f"{len(invalid_coordinates)}\n"
    )

    f.write(
        f"Too few polygon points: "
        f"{len(too_few_points)}\n"
    )

    f.write(
        f"Duplicate image stems: "
        f"{len(duplicate_image_stems)}\n"
    )

    f.write(
        f"Documented exclusions: "
        f"{len(rows) if error_report.exists() else 0}\n"
    )

    f.write("\nCHECK RESULTS\n")

    for name, passed in checks:

        f.write(
            f"{'PASS' if passed else 'FAIL'}: "
            f"{name}\n"
        )


# ============================================================
# FINAL DECISION
# ============================================================

all_passed = all(
    passed
    for _, passed in checks
)

print()
print("=" * 70)

if all_passed:

    print(
        "STEP 7B-5 PASSED"
    )

    print()
    print(
        "The corrected YOLOv8 dataset is "
        "ready for the next verification stage."
    )

else:

    print(
        "STEP 7B-5 FAILED"
    )

    print()
    print(
        "Do NOT start YOLO training yet."
    )

print()
print("Verification report:")
print(verification_report)

print("=" * 70)