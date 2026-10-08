import os
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# 1. CHANGE THESE TWO PATHS
# ============================================================

IMAGE_DIR = r"D:\Projects\Plant Leaf Disease\WeedyRice-RGBMS-DB\Multispectral"
MASK_DIR  = r"D:\Projects\Plant Leaf Disease\WeedyRice-RGBMS-DB\Masks"

OUTPUT_DIR = r"D:\Projects\Plant Leaf Disease\WeedyRice_Uneven"


# ============================================================
# 2. SETTINGS
# ============================================================

# CROP_SIZE = 512
HT_SIZE = 512
WD_SIZE = 512

# White pixels in the mask are considered WEED.
# Example mask: white = weed, black = healthy.
MASK_THRESHOLD = 127

# Minimum percentage of weed pixels required for a crop
# to be considered a WEEDY crop.
WEED_THRESHOLD = 0.10       # 10%

# Maximum percentage of weed pixels allowed for a
# HEALTHY crop.
HEALTHY_THRESHOLD = 0.05    # 5%

# Sliding window step.
# Smaller = more possible crop locations but slower.
STEP = 128


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

healthy_dir = Path(OUTPUT_DIR) / "healthy-mps"
weedy_dir = Path(OUTPUT_DIR) / "weedy-mps"

healthy_dir.mkdir(parents=True, exist_ok=True)
weedy_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. FIND A 512x512 CROP WITH A DESIRED MASK RATIO
# ============================================================

def find_best_crop(mask, target_class):
    """
    Find the best 512x512 region from the mask.

    target_class:
        "healthy" -> region with minimum weed percentage
        "weedy"   -> region with maximum weed percentage
    """

    h, w = mask.shape

    # If image is smaller than 512, resize it first.
    if h < HT_SIZE or w < WD_SIZE:
        scale = max(HT_SIZE / h, WD_SIZE / w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        mask = cv2.resize(
            mask,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )

        h, w = mask.shape

    best_score = None
    best_position = None

    for y in range(0, h - HT_SIZE + 1, STEP):

        for x in range(0, w - WD_SIZE + 1, STEP):

            crop_mask = mask[
                y:y + HT_SIZE,
                x:x + WD_SIZE
            ]

            weed_ratio = np.mean(
                crop_mask > MASK_THRESHOLD
            )

            if target_class == "weedy":

                # Higher weed ratio = better
                score = weed_ratio

                if best_score is None or score > best_score:
                    best_score = score
                    best_position = (x, y)

            else:

                # Lower weed ratio = better
                score = weed_ratio

                if best_score is None or score < best_score:
                    best_score = score
                    best_position = (x, y)

    return best_position, best_score, mask


# ============================================================
# 5. GET CROP FROM RGB IMAGE
# ============================================================

def extract_crop(image, x, y):

    h, w = image.shape[:2]

    # If image is smaller than 512x512,
    # pad it with reflection.
    if h < HT_SIZE or w < WD_SIZE:

        pad_h = max(0, HT_SIZE - h)
        pad_w = max(0, WD_SIZE - w)

        image = cv2.copyMakeBorder(
            image,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_REFLECT
        )

    return image[
        y:y + HT_SIZE,
        x:x + WD_SIZE
    ]


# ============================================================
# 6. PROCESS ONE IMAGE
# ============================================================

def process_image(image_path):

    image_name = image_path.stem

    # --------------------------------------------------------
    # Find corresponding mask
    # --------------------------------------------------------

    possible_masks = [
        Path(MASK_DIR) / f"{image_name}.png",
        Path(MASK_DIR) / f"{image_name}.jpg",
        Path(MASK_DIR) / f"{image_name}.jpeg",
        Path(MASK_DIR) / f"{image_name}.tif",
        Path(MASK_DIR) / f"{image_name}.tiff",
    ]

    mask_path = None

    for p in possible_masks:
        if p.exists():
            mask_path = p
            break

    if mask_path is None:
        print(f"[WARNING] Mask not found: {image_name}")
        return

    # --------------------------------------------------------
    # Read image and mask
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return

    if mask is None:
        print(f"[WARNING] Could not read mask: {mask_path}")
        return

    # --------------------------------------------------------
    # Make image and mask same size
    # --------------------------------------------------------

    if image.shape[:2] != mask.shape[:2]:

        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # --------------------------------------------------------
    # Calculate total weed percentage
    # --------------------------------------------------------

    binary_mask = mask > MASK_THRESHOLD

    total_weed_ratio = np.mean(binary_mask)

    print(
        f"{image_name} | "
        f"Total weed area: {total_weed_ratio * 100:.2f}%"
    )

    # ========================================================
    # CASE 1: NO SIGNIFICANT WEEDS
    # ========================================================

    if total_weed_ratio < WEED_THRESHOLD:

        # Find best healthy crop
        healthy_pos, healthy_score, processed_mask = \
            find_best_crop(mask, "healthy")

        x, y = healthy_pos

        healthy_crop_1 = extract_crop(
            image,
            x,
            y
        )

        # ----------------------------------------------------
        # Find another healthy crop.
        # Start searching from another part of the image.
        # ----------------------------------------------------

        h, w = processed_mask.shape

        candidates = []

        for yy in range(0, h - HT_SIZE + 1, STEP):

            for xx in range(0, w - WD_SIZE + 1, STEP):

                crop_mask = processed_mask[
                    yy:yy + HT_SIZE,
                    xx:xx + WD_SIZE
                ]

                weed_ratio = np.mean(
                    crop_mask > MASK_THRESHOLD
                )

                if weed_ratio <= HEALTHY_THRESHOLD:

                    # Distance from first crop
                    distance = (
                        (xx - x) ** 2 +
                        (yy - y) ** 2
                    )

                    candidates.append(
                        (distance, xx, yy)
                    )

        if candidates:

            candidates.sort(reverse=True)

            _, x2, y2 = candidates[0]

            healthy_crop_2 = extract_crop(
                image,
                x2,
                y2
            )

        else:

            # Fallback: use another crop position
            x2 = max(0, w - WD_SIZE)
            y2 = max(0, h - HT_SIZE)

            healthy_crop_2 = extract_crop(
                image,
                x2,
                y2
            )

        # ----------------------------------------------------
        # Save two healthy crops
        # ----------------------------------------------------

        cv2.imwrite(
            str(
                healthy_dir /
                f"{image_name}_healthy_1.jpg"
            ),
            healthy_crop_1
        )

        cv2.imwrite(
            str(
                healthy_dir /
                f"{image_name}_healthy_2.jpg"
            ),
            healthy_crop_2
        )

        print("  -> 2 healthy crops")

    # ========================================================
    # CASE 2: WEEDS ARE PRESENT
    # ========================================================

    else:

        # ----------------------------------------------------
        # Find weedy crop
        # ----------------------------------------------------

        weedy_pos, weedy_score, processed_mask = \
            find_best_crop(mask, "weedy")

        xw, yw = weedy_pos

        weedy_crop = extract_crop(
            image,
            xw,
            yw
        )

        # ----------------------------------------------------
        # Find healthy crop
        # ----------------------------------------------------

        healthy_pos, healthy_score, _ = \
            find_best_crop(mask, "healthy")

        xh, yh = healthy_pos

        healthy_crop = extract_crop(
            image,
            xh,
            yh
        )

        # ----------------------------------------------------
        # Save crops
        # ----------------------------------------------------

        cv2.imwrite(
            str(
                weedy_dir /
                f"{image_name}_weedy_1.jpg"
            ),
            weedy_crop
        )

        cv2.imwrite(
            str(
                healthy_dir /
                f"{image_name}_healthy_1.jpg"
            ),
            healthy_crop
        )

        print(
            f"  -> healthy crop "
            f"(weed area = {healthy_score * 100:.2f}%)"
        )

        print(
            f"  -> weedy crop "
            f"(weed area = {weedy_score * 100:.2f}%)"
        )


# ============================================================
# 7. PROCESS ALL IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

image_files = [
    p for p in Path(IMAGE_DIR).iterdir()
    if p.suffix.lower() in image_extensions
]

print("=" * 60)
print(f"Found {len(image_files)} RGB images")
print("=" * 60)

for image_path in image_files:

    try:
        process_image(image_path)

    except Exception as e:

        print(
            f"[ERROR] {image_path.name}: {e}"
        )


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"Healthy crops: {healthy_dir}")
print(f"Weedy crops:   {weedy_dir}")