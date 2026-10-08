import os
import cv2
import numpy as np
import rasterio
from pathlib import Path


# ============================================================
# 1. CHANGE THESE PATHS
# ============================================================

MULTISPECTRAL_DIR = r"D:\Projects\Plant Leaf Disease\WeedyRice-RGBMS-DB\Multispectral"
MASK_DIR = r"D:\Projects\Plant Leaf Disease\WeedyRice-RGBMS-DB\Masks"

OUTPUT_DIR = r"D:\Projects\Plant Leaf Disease\WeedyRice_Uneven"


# ============================================================
# 2. SETTINGS
# ============================================================

HT_SIZE = 512
WD_SIZE = 512

# White pixels in mask = unhealthy / weed
MASK_THRESHOLD = 127

# Minimum weed percentage for unhealthy crop
WEED_THRESHOLD = 0.10

# Maximum weed percentage for healthy crop
HEALTHY_THRESHOLD = 0.05

# Sliding window step
STEP = 128


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

healthy_r_dir = Path(OUTPUT_DIR) / "healthy" / "R"
healthy_g_dir = Path(OUTPUT_DIR) / "healthy" / "G"
healthy_nir_dir = Path(OUTPUT_DIR) / "healthy" / "NIR"

unhealthy_r_dir = Path(OUTPUT_DIR) / "unhealthy" / "R"
unhealthy_g_dir = Path(OUTPUT_DIR) / "unhealthy" / "G"
unhealthy_nir_dir = Path(OUTPUT_DIR) / "unhealthy" / "NIR"


for directory in [
    healthy_r_dir,
    healthy_g_dir,
    healthy_nir_dir,
    unhealthy_r_dir,
    unhealthy_g_dir,
    unhealthy_nir_dir
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. FIND BEST CROP FROM MASK
# ============================================================

def find_best_crop(mask, target_class):

    h, w = mask.shape

    # Resize mask if smaller than 512x512
    if h < HT_SIZE or w < WD_SIZE:

        scale = max(
            HT_SIZE / h,
            WD_SIZE / w
        )

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

            if target_class == "unhealthy":

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
# 5. EXTRACT 512x512 CROP FROM TIFF
# ============================================================

def extract_crop(dataset, x, y):

    image = dataset.read(1)

    h, w = image.shape

    # Pad if image is smaller than 512x512
    if h < HT_SIZE or w < WD_SIZE:

        pad_h = max(0, HT_SIZE - h)
        pad_w = max(0, WD_SIZE - w)

        image = np.pad(
            image,
            (
                (0, pad_h),
                (0, pad_w)
            ),
            mode="reflect"
        )

    return image[
        y:y + HT_SIZE,
        x:x + WD_SIZE
    ]


# ============================================================
# 6. SAVE TIFF CROP
# ============================================================

def save_tiff_crop(
    crop,
    source_dataset,
    output_path
):

    profile = source_dataset.profile.copy()

    profile.update(
        height=HT_SIZE,
        width=WD_SIZE,
        count=1
    )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(
            crop,
            1
        )


# ============================================================
# 7. FIND MULTISPECTRAL FILES
# ============================================================

def find_band_files(base_name):

    files = {
        "R": None,
        "G": None,
        "NIR": None
    }

    for suffix in ["_R", "_G", "_NIR"]:

        possible_files = [
            Path(MULTISPECTRAL_DIR) /
            f"{base_name}{suffix}.tif",

            Path(MULTISPECTRAL_DIR) /
            f"{base_name}{suffix}.tiff"
        ]

        for file in possible_files:

            if file.exists():

                band = suffix.replace("_", "")

                files[band] = file
                break

    return files


# ============================================================
# 8. PROCESS ONE MASK
# ============================================================

def process_mask(mask_path):

    mask_name = mask_path.stem

    print("\n" + "=" * 60)
    print(f"Processing: {mask_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Find R, G and NIR
    # --------------------------------------------------------

    band_files = find_band_files(mask_name)

    if band_files["R"] is None:
        print("[WARNING] R band not found")
        return

    if band_files["G"] is None:
        print("[WARNING] G band not found")
        return

    if band_files["NIR"] is None:
        print("[WARNING] NIR band not found")
        return

    print(f"R   : {band_files['R'].name}")
    print(f"G   : {band_files['G'].name}")
    print(f"NIR : {band_files['NIR'].name}")

    # --------------------------------------------------------
    # Read mask
    # --------------------------------------------------------

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        print("[WARNING] Could not read mask")
        return

    # --------------------------------------------------------
    # Open multispectral TIFFs
    # --------------------------------------------------------

    r_dataset = rasterio.open(
        band_files["R"]
    )

    g_dataset = rasterio.open(
        band_files["G"]
    )

    nir_dataset = rasterio.open(
        band_files["NIR"]
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    r_shape = (
        r_dataset.height,
        r_dataset.width
    )

    g_shape = (
        g_dataset.height,
        g_dataset.width
    )

    nir_shape = (
        nir_dataset.height,
        nir_dataset.width
    )

    if not (
        r_shape == g_shape == nir_shape
    ):
        print("[WARNING] R/G/NIR dimensions do not match")

        r_dataset.close()
        g_dataset.close()
        nir_dataset.close()

        return

    # --------------------------------------------------------
    # Make mask same size as multispectral image
    # --------------------------------------------------------

    image_height = r_dataset.height
    image_width = r_dataset.width

    if mask.shape != (
        image_height,
        image_width
    ):

        mask = cv2.resize(
            mask,
            (
                image_width,
                image_height
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # --------------------------------------------------------
    # Calculate total unhealthy percentage
    # --------------------------------------------------------

    binary_mask = (
        mask > MASK_THRESHOLD
    )

    total_unhealthy_ratio = np.mean(
        binary_mask
    )

    print(
        f"Total unhealthy area: "
        f"{total_unhealthy_ratio * 100:.2f}%"
    )

    # ========================================================
    # CASE 1: NO SIGNIFICANT UNHEALTHY AREA
    # ========================================================

    if total_unhealthy_ratio < WEED_THRESHOLD:

        healthy_pos, healthy_score, processed_mask = \
            find_best_crop(
                mask,
                "healthy"
            )

        x, y = healthy_pos

        # ----------------------------------------------------
        # First healthy crop
        # ----------------------------------------------------

        healthy_r = extract_crop(
            r_dataset,
            x,
            y
        )

        healthy_g = extract_crop(
            g_dataset,
            x,
            y
        )

        healthy_nir = extract_crop(
            nir_dataset,
            x,
            y
        )

        # ----------------------------------------------------
        # Find second healthy crop
        # ----------------------------------------------------

        h, w = processed_mask.shape

        candidates = []

        for yy in range(
            0,
            h - HT_SIZE + 1,
            STEP
        ):

            for xx in range(
                0,
                w - WD_SIZE + 1,
                STEP
            ):

                crop_mask = processed_mask[
                    yy:yy + HT_SIZE,
                    xx:xx + WD_SIZE
                ]

                weed_ratio = np.mean(
                    crop_mask > MASK_THRESHOLD
                )

                if weed_ratio <= HEALTHY_THRESHOLD:

                    distance = (
                        (xx - x) ** 2 +
                        (yy - y) ** 2
                    )

                    candidates.append(
                        (
                            distance,
                            xx,
                            yy
                        )
                    )

        if candidates:

            candidates.sort(
                reverse=True
            )

            _, x2, y2 = candidates[0]

        else:

            x2 = max(
                0,
                w - WD_SIZE
            )

            y2 = max(
                0,
                h - HT_SIZE
            )

        # ----------------------------------------------------
        # Second healthy crop
        # ----------------------------------------------------

        healthy_r_2 = extract_crop(
            r_dataset,
            x2,
            y2
        )

        healthy_g_2 = extract_crop(
            g_dataset,
            x2,
            y2
        )

        healthy_nir_2 = extract_crop(
            nir_dataset,
            x2,
            y2
        )

        # ----------------------------------------------------
        # Save first healthy crop
        # ----------------------------------------------------

        save_tiff_crop(
            healthy_r,
            r_dataset,
            healthy_r_dir /
            f"{mask_name}_healthy_1_R.tif"
        )

        save_tiff_crop(
            healthy_g,
            g_dataset,
            healthy_g_dir /
            f"{mask_name}_healthy_1_G.tif"
        )

        save_tiff_crop(
            healthy_nir,
            nir_dataset,
            healthy_nir_dir /
            f"{mask_name}_healthy_1_NIR.tif"
        )

        # ----------------------------------------------------
        # Save second healthy crop
        # ----------------------------------------------------

        save_tiff_crop(
            healthy_r_2,
            r_dataset,
            healthy_r_dir /
            f"{mask_name}_healthy_2_R.tif"
        )

        save_tiff_crop(
            healthy_g_2,
            g_dataset,
            healthy_g_dir /
            f"{mask_name}_healthy_2_G.tif"
        )

        save_tiff_crop(
            healthy_nir_2,
            nir_dataset,
            healthy_nir_dir /
            f"{mask_name}_healthy_2_NIR.tif"
        )

        print("-> 2 healthy crops saved")

    # ========================================================
    # CASE 2: UNHEALTHY AREA IS PRESENT
    # ========================================================

    else:

        # ----------------------------------------------------
        # Find unhealthy crop
        # ----------------------------------------------------

        unhealthy_pos, unhealthy_score, processed_mask = \
            find_best_crop(
                mask,
                "unhealthy"
            )

        xu, yu = unhealthy_pos

        # ----------------------------------------------------
        # Find healthy crop
        # ----------------------------------------------------

        healthy_pos, healthy_score, _ = \
            find_best_crop(
                mask,
                "healthy"
            )

        xh, yh = healthy_pos

        # ----------------------------------------------------
        # Extract unhealthy R/G/NIR
        # ----------------------------------------------------

        unhealthy_r = extract_crop(
            r_dataset,
            xu,
            yu
        )

        unhealthy_g = extract_crop(
            g_dataset,
            xu,
            yu
        )

        unhealthy_nir = extract_crop(
            nir_dataset,
            xu,
            yu
        )

        # ----------------------------------------------------
        # Extract healthy R/G/NIR
        # ----------------------------------------------------

        healthy_r = extract_crop(
            r_dataset,
            xh,
            yh
        )

        healthy_g = extract_crop(
            g_dataset,
            xh,
            yh
        )

        healthy_nir = extract_crop(
            nir_dataset,
            xh,
            yh
        )

        # ----------------------------------------------------
        # Save unhealthy crop
        # ----------------------------------------------------

        save_tiff_crop(
            unhealthy_r,
            r_dataset,
            unhealthy_r_dir /
            f"{mask_name}_unhealthy_1_R.tif"
        )

        save_tiff_crop(
            unhealthy_g,
            g_dataset,
            unhealthy_g_dir /
            f"{mask_name}_unhealthy_1_G.tif"
        )

        save_tiff_crop(
            unhealthy_nir,
            nir_dataset,
            unhealthy_nir_dir /
            f"{mask_name}_unhealthy_1_NIR.tif"
        )

        # ----------------------------------------------------
        # Save healthy crop
        # ----------------------------------------------------

        save_tiff_crop(
            healthy_r,
            r_dataset,
            healthy_r_dir /
            f"{mask_name}_healthy_1_R.tif"
        )

        save_tiff_crop(
            healthy_g,
            g_dataset,
            healthy_g_dir /
            f"{mask_name}_healthy_1_G.tif"
        )

        save_tiff_crop(
            healthy_nir,
            nir_dataset,
            healthy_nir_dir /
            f"{mask_name}_healthy_1_NIR.tif"
        )

        print(
            f"-> Healthy crop "
            f"(unhealthy area = "
            f"{healthy_score * 100:.2f}%)"
        )

        print(
            f"-> Unhealthy crop "
            f"(unhealthy area = "
            f"{unhealthy_score * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # Close TIFF files
    # --------------------------------------------------------

    r_dataset.close()
    g_dataset.close()
    nir_dataset.close()


# ============================================================
# 9. FIND ALL MASKS
# ============================================================

mask_extensions = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
}

mask_files = [
    p
    for p in Path(MASK_DIR).iterdir()
    if p.suffix.lower() in mask_extensions
]


print("=" * 60)
print(f"Found {len(mask_files)} masks")
print("=" * 60)


# ============================================================
# 10. PROCESS ALL MASKS
# ============================================================

for mask_path in mask_files:

    try:

        process_mask(mask_path)

    except Exception as e:

        print(
            f"[ERROR] "
            f"{mask_path.name}: {e}"
        )


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"Healthy R:   {healthy_r_dir}"
)

print(
    f"Healthy G:   {healthy_g_dir}"
)

print(
    f"Healthy NIR: {healthy_nir_dir}"
)

print(
    f"Unhealthy R:   {unhealthy_r_dir}"
)

print(
    f"Unhealthy G:   {unhealthy_g_dir}"
)

print(
    f"Unhealthy NIR: {unhealthy_nir_dir}"
)