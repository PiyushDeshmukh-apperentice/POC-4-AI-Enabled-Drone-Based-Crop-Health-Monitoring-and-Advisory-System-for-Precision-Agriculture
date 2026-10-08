import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ---------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------
MODEL_PATH = r"D:\Projects\Plant Leaf Disease\Evals\efficientnet_model_PF_4500_B2.keras"
IMAGE_PATH = r"D:\Projects\Plant Leaf Disease\WeedyRice-RGBMS-DB\RGB\DJI_DateTime_2024_06_02_13_43_0038_lat_10.3039632_lon_105.2619268_alt_20.009m.JPG"

# Automatically detect model input shape if possible
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    input_shape = model.input_shape
    PATCH_H, PATCH_W = input_shape[1], input_shape[2]
    if PATCH_H is None or PATCH_W is None:
        PATCH_H, PATCH_W = 260, 260
except Exception:
    PATCH_H, PATCH_W = 260, 260

PATCH_SIZE = (PATCH_W, PATCH_H)
STRIDE = PATCH_W // 2  # 50% overlap for smooth blending

# Set threshold boundaries
# Class mapping: prob represents 'Unhealthy' score
HEALTHY_THRESH = 0.35    # Below 0.35 -> Green (Healthy)
UNHEALTHY_THRESH = 0.65  # Above 0.65 -> Red (Unhealthy)
                         # Between 0.35 and 0.65 -> Yellow (At Risk)

BATCH_SIZE = 32

# ---------------------------------------------------------
# 2. Image Loading & Patch Extraction
# ---------------------------------------------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}")

img_h, img_w, _ = image.shape
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

patches = []
coordinates = []

for y in range(0, img_h - PATCH_SIZE[1] + 1, STRIDE):
    for x in range(0, img_w - PATCH_SIZE[0] + 1, STRIDE):
        patch = image_rgb[y : y + PATCH_SIZE[1], x : x + PATCH_SIZE[0]]
        patches.append(patch)
        coordinates.append((x, y))

patches_array = np.array(patches, dtype=np.float32)
patches_preprocessed = preprocess_input(patches_array)

# ---------------------------------------------------------
# 3. Model Inference
# ---------------------------------------------------------
predictions = model.predict(patches_preprocessed, batch_size=BATCH_SIZE)

if predictions.shape[-1] == 1:
    unhealthy_probs = predictions.flatten()
else:
    # If index 1 is Unhealthy; swap to predictions[:, 0] if index 0 is Unhealthy
    unhealthy_probs = predictions[:, 1]

# ---------------------------------------------------------
# 4. Generate Green-Yellow-Red Overlay (No Blue Background)
# ---------------------------------------------------------
overlay = image.copy()
heat_mask = np.zeros((img_h, img_w, 3), dtype=np.float32)
counts_map = np.zeros((img_h, img_w), dtype=np.float32)

records = []

for idx, ((x, y), prob) in enumerate(zip(coordinates, unhealthy_probs)):
    # Categorize patch
    if prob >= UNHEALTHY_THRESH:
        status = "Unhealthy"
        color_bgr = (0, 0, 255)     # Red
    elif prob <= HEALTHY_THRESH:
        status = "Healthy"
        color_bgr = (0, 255, 0)     # Green
    else:
        status = "At-Risk / Moderate"
        color_bgr = (0, 255, 255)   # Yellow

    # Accumulate colors for smooth spatial overlay
    heat_mask[y : y + PATCH_SIZE[1], x : x + PATCH_SIZE[0]] += np.array(color_bgr, dtype=np.float32)
    counts_map[y : y + PATCH_SIZE[1], x : x + PATCH_SIZE[0]] += 1.0

    # Store structured patch evaluations
    records.append({
        "Patch_ID": idx + 1,
        "X_Min": x,
        "Y_Min": y,
        "X_Max": x + PATCH_SIZE[0],
        "Y_Max": y + PATCH_SIZE[1],
        "Unhealthy_Probability": round(float(prob), 4),
        "Health_Status": status
    })

# Average overlapping patch colors
valid_pixels = counts_map > 0
for c in range(3):
    heat_mask[:, :, c][valid_pixels] /= counts_map[valid_pixels]

heat_mask = heat_mask.astype(np.uint8)

# Blend heatmap onto original image only where patches were scanned
alpha = 0.4
blended_image = image.copy()
blended_image[valid_pixels] = cv2.addWeighted(
    image[valid_pixels], 1.0 - alpha, heat_mask[valid_pixels], alpha, 0
)

# ---------------------------------------------------------
# 5. Agro-Department & Farmer Evaluation Metrics
# ---------------------------------------------------------
df_results = pd.DataFrame(records)
total_patches = len(df_results)

healthy_count = len(df_results[df_results["Health_Status"] == "Healthy"])
at_risk_count = len(df_results[df_results["Health_Status"] == "At-Risk / Moderate"])
unhealthy_count = len(df_results[df_results["Health_Status"] == "Unhealthy"])

healthy_pct = (healthy_count / total_patches) * 100
at_risk_pct = (at_risk_count / total_patches) * 100
unhealthy_pct = (unhealthy_count / total_patches) * 100

# Overall Field Infestation Severity Index (0 to 100%)
severity_index = df_results["Unhealthy_Probability"].mean() * 100

# Action Recommendation Logic
if severity_index >= 50 or unhealthy_pct >= 30:
    action_level = "HIGH PRIORITY: Widespread infection detected. Immediate targeted spraying required."
elif severity_index >= 20 or at_risk_pct >= 30:
    action_level = "MODERATE PRIORITY: Localized risk detected. Inspect highlighted yellow/red sectors."
else:
    action_level = "LOW PRIORITY: Field condition optimal. Routine monitoring recommended."

# Draw Executive Summary Card on Output Image
summary_box = np.zeros((180, 520, 3), dtype=np.uint8)
cv2.putText(summary_box, "CROP HEALTH FIELD EVALUATION", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.putText(summary_box, f"Healthy Field Area:   {healthy_pct:.1f}%", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(summary_box, f"At-Risk Area:         {at_risk_pct:.1f}%", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
cv2.putText(summary_box, f"Unhealthy Area:       {unhealthy_pct:.1f}%", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(summary_box, f"Infestation Severity: {severity_index:.1f}%", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(summary_box, f"Status: {action_level[:35]}...", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

# Overlay summary card onto top-left corner
blended_image[10:190, 10:530] = cv2.addWeighted(blended_image[10:190, 10:530], 0.2, summary_box, 0.8, 0)

# Save Outputs
cv2.imwrite("field_crop_health_map.jpg", blended_image)
df_results.to_csv("patch_evaluation_report.csv", index=False)

# ---------------------------------------------------------
# 6. Print Executive Report to Console
# ---------------------------------------------------------
print("=" * 60)
print("             AGRO-DEPARTMENT CROP EVALUATION REPORT          ")
print("=" * 60)
print(f"Total Sectors Analyzed   : {total_patches}")
print(f"Healthy Sector Coverage  : {healthy_pct:.2f}% ({healthy_count} patches)")
print(f"At-Risk Sector Coverage  : {at_risk_pct:.2f}% ({at_risk_count} patches)")
print(f"Unhealthy Sector Coverage: {unhealthy_pct:.2f}% ({unhealthy_count} patches)")
print(f"Field Infestation Index  : {severity_index:.2f} / 100")
print("-" * 60)
print(f"ACTION REQUIRED: {action_level}")
print("=" * 60)
print("Outputs saved successfully:")
print(" - Visual Overlay : 'field_crop_health_map.jpg'")
print(" - Detailed CSV    : 'patch_evaluation_report.csv'")