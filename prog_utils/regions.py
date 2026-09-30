from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = Path(__file__).resolve().parents[1] / "model_weights" / "rice_regions.keras"
HEALTHY_THRESH = 0.35
UNHEALTHY_THRESH = 0.65
BATCH_SIZE = 32
DEFAULT_PATCH_SIZE = 260


def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def analyze_rgb_image(image_rgb, model):
    """Return an RGB health overlay, patch report, and field summary."""
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("RGB input must have exactly three color channels.")

    img_h, img_w, _ = image_rgb.shape
    input_shape = model.input_shape
    patch_h = input_shape[1] or DEFAULT_PATCH_SIZE
    patch_w = input_shape[2] or DEFAULT_PATCH_SIZE
    if img_h < patch_h or img_w < patch_w:
        raise ValueError(
            f"RGB image must be at least {patch_w} x {patch_h} pixels for this model."
        )

    stride = max(1, patch_w // 2)
    x_positions = list(range(0, img_w - patch_w + 1, stride))
    y_positions = list(range(0, img_h - patch_h + 1, stride))
    if x_positions[-1] != img_w - patch_w:
        x_positions.append(img_w - patch_w)
    if y_positions[-1] != img_h - patch_h:
        y_positions.append(img_h - patch_h)

    coordinates = [(x, y) for y in y_positions for x in x_positions]
    patches = np.array([
        image_rgb[y:y + patch_h, x:x + patch_w]
        for x, y in coordinates
    ], dtype=np.float32)
    predictions = model.predict(preprocess_input(patches), batch_size=BATCH_SIZE, verbose=0)

    if predictions.shape[-1] == 1:
        unhealthy_probs = predictions.reshape(-1)
    else:
        unhealthy_probs = predictions[:, 1]

    heat_mask = np.zeros((img_h, img_w, 3), dtype=np.float32)
    counts_map = np.zeros((img_h, img_w), dtype=np.float32)
    records = []

    for patch_id, ((x, y), probability) in enumerate(zip(coordinates, unhealthy_probs), start=1):
        if probability >= UNHEALTHY_THRESH:
            status = "Unhealthy"
            color_rgb = (255, 0, 0)
        elif probability <= HEALTHY_THRESH:
            status = "Healthy"
            color_rgb = (0, 255, 0)
        else:
            status = "At-Risk / Moderate"
            color_rgb = (255, 255, 0)

        heat_mask[y:y + patch_h, x:x + patch_w] += color_rgb
        counts_map[y:y + patch_h, x:x + patch_w] += 1
        records.append({
            "Patch_ID": patch_id,
            "X_Min": x,
            "Y_Min": y,
            "X_Max": x + patch_w,
            "Y_Max": y + patch_h,
            "Unhealthy_Probability": round(float(probability), 4),
            "Health_Status": status,
        })

    valid_pixels = counts_map > 0
    heat_mask[valid_pixels] /= counts_map[valid_pixels, None]
    blended_image = image_rgb.copy()
    blended_image[valid_pixels] = (
        image_rgb[valid_pixels] * 0.6 + heat_mask[valid_pixels] * 0.4
    ).astype(np.uint8)

    results = pd.DataFrame(records)
    total_patches = len(results)
    healthy_count = int((results["Health_Status"] == "Healthy").sum())
    at_risk_count = int((results["Health_Status"] == "At-Risk / Moderate").sum())
    unhealthy_count = int((results["Health_Status"] == "Unhealthy").sum())
    healthy_pct = healthy_count / total_patches * 100
    at_risk_pct = at_risk_count / total_patches * 100
    unhealthy_pct = unhealthy_count / total_patches * 100
    severity_index = float(results["Unhealthy_Probability"].mean() * 100)

    if severity_index >= 50 or unhealthy_pct >= 30:
        action_level = "High priority: widespread infection detected."
    elif severity_index >= 20 or at_risk_pct >= 30:
        action_level = "Moderate priority: inspect highlighted yellow and red areas."
    else:
        action_level = "Low priority: routine monitoring recommended."

    summary = {
        "total_patches": total_patches,
        "healthy_count": healthy_count,
        "at_risk_count": at_risk_count,
        "unhealthy_count": unhealthy_count,
        "healthy_pct": healthy_pct,
        "at_risk_pct": at_risk_pct,
        "unhealthy_pct": unhealthy_pct,
        "severity_index": severity_index,
        "action_level": action_level,
    }
    return blended_image, results, summary