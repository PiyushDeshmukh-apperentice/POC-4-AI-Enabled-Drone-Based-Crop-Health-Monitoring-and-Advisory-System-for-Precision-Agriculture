import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# 1. Configuration & Constants
MODEL_PATH = r"d:\Projects\Plant Leaf Disease\Evals\efficientnet_model_PF_4500_B2.keras"
IMAGE_PATH = r"d:\Projects\Plant Leaf Disease\WeedyRice-RGBMS-DB\RGB\DJI_DateTime_2024_06_02_13_42_0036_lat_10.3040274_lon_105.2619303_alt_19.998m.JPG"  # Path to your 1280x960 image
PATCH_SIZE = (260, 260)
STRIDE = 112  # 50% overlap

# Decision Thresholds for Color Coding
# Assuming Class 1 = Unhealthy Probability
HEALTHY_THRESHOLD = 0.35  # Below this -> Green (Healthy)
UNHEALTHY_THRESHOLD = 0.65  # Above this -> Red (Unhealthy)
# In-between [0.35, 0.65] -> Yellow (Uncertain)

BATCH_SIZE = 32

# 2. Load Model & Image
model = tf.keras.models.load_model(MODEL_PATH)
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image at {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = image.shape

# 3. Extract Patches using Sliding Window
patches = []
coordinates = []

for y in range(0, img_h - PATCH_SIZE[1] + 1, STRIDE):
    for x in range(0, img_w - PATCH_SIZE[0] + 1, STRIDE):
        patch = image_rgb[y : y + PATCH_SIZE[1], x : x + PATCH_SIZE[0]]
        patches.append(patch)
        coordinates.append((x, y))

patches_array = np.array(patches, dtype=np.float32)
patches_preprocessed = preprocess_input(patches_array)

# 4. Batch Inference
predictions = model.predict(patches_preprocessed, batch_size=BATCH_SIZE)

# Get probability score representing "unhealthy" severity
if predictions.shape[-1] == 1:
    unhealthy_probs = predictions.flatten()
else:
    unhealthy_probs = predictions[:, 1]

# 5. Color-Coded Visualization
annotated_img = image.copy()
probability_map = np.zeros((img_h, img_w), dtype=np.float32)
counts_map = np.zeros((img_h, img_w), dtype=np.float32)

for (x, y), prob in zip(coordinates, unhealthy_probs):
    # Accumulate probabilities for heatmaps
    probability_map[y : y + PATCH_SIZE[1], x : x + PATCH_SIZE[0]] += prob
    counts_map[y : y + PATCH_SIZE[1], x : x + PATCH_SIZE[0]] += 1.0

    # Determine BGR Color based on score
    if prob >= UNHEALTHY_THRESHOLD:
        color = (0, 0, 255)  # RED (BGR format) in OpenCV
        label = f"Unhealthy {prob:.2f}"
    elif prob <= HEALTHY_THRESHOLD:
        color = (0, 255, 0)  # GREEN
        label = f"Healthy {1.0 - prob:.2f}"
    else:
        color = (0, 255, 255)  # YELLOW
        label = f"In-between {prob:.2f}"

    # Draw box and label for each patch
    cv2.rectangle(
        annotated_img,
        (x, y),
        (x + PATCH_SIZE[0], y + PATCH_SIZE[1]),
        color,
        2,
    )
    cv2.putText(
        annotated_img,
        label,
        (x + 5, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
    )

# 6. Generate Smooth Green-Yellow-Red Heatmap Overlay
counts_map[counts_map == 0] = 1.0
avg_prob_map = probability_map / counts_map

# Scale average probabilities to 0-255 uint8 range
heatmap_gray = (avg_prob_map * 255).astype(np.uint8)

# Apply COLORMAP_JET (0=Blue/Green region, 128=Yellow, 255=Red)
heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

# Blend heatmap onto original image
blended_heatmap = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

# 7. Save Results
cv2.imwrite("colored_patches.jpg", annotated_img)
cv2.imwrite("colored_heatmap.jpg", blended_heatmap)

print("Classification complete!")
print("Saved outputs: 'colored_patches.jpg' and 'colored_heatmap.jpg'")