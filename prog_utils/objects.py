
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path(__file__).resolve().parents[1] / "model_weights" / "rice_objects.keras"
OUTPUT_DIR = Path(__file__).parent / "sample_images" / "predictions"
MODEL_SIZE = (512, 512)
PREDICTION_THRESHOLD = 0.5


def load_objects_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def analyze_object_image(image, model):
    image = image.convert("RGB")
    original_array = np.asarray(image)

    resized = image.resize(MODEL_SIZE, Image.Resampling.BILINEAR)
    input_array = np.asarray(resized, dtype=np.float32) / 255.0
    input_array = np.expand_dims(input_array, axis=0)

    prediction = model.predict(input_array, verbose=0)
    if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[-1] != 1:
        raise ValueError("The U-Net model must return a single-channel segmentation mask.")

    mask = (prediction[0, :, :, 0] >= PREDICTION_THRESHOLD).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)
    mask_array = np.asarray(mask_image)

    foreground = mask_array > 0
    overlay_array = original_array.copy()
    red = np.array([255, 40, 40], dtype=np.float32)
    overlay_array[foreground] = (
        0.55 * overlay_array[foreground] + 0.45 * red
    ).astype(np.uint8)

    summary = {
        "image_width": image.width,
        "image_height": image.height,
        "foreground_pixels": int(foreground.sum()),
        "foreground_pct": float(foreground.mean() * 100),
    }
    return mask_image, Image.fromarray(overlay_array), summary


def predict_image(image_path, model, output_dir):
    image_path = Path(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = Image.open(image_path).convert("RGB")
    mask_image, overlay, summary = analyze_object_image(original, model)
    mask_array = np.asarray(mask_image)

    # Create side-by-side comparison
    comparison = Image.new(
        "RGB", (original.width * 3, original.height), "white"
    )
    comparison.paste(original, (0, 0))
    comparison.paste(Image.fromarray(
        np.repeat(mask_array[:, :, None], 3, axis=2)
    ), (original.width, 0))
    comparison.paste(overlay, (original.width * 2, 0))

    # Save outputs
    stem = image_path.stem
    Image.fromarray(mask_array).save(output_dir / f"{stem}_mask.png")
    overlay.save(output_dir / f"{stem}_overlay.png")
    comparison.save(output_dir / f"{stem}_comparison.png")

    print(f"Processed: {image_path.name}")
    print(f"  Predicted foreground: {summary['foreground_pct']:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an image or a folder of images"
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Folder for prediction outputs"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    model = load_objects_model()

    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )
    else:
        raise FileNotFoundError(input_path)

    if not images:
        raise ValueError(f"No supported images found in {input_path}")

    for image_path in images:
        predict_image(image_path, model, output_dir)


if __name__ == "__main__":
    main()
