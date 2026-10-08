"""
Build synthetic heterogeneous test composites from real single-stage held-out
images, with exact per-region ground truth -- for testing whether resize
forces one label across a multi-parcel scene, and whether the tile model
correctly maps per-parcel stages.

Usage:
    python build_composite_heterogeneity_test.py \
        --seedling-dir "e:/paddy_test/holdout/seedling" \
        --vegetative-dir "e:/paddy_test/holdout/vegetative" \
        --flowering-dir "e:/paddy_test/holdout/flowering" \
        --mature-dir "e:/paddy_test/holdout/mature" \
        --n-composites 50 \
        --out-dir "e:/paddy_test/heterogeneity_test"
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

STAGES = ["seedling", "vegetative", "flowering", "mature"]


def load_candidates(stage_dirs):
    candidates = {}
    for stage, d in stage_dirs.items():
        files = sorted(
            list(Path(d).glob("*.jpg")) + list(Path(d).glob("*.JPG")) + list(Path(d).glob("*.png"))
        )
        if not files:
            raise ValueError(f"No images found in {d} for stage '{stage}'")
        candidates[stage] = files
    return candidates


def build_composite(quadrant_stages, candidates, canvas_w, canvas_h, bund_w, rng, mode="crop"):
    """
    quadrant_stages: e.g. {"top_left": "seedling", "top_right": "vegetative", ...}
    mode="crop" (default): take a native-resolution CENTER CROP of size
        (half_w, half_h) from each source image, so quadrants preserve true
        per-pixel GSD -- no downscale confound.
    mode="resize": old behavior, squishes the whole source image into the quadrant.
    """
    half_w = (canvas_w - bund_w) // 2
    half_h = (canvas_h - bund_w) // 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (120, 110, 95))  # bund/roadway color
    regions = []

    positions = {
        "top_left": (0, 0),
        "top_right": (half_w + bund_w, 0),
        "bottom_left": (0, half_h + bund_w),
        "bottom_right": (half_w + bund_w, half_h + bund_w),
    }

    for quad, stage in quadrant_stages.items():
        src_path = rng.choice(candidates[stage])
        img = Image.open(src_path).convert("RGB")

        if mode == "crop" and img.width >= half_w and img.height >= half_h:
            left = (img.width - half_w) // 2
            top = (img.height - half_h) // 2
            img = img.crop((left, top, left + half_w, top + half_h))
        else:
            if mode == "crop":
                print(f"WARNING: {src_path} ({img.size}) smaller than quadrant "
                      f"({half_w}x{half_h}) -- falling back to resize for this image.")
            img = img.resize((half_w, half_h), Image.Resampling.BICUBIC)

        x0, y0 = positions[quad]
        canvas.paste(img, (x0, y0))
        regions.append({
            "stage": stage,
            "source_image": str(src_path),
            "bbox": [x0, y0, x0 + half_w, y0 + half_h],
        })

    return canvas, regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedling-dir", required=True)
    ap.add_argument("--vegetative-dir", required=True)
    ap.add_argument("--flowering-dir", required=True)
    ap.add_argument("--mature-dir", required=True)
    ap.add_argument("--n-composites", type=int, default=50)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--canvas-width", type=int, default=5280)
    ap.add_argument("--canvas-height", type=int, default=3956)
    ap.add_argument("--bund-width", type=int, default=60,
                     help="Width in px of the neutral gap strip between quadrants")
    ap.add_argument("--mode", choices=["crop", "resize"], default="crop",
                     help="'crop' (default, recommended): native-resolution center crop per "
                          "quadrant, no downscale confound. 'resize': old behavior, squishes "
                          "the whole source image into the quadrant.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stage_dirs = {
        "seedling": args.seedling_dir,
        "vegetative": args.vegetative_dir,
        "flowering": args.flowering_dir,
        "mature": args.mature_dir,
    }
    candidates = load_candidates(stage_dirs)
    rng = random.Random(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i in range(args.n_composites):
        n_stages = rng.choice([2, 2, 3, 4])
        chosen_stages = rng.sample(STAGES, n_stages)
        quads = ["top_left", "top_right", "bottom_left", "bottom_right"]
        rng.shuffle(quads)
        quadrant_stages = {quad: chosen_stages[j % len(chosen_stages)] for j, quad in enumerate(quads)}

        composite, regions = build_composite(
            quadrant_stages, candidates, args.canvas_width, args.canvas_height,
            args.bund_width, rng, mode=args.mode
        )

        fname = f"composite_{i:03d}.jpg"
        composite.save(args.out_dir / fname, quality=92)
        manifest.append({
            "file": fname,
            "regions": regions,
            "bund_width": args.bund_width,
            "canvas_width": args.canvas_width,
            "canvas_height": args.canvas_height,
        })
        if (i + 1) % 10 == 0 or (i + 1) == args.n_composites:
            print(f"Generated {i + 1}/{args.n_composites} composites (mode={args.mode})...")

    with open(args.out_dir / "ground_truth_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBuilt {args.n_composites} composites in {args.out_dir}")
    print(f"Ground truth manifest: {args.out_dir / 'ground_truth_manifest.json'}")


if __name__ == "__main__":
    main()
