"""
prepare_not_rice_class.py -- Prepare 'not_rice' negative examples for a 5th classifier class,
from one or more source directories, with a train/held-out-test split done at the
SOURCE level when possible (not just the image level) -- so you can test
whether the model learned a genuine 'doesn't look like rice' concept, or
just memorized one negative dataset's visual signature.

Also flags, per image, whether it needed upsampling to reach your tile size
-- upsampled images carry a real risk of teaching a resolution shortcut
instead of a content distinction. Native-resolution sources are strongly preferred.

Usage:
    python prepare_not_rice_class.py \
        --source "e:/paddy_test/ood_test" --source-name resisc45_smoketest \
        --tile-size 1024 \
        --out-dir "e:/paddy_test/not_rice_class"
"""

import argparse
import json
import random
from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_images(d: Path):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS)


def process_image(src_path: Path, tile_size: int, out_path: Path):
    img = Image.open(src_path).convert("RGB")
    was_upsampled = img.width < tile_size or img.height < tile_size

    if was_upsampled:
        # cannot crop native res that isn't there -- resize up, flagged
        img = img.resize((tile_size, tile_size), Image.BICUBIC)
    else:
        # native-or-larger: CENTER CROP, never downscale-then-upscale
        left = (img.width - tile_size) // 2
        top = (img.height - tile_size) // 2
        img = img.crop((left, top, left + tile_size, top + tile_size))

    img.save(out_path, quality=92)
    return was_upsampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, dest="sources",
                    help="Path to a source image directory. Repeatable.")
    ap.add_argument("--source-name", action="append", required=True, dest="source_names",
                    help="Short name for each --source, same order/count.")
    ap.add_argument("--holdout-source", default=None,
                    help="If given, this whole named source is held out for test "
                         "(cross-source generalization check). Must match a --source-name.")
    ap.add_argument("--test-fraction", type=float, default=0.2,
                    help="Used only for image-level holdout when --holdout-source isn't set, "
                         "or as the within-source split for sources not held out entirely.")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if len(args.sources) != len(args.source_names):
        raise ValueError("--source and --source-name counts must match")

    rng = random.Random(args.seed)
    train_dir = args.out_dir / "train" / "not_rice"
    test_dir = args.out_dir / "test" / "not_rice"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_sources = len(args.sources)

    if args.holdout_source and args.holdout_source not in args.source_names:
        raise ValueError(f"--holdout-source '{args.holdout_source}' not among --source-name values")

    if n_sources == 1 and not args.holdout_source:
        print("NOTE: only one source given, no --holdout-source set -- falling back to "
              "image-level holdout within this source. This only tests whether the model "
              "generalizes to unseen IMAGES from the same dataset, not to a genuinely "
              "different negative source. Add a second --source for a stronger test.")

    idx = 0
    for src_dir, src_name in zip(args.sources, args.source_names):
        images = find_images(Path(src_dir))
        if not images:
            print(f"WARNING: no images found under {src_dir} for source '{src_name}'")
            continue
        rng.shuffle(images)

        this_source_is_holdout = (src_name == args.holdout_source)
        if this_source_is_holdout:
            split_assignment = ["test"] * len(images)
        else:
            n_test = int(len(images) * args.test_fraction) if not args.holdout_source else 0
            split_assignment = ["test"] * n_test + ["train"] * (len(images) - n_test)
            rng.shuffle(split_assignment)

        n_upsampled = 0
        for img_path, split in zip(images, split_assignment):
            out_dir = train_dir if split == "train" else test_dir
            out_name = f"{src_name}_{idx:05d}.jpg"
            out_path = out_dir / out_name
            was_upsampled = process_image(img_path, args.tile_size, out_path)
            n_upsampled += int(was_upsampled)
            manifest.append({
                "file": str(out_path.relative_to(args.out_dir)),
                "source_name": src_name,
                "source_image": str(img_path),
                "split": split,
                "upsampled": was_upsampled,
                "label": "not_rice",
            })
            idx += 1

        pct_up = 100 * n_upsampled / len(images)
        print(f"[{src_name}] {len(images)} images -> "
              f"{split_assignment.count('train')} train / {split_assignment.count('test')} test "
              f"({pct_up:.0f}% needed upsampling{' -- ALL, low-res source' if pct_up > 90 else ''})")

    with open(args.out_dir / "not_rice_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_train = sum(1 for m in manifest if m["split"] == "train")
    n_test = sum(1 for m in manifest if m["split"] == "test")
    n_up_train = sum(1 for m in manifest if m["split"] == "train" and m["upsampled"])
    print(f"\nTotal: {n_train} train, {n_test} test")
    if n_up_train:
        print(f"WARNING: {n_up_train}/{n_train} training images ({100*n_up_train/n_train:.0f}%) "
              f"needed upsampling -- if the model's not_rice detection later fails on real "
              f"native-resolution OOD frames, this is the first thing to suspect.")
    print(f"Manifest saved to: {args.out_dir / 'not_rice_manifest.json'}")


if __name__ == "__main__":
    main()
