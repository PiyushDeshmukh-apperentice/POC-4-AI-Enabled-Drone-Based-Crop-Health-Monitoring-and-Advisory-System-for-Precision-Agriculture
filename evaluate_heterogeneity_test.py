"""
Full heterogeneity evaluation with:
1. Resize model baseline (single-label, checks match rate + bias distribution)
2. Tile model spatial mapping (per-tile vs true quadrant ground truth)
3. Boundary confidence diagnostic with distribution overlap test
4. Resize bias check on composites where Vegetative is NOT a true stage
5. All four stage accuracies reported -- no selective omission

Usage:
    python evaluate_heterogeneity_test.py --composite-dir "e:/paddy_test/heterogeneity_test_crop"
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy import stats as scipy_stats

sys.path.append(str(Path("training").resolve()))
from predict import load_model

STAGES = ["seedling", "vegetative", "flowering", "mature"]
CLASSES = ["Seedling", "Vegetative", "Flowering", "Mature"]
IDX_TO_STAGE = {i: c.lower() for i, c in enumerate(CLASSES)}

TILE_SIZE = 1024
STRIDE = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resize_model():
    ckpt_path = Path("training/checkpoints/checkpoint_resize_best.pt")
    state = torch.load(ckpt_path, map_location=DEVICE)
    res = state.get("resolution", 384)
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(state["model"])
    model.to(DEVICE).eval()
    tf = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf


def predict_resize(model_tuple, pil_image):
    model, tf = model_tuple
    tensor = tf(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred_idx = probs.argmax().item()
    return IDX_TO_STAGE[pred_idx], float(probs[pred_idx])


def load_tile_model():
    ckpt_path = Path("training/checkpoints/checkpoint_enhanced_ann.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("training/checkpoints/checkpoint_round1.pt")
    model, input_size, arch = load_model(ckpt_path, DEVICE)
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf, arch


def predict_tile(model_tuple, pil_tile):
    model, tf, _ = model_tuple
    tensor = tf(pil_tile).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred_idx = probs.argmax().item()
    return IDX_TO_STAGE[pred_idx], float(probs[pred_idx])


def get_tile_bboxes(image: Image.Image):
    w, h = image.size
    x_starts = list(range(0, w - TILE_SIZE + 1, STRIDE))
    if x_starts[-1] + TILE_SIZE < w:
        x_starts.append(w - TILE_SIZE)
    y_starts = list(range(0, h - TILE_SIZE + 1, STRIDE))
    if y_starts[-1] + TILE_SIZE < h:
        y_starts.append(h - TILE_SIZE)
    return [(x, y, x + TILE_SIZE, y + TILE_SIZE) for y in y_starts for x in x_starts]


def tile_region_info(tile_bbox, regions, bund_w):
    """Returns (is_boundary: bool, dominant_stage: str, purity: float)
    where purity = fraction of tile area covered by a single true quadrant."""
    tx0, ty0, tx1, ty1 = tile_bbox
    max_area = 0
    dom_stage = "boundary"
    total_region_area = 0

    for r in regions:
        rx0, ry0, rx1, ry1 = r["bbox"]
        ix0, iy0 = max(tx0, rx0), max(ty0, ry0)
        ix1, iy1 = min(tx1, rx1), min(ty1, ry1)
        if ix1 > ix0 and iy1 > iy0:
            area = (ix1 - ix0) * (iy1 - iy0)
            total_region_area += area
            if area > max_area:
                max_area = area
                dom_stage = r["stage"]

    tile_area = (tx1 - tx0) * (ty1 - ty0)
    bund_area = tile_area - total_region_area
    is_boundary = bund_area > (0.05 * tile_area)  # >5% overlap with bund => boundary
    purity = max_area / tile_area

    return is_boundary, dom_stage, purity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composite-dir", required=True, type=Path)
    ap.add_argument("--purity-threshold", type=float, default=0.90,
                    help="Min fraction of tile area in one quadrant to be counted as 'pure interior'. Default 0.90.")
    args = ap.parse_args()

    with open(args.composite_dir / "ground_truth_manifest.json") as f:
        manifest = json.load(f)

    resize_model = load_resize_model()
    tile_model = load_tile_model()
    tile_arch = tile_model[2]
    print(f"Tile model architecture: {tile_arch}")
    print(f"Running on: {DEVICE}")
    print(f"Purity threshold for 'interior': {args.purity_threshold:.0%}")
    print(f"Composites to evaluate: {len(manifest)}\n")

    resize_results = []
    per_stage = {s: {"correct": 0, "total": 0} for s in STAGES}
    complexity_res = {2: [], 3: [], 4: []}

    pure_confidences = []
    boundary_confidences = []
    resize_preds_count = {s: 0 for s in STAGES}
    resize_preds_no_veg = {s: 0 for s in STAGES}  # only composites where veg not present
    n_no_veg = 0

    for i, entry in enumerate(manifest):
        img = Image.open(args.composite_dir / entry["file"]).convert("RGB")
        regions = entry["regions"]
        bund_w = entry.get("bund_width", 60)
        true_stages = {r["stage"] for r in regions}
        n_unique = len(true_stages)

        # --- resize ---
        pred_r, conf_r = predict_resize(resize_model, img)
        resize_preds_count[pred_r] += 1
        matched = pred_r in true_stages
        resize_results.append({
            "file": entry["file"],
            "true_stages": sorted(true_stages),
            "predicted": pred_r,
            "conf": conf_r,
            "matched": matched,
        })
        if "vegetative" not in true_stages:
            n_no_veg += 1
            resize_preds_no_veg[pred_r] += 1

        # --- tile ---
        comp_pure_c, comp_pure_t = 0, 0
        for bbox in get_tile_bboxes(img):
            tile_img = img.crop(bbox)
            pred_t, conf_t = predict_tile(tile_model, tile_img)
            is_bnd, dom_stage, purity = tile_region_info(bbox, regions, bund_w)

            if is_bnd or purity < args.purity_threshold:
                boundary_confidences.append(conf_t)
            else:
                pure_confidences.append(conf_t)
                per_stage[dom_stage]["total"] += 1
                comp_pure_t += 1
                if pred_t == dom_stage:
                    per_stage[dom_stage]["correct"] += 1
                    comp_pure_c += 1

        if comp_pure_t > 0:
            complexity_res[n_unique].append(comp_pure_c / comp_pure_t)

        if (i + 1) % 10 == 0 or (i + 1) == len(manifest):
            print(f"  Processed {i+1}/{len(manifest)} composites...")

    # ---- compile summary ----
    n = len(resize_results)
    match_rate = sum(r["matched"] for r in resize_results) / n
    total_pure = sum(d["total"] for d in per_stage.values())
    total_correct = sum(d["correct"] for d in per_stage.values())
    overall_tile_acc = total_correct / total_pure if total_pure else 0

    print("\n==================================================================")
    print(f"      HETEROGENEITY EVALUATION (N={n} scenes, CROP mode)          ")
    print("==================================================================")

    print(f"\n1. RESIZE MODEL (Full-Image 384px single-label output):")
    print(f"   Matched at least one true stage present : {match_rate:.1%}")
    print(f"   Default prediction distribution (all {n} composites):")
    for s in STAGES:
        c = resize_preds_count[s]
        print(f"     {s.capitalize():<12} : {c:2d} / {n}  ({c/n*100:5.1f}%)")
    print(f"\n   Vegetative bias check -- composites where 'vegetative' is NOT a true stage")
    print(f"   (n={n_no_veg}): resize model still predicted Vegetative in "
          f"{resize_preds_no_veg['vegetative']} / {n_no_veg} of them "
          f"({resize_preds_no_veg['vegetative']/n_no_veg*100:.1f}% false Vegetative rate)")

    print(f"\n2. TILE MODEL per-stage accuracy on pure interior tiles (purity >= {args.purity_threshold:.0%}):")
    print(f"   Overall (all stages) : {overall_tile_acc:.1%} ({total_correct}/{total_pure})")
    print(f"   Random chance baseline: 25.0%")
    for s in STAGES:
        d = per_stage[s]
        acc = d["correct"] / d["total"] * 100 if d["total"] else float("nan")
        print(f"   {s.capitalize():<12} : {d['correct']:4d} / {d['total']:4d}  ({acc:5.1f}%)")

    print(f"\n   Scene complexity breakdown:")
    for k in [2, 3, 4]:
        accs = complexity_res[k]
        if accs:
            print(f"   {k}-stage scenes (n={len(accs):2d}) : mean tile acc = {np.mean(accs)*100:.1f}%")

    # Boundary confidence: t-test + overlap
    pure_arr = np.array(pure_confidences)
    bnd_arr = np.array(boundary_confidences)
    t_stat, p_val = scipy_stats.ttest_ind(pure_arr, bnd_arr, equal_var=False)
    # Overlap: fraction of boundary tile confidences above the pure tile median
    pure_med = np.median(pure_arr)
    overlap_frac = np.mean(bnd_arr >= pure_med)

    print(f"\n3. BOUNDARY CONFIDENCE DIAGNOSTIC:")
    print(f"   Interior tiles  : n={len(pure_arr):5d}  mean={pure_arr.mean():.4f}  "
          f"std={pure_arr.std():.4f}  median={np.median(pure_arr):.4f}")
    print(f"   Boundary tiles  : n={len(bnd_arr):5d}  mean={bnd_arr.mean():.4f}  "
          f"std={bnd_arr.std():.4f}  median={np.median(bnd_arr):.4f}")
    print(f"   Mean confidence drop at boundaries : {pure_arr.mean() - bnd_arr.mean():+.4f}")
    print(f"   Welch t-test (unequal var)         : t={t_stat:.2f}, p={p_val:.2e}")
    print(f"   Distribution overlap:              : {overlap_frac:.1%} of boundary tiles "
          f"fall above the interior median")
    print(f"   => Threshold viability for escalation trigger: "
          f"{'LOW (heavy overlap)' if overlap_frac > 0.35 else 'MODERATE' if overlap_frac > 0.20 else 'HIGH (clean separation)'}")

    out = {
        "resize": resize_results,
        "per_stage_tile": {s: per_stage[s] for s in STAGES},
        "complexity": {str(k): np.mean(v).tolist() if v else None for k, v in complexity_res.items()},
        "boundary_diagnostic": {
            "n_interior": int(len(pure_arr)), "n_boundary": int(len(bnd_arr)),
            "interior_mean_conf": float(pure_arr.mean()), "boundary_mean_conf": float(bnd_arr.mean()),
            "t_stat": float(t_stat), "p_value": float(p_val),
            "overlap_fraction": float(overlap_frac),
        },
        "summary": {
            "resize_match_rate": match_rate,
            "overall_tile_accuracy": overall_tile_acc,
            "veg_false_positive_rate_when_not_present": (
                resize_preds_no_veg['vegetative'] / n_no_veg if n_no_veg else None
            ),
        }
    }
    out_path = args.composite_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
