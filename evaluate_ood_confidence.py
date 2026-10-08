"""
evaluate_ood_confidence.py -- Compare calibration and confidence drop-off between
Resize model and Full-Finetune Tile model on In-Distribution vs Near-OOD vs Far-OOD images.

Usage:
    python evaluate_ood_confidence.py --ood-dir "e:/paddy_test/ood_test"
"""

import argparse
from pathlib import Path
import csv
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms

CLASSES_4 = ["Seedling", "Vegetative", "Flowering", "Mature"]
CLASSES_5 = ["Seedling", "Vegetative", "Flowering", "Mature", "Not_Rice"]

class EnhancedRiceClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.efficientnet_b0()
        self.features = base.features
        self.avgpool  = base.avgpool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 384), nn.LayerNorm(384), nn.SiLU(), nn.Dropout(0.35),
            nn.Linear(384, 128),  nn.LayerNorm(128),  nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.head(self.avgpool(self.features(x)))

def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"]
    input_size = ckpt.get("input_size", 224)

    if any(k.startswith("head.") for k in state.keys()):
        num_classes = state.get("head.9.weight", state.get("head.7.weight", torch.zeros(4))).shape[0]
        class_list = CLASSES_5 if num_classes == 5 else CLASSES_4
        model = EnhancedRiceClassifier(num_classes=num_classes)
        arch = f"Tile Model ({num_classes}-Class)"
    else:
        num_classes = 4
        class_list = CLASSES_4
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        arch = "Resize Model (4-Class)"

    model.load_state_dict(state)
    model.to(device).eval()
    return model, input_size, arch, class_list

def predict_single(image_path: Path, model, input_size: int, class_list: list, device):
    img = Image.open(image_path)
    img.draft("RGB", (input_size, input_size))
    img = img.convert("RGB")

    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    return class_list[pred_idx], conf, probs

def evaluate_tier(model, input_size, class_list, device, image_paths, tier_name):
    results = []
    for p in image_paths:
        pred_cls, conf, probs = predict_single(p, model, input_size, class_list, device)
        results.append({
            "path": str(p),
            "tier": tier_name,
            "pred_class": pred_cls,
            "confidence": conf,
            "probs": probs
        })
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood-dir", type=Path, default=Path("e:/paddy_test/ood_test"))
    ap.add_argument("--resize-ckpt", type=Path, default=Path("e:/paddy_test/training/checkpoints/checkpoint_resize_best.pt"))
    ap.add_argument("--tile-ckpt", type=Path, default=Path("e:/paddy_test/training/checkpoints/checkpoint_full_finetune.pt"))
    ap.add_argument("--five-class-ckpt", type=Path, default=Path("e:/paddy_test/training/checkpoints/checkpoint_5class_finetune.pt"))
    ap.add_argument("--test-csv", type=Path, default=Path("e:/paddy_test/training/splits/test.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("e:/paddy_test/training/eval_results"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Gather images
    near_ood_imgs = sorted(list((args.ood_dir / "near_ood").glob("**/*.jpg")) + list((args.ood_dir / "near_ood").glob("**/*.png")))
    far_ood_imgs  = sorted(list((args.ood_dir / "far_ood").glob("**/*.jpg")) + list((args.ood_dir / "far_ood").glob("**/*.png")))

    # Load in-dist samples from test.csv
    in_dist_imgs = []
    if args.test_csv.exists():
        with open(args.test_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            np.random.seed(42)
            sample_rows = np.random.choice(rows, size=min(100, len(rows)), replace=False)
            for r in sample_rows:
                p = Path(r["tile_file_path"])
                if p.exists():
                    in_dist_imgs.append(p)

    print(f"Dataset Counts -> In-Distribution: {len(in_dist_imgs)} | Near-OOD: {len(near_ood_imgs)} | Far-OOD: {len(far_ood_imgs)}")

    if not near_ood_imgs and not far_ood_imgs:
        print(f"\n[!] No OOD images found in {args.ood_dir}.")
        print("Please run fetch_ood_test_images.py first:")
        print(f"    python fetch_ood_test_images.py --out-dir {args.ood_dir}")
        return

    models_to_test = [
        ("Resize (checkpoint_resize_best.pt)", args.resize_ckpt),
        ("Tile (checkpoint_full_finetune.pt)", args.tile_ckpt),
        ("5-Class (checkpoint_5class_finetune.pt)", args.five_class_ckpt),
    ]

    all_summary = {}

    for name, ckpt_path in models_to_test:
        if not ckpt_path.exists():
            continue

        print(f"\nEvaluating: {name} ...")
        model, input_size, arch, class_list = load_model(ckpt_path, device)

        results_in_dist = evaluate_tier(model, input_size, class_list, device, in_dist_imgs, "In-Dist") if in_dist_imgs else []
        results_near    = evaluate_tier(model, input_size, class_list, device, near_ood_imgs, "Near-OOD")
        results_far     = evaluate_tier(model, input_size, class_list, device, far_ood_imgs, "Far-OOD")

        all_summary[name] = {
            "In-Dist": results_in_dist,
            "Near-OOD": results_near,
            "Far-OOD": results_far,
            "classes": class_list
        }

    # Print summary table
    print("\n" + "="*85)
    print(f"{'Model':<40} | {'Tier':<10} | {'Count':<5} | {'Mean Conf':<10} | {'Median Conf':<11} | {'>90% Conf':<9}")
    print("="*85)

    for name, info in all_summary.items():
        for tier_name in ["In-Dist", "Near-OOD", "Far-OOD"]:
            res = info.get(tier_name, [])
            if not res:
                continue
            confs = [r["confidence"] for r in res]
            mean_c = np.mean(confs) * 100
            med_c  = np.median(confs) * 100
            over_90 = sum(1 for c in confs if c >= 0.90) / len(confs) * 100
            print(f"{name:<40} | {tier_name:<10} | {len(res):<5} | {mean_c:>8.2f}% | {med_c:>9.2f}% | {over_90:>7.1f}%")
        print("-" * 85)

    # Class bias / rejection analysis on OOD
    print("\nOOD PREDICTION CLASS BREAKDOWN (% of predictions assigned to each class):")
    print(f"{'Model':<40} | {'Tier':<10} | {'Seedling':<9} | {'Vegetative':<10} | {'Flowering':<9} | {'Mature':<8} | {'Not_Rice':<9}")
    print("-" * 105)
    for name, info in all_summary.items():
        for tier_name in ["Near-OOD", "Far-OOD"]:
            res = info.get(tier_name, [])
            if not res:
                continue
            preds = [r["pred_class"] for r in res]
            counts = {c: preds.count(c) / len(preds) * 100 for c in CLASSES_5}
            print(f"{name:<40} | {tier_name:<10} | {counts['Seedling']:>7.1f}% | {counts['Vegetative']:>8.1f}% | {counts['Flowering']:>7.1f}% | {counts['Mature']:>6.1f}% | {counts['Not_Rice']:>7.1f}%")

    # Plot confidence distributions
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_models = len(all_summary)
    fig, axes = plt.subplots(1, max(1, n_models), figsize=(5 * max(1, n_models), 5), sharey=True, squeeze=False)

    tier_colors = {"In-Dist": "#2ecc71", "Near-OOD": "#f39c12", "Far-OOD": "#e74c3c"}

    for idx, (name, info) in enumerate(all_summary.items()):
        ax = axes[0, idx]
        data_to_plot = []
        labels = []
        for t_name in ["In-Dist", "Near-OOD", "Far-OOD"]:
            res = info.get(t_name, [])
            if res:
                data_to_plot.append([r["confidence"] * 100 for r in res])
                labels.append(f"{t_name}\n(n={len(res)})")

        if data_to_plot:
            bplot = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=labels)
            for patch, t_name in zip(bplot['boxes'], [t for t in ["In-Dist", "Near-OOD", "Far-OOD"] if info.get(t)]):
                patch.set_facecolor(tier_colors.get(t_name, "#3498db"))
                patch.set_alpha(0.7)

        ax.set_title(f"{name}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Max Softmax Confidence (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("OOD Confidence Calibration: In-Distribution vs Near-OOD vs Far-OOD", fontsize=13)
    plt.tight_layout()
    plot_path = args.out_dir / "ood_confidence_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nConfidence comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
