"""
evaluate_grid_models.py

Comprehensive evaluation engine for 4-Class Growth-Stage Model Comparison Grid:
- Direct Resize Models: Evaluated on all 818 held-out Field 2 full captures (384x384).
- Tile Models: Evaluated on all 28,630 held-out Field 2 tiles (288x288) AND
  aggregated via 35-tile majority vote into 818 full-capture predictions.
- Generates:
  1. Full classification report (precision, recall, f1, support, macro-F1, weighted-F1, accuracy).
  2. Confusion matrices (both raw counts and row-normalized recall percentages).
  3. Vegetative -> Flowering and Flowering -> Vegetative specific confusion counts.
  4. High-resolution dual-panel heatmap figures (raw + normalized) saved to confusion_matrices/.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["Seedling", "Vegetative", "Flowering", "Mature"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

BUCKET_MAP = {
    "Nursery_Stage": "Seedling",
    "Vegetative_Stage": "Vegetative",
    "Booting_Stage": "Flowering",
    "Flowering_Stage": "Flowering",
    "Mature_Stage": "Mature",
    "Mature_stage": "Mature",
}

CACHE_DIR_512 = Path(r"C:/Users/LOQ/.gemini/antigravity-ide/brain/d004d148-72b1-4686-a1b8-31fe725f186c/scratch/cache_512")
SPLITS_DIR = Path("training/splits")
OUT_DIR = Path("training/grid_comparison")
CKPTS_DIR = OUT_DIR / "checkpoints"
CM_DIR = OUT_DIR / "confusion_matrices"


def build_model(arch: str, num_classes: int = 4) -> nn.Module:
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif arch == "efficientnet_b1":
        m = models.efficientnet_b1(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif arch == "efficientnet_b2":
        m = models.efficientnet_b2(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif arch == "efficientnet_b3":
        m = models.efficientnet_b3(weights=None)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def load_checkpoint(ckpt_path: Path, device: torch.device):
    data = torch.load(ckpt_path, map_location=device)
    arch = data["arch"]
    model = build_model(arch, len(CLASSES))
    model.load_state_dict(data["model_state"])
    model = model.to(device).eval()
    return model, data


# ── Resize Evaluation ─────────────────────────────────────────────────────────
class CachedTestResizeDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int, str]], input_size: int = 384):
        self.samples = samples
        self.tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cid = self.samples[idx]
        img = Image.open(path)
        return self.tf(img), label, cid


def evaluate_resize_model(model: nn.Module, res: int, device: torch.device) -> Dict:
    manifest_df = pd.read_csv("manifest.csv", low_memory=False)
    sources_df = manifest_df[
        (manifest_df["variant"] == "D") &
        (manifest_df["tile_file_path"].isna() | (manifest_df["tile_file_path"] == ""))
    ].drop_duplicates("capture_id").set_index("capture_id")

    test_split = pd.read_csv(SPLITS_DIR / "test.csv")
    test_cids = sorted(test_split["source_capture_id"].dropna().unique())

    samples = []
    for cid in test_cids:
        if cid in sources_df.index:
            stg = BUCKET_MAP.get(sources_df.loc[cid, "growth_stage"])
            if stg in CLASS_IDX:
                cache_p = CACHE_DIR_512 / f"{cid}_D_512.jpg"
                if cache_p.exists():
                    samples.append((str(cache_p), CLASS_IDX[stg], cid))

    loader = DataLoader(CachedTestResizeDataset(samples, input_size=res), batch_size=32, shuffle=False)

    preds, targets, cids = [], [], []
    with torch.no_grad():
        for imgs, lbls, cs in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(lbls.tolist())
            cids.extend(cs)

    return compute_full_metrics(targets, preds, item_type="Full UAV Captures (Field 2)")


# ── Tile Evaluation ───────────────────────────────────────────────────────────
class TestTileDataset(Dataset):
    def __init__(self, csv_path: Path, input_size: int = 288):
        df = pd.read_csv(csv_path)
        cache_dir = Path("scratch/cache_tiles_288/test")
        use_cache = cache_dir.exists()
        paths = df["tile_file_path"].tolist()
        buckets = df["bucket"].tolist()
        cids = df["source_capture_id"].tolist()
        self.samples = []
        for p, cid, b in zip(paths, cids, buckets):
            if b in CLASS_IDX:
                cp = str(cache_dir / f"{Path(p).stem}_288.jpg") if use_cache else str(Path(p))
                self.samples.append((cp, CLASS_IDX[b], cid))
        self.input_size = input_size
        self.tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cid = self.samples[idx]
        img = Image.open(path)
        return self.tf(img), label, cid


def evaluate_tile_model(model: nn.Module, res: int, device: torch.device) -> Tuple[Dict, Dict]:
    loader = DataLoader(TestTileDataset(SPLITS_DIR / "test.csv", input_size=res), batch_size=64, shuffle=False)

    tile_preds, tile_targets = [], []
    capture_tile_preds = defaultdict(list)
    capture_targets = {}

    with torch.no_grad():
        for imgs, lbls, cids in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            ps = logits.argmax(1).cpu().tolist()
            ls = lbls.tolist()

            tile_preds.extend(ps)
            tile_targets.extend(ls)

            for p, l, cid in zip(ps, ls, cids):
                capture_tile_preds[cid].append(p)
                capture_targets[cid] = l

    # 1. Tile-level metrics (all 28,630 tiles)
    tile_metrics = compute_full_metrics(tile_targets, tile_preds, item_type="Tiles (Field 2)")

    # 2. Capture-level metrics via 35-tile majority vote (818 captures)
    cap_preds, cap_targets = [], []
    for cid, plist in capture_tile_preds.items():
        vote = Counter(plist).most_common(1)[0][0]
        cap_preds.append(vote)
        cap_targets.append(capture_targets[cid])

    majority_metrics = compute_full_metrics(cap_targets, cap_preds, item_type="35-Tile Majority Vote (Field 2)")

    return majority_metrics, tile_metrics


# ── Metrics Computation ───────────────────────────────────────────────────────
def compute_full_metrics(y_true: List[int], y_pred: List[int], item_type: str) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec, rec, f1s, supp = precision_recall_fscore_support(y_true, y_pred, labels=list(range(4)), zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    veg_idx = CLASS_IDX["Vegetative"]
    flow_idx = CLASS_IDX["Flowering"]
    veg_to_flow = int(cm[veg_idx, flow_idx])
    flow_to_veg = int(cm[flow_idx, veg_idx])

    per_class = {}
    for i, c in enumerate(CLASSES):
        per_class[c] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1s[i]),
            "support": int(supp[i])
        }

    return {
        "item_type": item_type,
        "total_samples": len(y_true),
        "overall_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "veg_to_flow_confusion": veg_to_flow,
        "flow_to_veg_confusion": flow_to_veg,
        "y_true": y_true,
        "y_pred": y_pred
    }


# ── Confusion Matrix Heatmap Plotting ──────────────────────────────────────────
def plot_dual_confusion_matrix(cm_raw: np.ndarray, cm_norm: np.ndarray, title: str, subtitle: str, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    plt.subplots_adjust(wspace=0.35)

    # 1. Raw Counts
    im1 = ax1.imshow(cm_raw, cmap="Blues")
    ax1.set_title(f"{title}\nRaw Counts", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    ax1.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax1.set_yticklabels(CLASSES, fontsize=9)
    ax1.set_xlabel("Predicted Stage", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Ground Truth Stage", fontsize=10, fontweight="bold")

    for i in range(4):
        for j in range(4):
            val = cm_raw[i, j]
            color = "white" if val > cm_raw.max() * 0.5 else "black"
            ax1.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=9.5, fontweight="bold")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Row-Normalized Recall %
    im2 = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    ax2.set_title(f"{title}\nRow-Normalized Recall (%)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax2.set_yticklabels(CLASSES, fontsize=9)
    ax2.set_xlabel("Predicted Stage", fontsize=10, fontweight="bold")
    ax2.set_ylabel("True Ground Truth Stage", fontsize=10, fontweight="bold")

    for i in range(4):
        for j in range(4):
            pct = cm_norm[i, j] * 100
            val = cm_raw[i, j]
            txt = f"{pct:.1f}%\n({val:,})" if val > 0 else "0.0%"
            color = "white" if pct > 50 else "black"
            ax2.text(j, i, txt, ha="center", va="center", color=color, fontsize=8.5, fontweight="bold" if pct > 50 else "normal")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle(subtitle, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def evaluate_model_checkpoint(ckpt_file: Path, device: torch.device) -> Dict:
    model, meta = load_checkpoint(ckpt_file, device)
    arch = meta["arch"]
    method = meta["method"]
    res = meta["resolution"]
    param_count = meta.get("param_count", 0)
    training_time = meta.get("training_time", 0.0)
    model_key = f"{arch}_{method}"

    print(f"\nEvaluating {model_key.upper()} (Res: {res}x{res})...", flush=True)

    if method == "resize":
        metrics = evaluate_resize_model(model, res, device)
        tile_metrics = None
    elif method == "tile":
        metrics, tile_metrics = evaluate_tile_model(model, res, device)

    # Plot Confusion Matrix
    CM_DIR.mkdir(parents=True, exist_ok=True)
    cm_raw = np.array(metrics["confusion_matrix_raw"])
    cm_norm = np.array(metrics["confusion_matrix_norm"])
    cm_plot_path = CM_DIR / f"{model_key}_confusion_matrix.png"
    sub_title = f"{arch.upper()} [{method.upper()}] — Field 2 Holdout (Acc: {metrics['overall_accuracy']*100:.2f}%, Macro-F1: {metrics['macro_f1']:.4f})"
    plot_dual_confusion_matrix(cm_raw, cm_norm, f"{arch.upper()} ({method.capitalize()})", sub_title, cm_plot_path)

    metrics["arch"] = arch
    metrics["input_method"] = method
    metrics["resolution"] = res
    metrics["param_count"] = param_count
    metrics["training_time"] = training_time
    metrics["cm_plot_path"] = str(cm_plot_path)
    if tile_metrics:
        metrics["tile_metrics"] = tile_metrics

    print(f"  -> Acc: {metrics['overall_accuracy']*100:.2f}% | Macro-F1: {metrics['macro_f1']:.4f} | "
          f"V->F: {metrics['veg_to_flow_confusion']} | F->V: {metrics['flow_to_veg_confusion']}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Grid Models")
    parser.add_argument("--ckpt", type=Path, default=None, help="Specific checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    if args.ckpt:
        ckpts = [args.ckpt]
    else:
        ckpts = sorted(CKPTS_DIR.glob("*_best.pt"))

    if not ckpts:
        print("No checkpoints found in training/grid_comparison/checkpoints/")
        return

    all_eval_results = []
    for c in ckpts:
        res = evaluate_model_checkpoint(c, device)
        all_eval_results.append(res)

    out_json = OUT_DIR / "grid_evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        # Save serializable subset
        save_list = []
        for r in all_eval_results:
            d = {k: v for k, v in r.items() if k not in ("y_true", "y_pred")}
            save_list.append(d)
        json.dump(save_list, f, indent=2)

    print(f"\nAll evaluations complete! Results saved to {out_json}")


if __name__ == "__main__":
    main()
