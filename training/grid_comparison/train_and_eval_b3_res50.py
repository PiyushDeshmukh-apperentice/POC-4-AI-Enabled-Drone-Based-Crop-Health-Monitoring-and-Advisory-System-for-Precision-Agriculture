"""
train_and_eval_b3_res50.py

Trains and evaluates:
1. efficientnet_b3 (tile 288x288) for 5 full epochs
2. resnet50 (tile 288x288) for 5 full epochs

Matched Protocol (Identical to B1 and B2):
- Resolution: 288x288 (using scratch/cache_tiles_288)
- 5 Epochs: 1 Warm-up (Head only, LR=1e-3) + 4 Full Fine-Tuning (CosineAnnealingLR, backbone LR=1e-4, head LR=5e-4)
- Batch size: 64 (prevents CUDA OOM on 8GB RTX 5050 GPU)
- Loss: Focal Loss (gamma=2.0) with square-root damped class weights
- AMP fp16

Outputs:
- Checkpoints: training/grid_comparison/checkpoints/{arch}_tile_best.pt
- 5-Epoch Learning Curves: training/grid_comparison/curves/{arch}_tile_curve.csv & .png
- Raw Tile Confusion Matrices (N = 28,630): training/grid_comparison/confusion_matrices/{arch}_raw_tile_confusion_matrix.png
- Capture Majority Vote Confusion Matrices (N = 818): training/grid_comparison/confusion_matrices/{arch}_tile_confusion_matrix.png
- Updates training/grid_comparison/grid_evaluation_results.json with both raw tile and capture metrics
"""

import os
import sys
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

CLASSES = ["Seedling", "Vegetative", "Flowering", "Mature"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

SPLITS_DIR = Path("training/splits")
OUT_DIR = Path("training/grid_comparison")
CKPTS_DIR = OUT_DIR / "checkpoints"
CURVES_DIR = OUT_DIR / "curves"
CM_DIR = OUT_DIR / "confusion_matrices"
EVAL_JSON = OUT_DIR / "grid_evaluation_results.json"
CACHE_TILES_ROOT = Path("scratch/cache_tiles_288")

CKPTS_DIR.mkdir(parents=True, exist_ok=True)
CURVES_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class CachedTileDataset(Dataset):
    def __init__(self, csv_path: Path, sub: str, input_size: int = 288, is_train: bool = True):
        df = pd.read_csv(csv_path)
        cache_dir = CACHE_TILES_ROOT / sub
        use_cache = cache_dir.exists()
        paths = df["tile_file_path"].tolist()
        buckets = df["bucket"].tolist()
        cids = df["source_capture_id"].tolist() if "source_capture_id" in df.columns else [""] * len(paths)

        self.samples = []
        for p, b, cid in zip(paths, buckets, cids):
            if b in CLASS_IDX:
                c_path = cache_dir / f"{Path(p).stem}_288.jpg"
                actual_path = str(c_path) if (use_cache and c_path.exists()) else str(Path(p))
                self.samples.append((actual_path, CLASS_IDX[b], str(cid)))

        self.input_size = input_size
        self.use_cache = use_cache

        t_list = []
        if is_train:
            t_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ])
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.tf = transforms.Compose(t_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, cid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if not self.use_cache:
            img = img.resize((self.input_size, self.input_size))
        return self.tf(img), label, cid


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_loss_weights() -> torch.Tensor:
    # Square-root damped class weights based on train set
    train_counts = [6930, 6650, 13300, 7035] # Seedling, Vegetative, Flowering, Mature
    max_c = max(train_counts)
    weights = [math.sqrt(max_c / c) for c in train_counts]
    return torch.tensor(weights, dtype=torch.float32)


# ── Model Builder ─────────────────────────────────────────────────────────────
def build_model(arch: str, num_classes: int = 4) -> nn.Module:
    if arch == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m


def freeze_backbone(model: nn.Module, arch: str):
    for param in model.parameters():
        param.requires_grad = False
    if "efficientnet" in arch:
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif "resnet" in arch:
        for param in model.fc.parameters():
            param.requires_grad = True


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ── Confusion Matrix Plotter ──────────────────────────────────────────────────
def plot_dual_confusion_matrix(cm_raw: np.ndarray, cm_norm: np.ndarray, title: str, subtitle: str, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    plt.subplots_adjust(wspace=0.35)

    im1 = ax1.imshow(cm_raw, cmap="Blues")
    ax1.set_title(f"{title}\nRaw Sample Counts", fontsize=11, fontweight="bold", pad=8)
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


def plot_learning_curve(rows: List[Dict], title: str, out_path: Path):
    df = pd.DataFrame(rows)
    epochs = df["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, df["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    ax1.plot(epochs, df["val_loss"], "s--", color="#ff7f0e", label="Val Loss", linewidth=2)
    ax1.set_title(f"{title} — Focal Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.plot(epochs, df["train_f1"], "o-", color="#2ca02c", label="Train Macro-F1", linewidth=2)
    ax2.plot(epochs, df["val_f1"], "s--", color="#d62728", label="Val Macro-F1", linewidth=2)
    ax2.plot(epochs, df["val_acc"], "^:", color="#9467bd", label="Val Accuracy", linewidth=1.5, alpha=0.7)
    ax2.set_title(f"{title} — Macro-F1 & Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_ylim(bottom=max(0.0, df["val_f1"].min() - 0.1), top=1.02)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_model(arch: str, device: torch.device, epochs: int = 5, batch_size: int = 64) -> Dict:
    print("\n" + "=" * 80, flush=True)
    print(f" TRAINING {arch.upper()} (TILE 288x288) FOR {epochs} MATCHED EPOCHS", flush=True)
    print("=" * 80, flush=True)

    train_ds = CachedTileDataset(SPLITS_DIR / "train.csv", sub="train", is_train=True)
    val_ds   = CachedTileDataset(SPLITS_DIR / "val.csv",   sub="val",   is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(arch, num_classes=4).to(device)
    weights = get_loss_weights().to(device)
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    scaler = torch.amp.GradScaler('cuda')

    # Param count
    param_cnt = sum(p.numel() for p in model.parameters())

    # Phase 1: Warm-up (Epoch 1)
    freeze_backbone(model, arch)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)

    best_val_f1 = 0.0
    best_state = None
    curve_rows = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()

        if epoch == 2:
            print("  [Epoch 2] Unfreezing all backbone layers for full fine-tuning...", flush=True)
            unfreeze_all(model)
            if "efficientnet" in arch:
                backbone_p = list(model.features.parameters())
                head_p = list(model.classifier.parameters())
            else:
                backbone_p = [p for n, p in model.named_parameters() if not n.startswith("fc")]
                head_p = list(model.fc.parameters())

            optimizer = optim.AdamW([
                {"params": backbone_p, "lr": 1e-4},
                {"params": head_p, "lr": 5e-4},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 1, eta_min=1e-6)

        # Train loop
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(labels)
            train_preds.extend(logits.argmax(1).cpu().tolist())
            train_targets.extend(labels.cpu().tolist())

        if epoch >= 2:
            scheduler.step()

        train_loss /= len(train_ds)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)

        # Val loop
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        val_loss /= len(val_ds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        ep_time = time.time() - ep_t0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = "[BEST]"
        else:
            tag = ""

        print(f" Ep {epoch:2d}/{epochs:2d} | TrLoss: {train_loss:.4f} TrAcc: {train_acc*100:5.2f}% TrF1: {train_f1:.4f} | "
              f"VaLoss: {val_loss:.4f} VaAcc: {val_acc*100:5.2f}% VaF1: {val_f1:.4f} | {ep_time:4.1f}s {tag}", flush=True)

        curve_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "epoch_time_s": ep_time
        })

    total_time = time.time() - t_start
    print(f" Total training time: {total_time:.1f}s ({total_time/60:.2f} min) | Best Val F1: {best_val_f1:.4f}", flush=True)

    # Save curve CSV & PNG
    csv_path = CURVES_DIR / f"{arch}_tile_curve.csv"
    png_path = CURVES_DIR / f"{arch}_tile_curve.png"
    pd.DataFrame(curve_rows).to_csv(csv_path, index=False)
    plot_learning_curve(curve_rows, f"{arch.upper()} (Tile 288px)", png_path)

    # Save best checkpoint
    ckpt_path = CKPTS_DIR / f"{arch}_tile_best.pt"
    torch.save({
        "arch": arch,
        "method": "tile",
        "resolution": 288,
        "epochs_trained": epochs,
        "model_state": best_state,
        "best_val_f1": best_val_f1,
        "param_count": param_cnt,
        "training_time": total_time,
        "classes": CLASSES
    }, ckpt_path)
    print(f" Checkpoint saved to: {ckpt_path}", flush=True)

    # Load best weights into model for evaluation
    model.load_state_dict(best_state)
    return model, total_time, param_cnt


# ── Dual-Scale Test Evaluation ────────────────────────────────────────────────
def evaluate_dual_scale(model: nn.Module, arch: str, total_time: float, param_cnt: int, device: torch.device):
    print(f"\nEvaluating {arch.upper()} on Held-Out Field 2 Test Set (Dual Scale)...", flush=True)
    test_ds = CachedTileDataset(SPLITS_DIR / "test.csv", sub="test", is_train=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    tile_preds, tile_targets = [], []
    capture_tile_preds = defaultdict(list)
    capture_targets = {}

    with torch.no_grad():
        for imgs, labels, cids in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            preds = logits.argmax(1).cpu().tolist()
            lbls = labels.tolist()

            tile_preds.extend(preds)
            tile_targets.extend(lbls)

            for cid, p, t in zip(cids, preds, lbls):
                capture_tile_preds[cid].append(p)
                capture_targets[cid] = t

    # 1. Scale A: Raw Tile Metrics (N = 28,630)
    t_acc = accuracy_score(tile_targets, tile_preds)
    t_f1  = f1_score(tile_targets, tile_preds, average="macro", zero_division=0)
    t_prec, t_rec, t_f1s, t_supp = precision_recall_fscore_support(tile_targets, tile_preds, labels=list(range(4)), zero_division=0)
    cm_t_raw = confusion_matrix(tile_targets, tile_preds, labels=list(range(4)))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_t_norm = np.nan_to_num(cm_t_raw.astype('float') / cm_t_raw.sum(axis=1)[:, np.newaxis])

    veg_idx, flow_idx = CLASS_IDX["Vegetative"], CLASS_IDX["Flowering"]
    v_to_f_tile = int(cm_t_raw[veg_idx, flow_idx])
    f_to_v_tile = int(cm_t_raw[flow_idx, veg_idx])

    # Plot Raw Tile CM
    raw_cm_path = CM_DIR / f"{arch}_raw_tile_confusion_matrix.png"
    plot_dual_confusion_matrix(
        cm_t_raw, cm_t_norm,
        f"{arch.upper()} — Raw Tiles",
        f"{arch.upper()} [RAW TILES] — Field 2 Holdout (N=28,630, Acc: {t_acc*100:.2f}%, Macro-F1: {t_f1:.4f})",
        raw_cm_path
    )
    print(f"  [Scale A - Raw Tiles]: Acc={t_acc*100:.2f}% | Macro-F1={t_f1:.4f} | V->F={v_to_f_tile} | CM saved to {raw_cm_path.name}", flush=True)

    # 2. Scale B: Capture Majority Vote (N = 818)
    cap_preds, cap_targets = [], []
    for cid, plist in capture_tile_preds.items():
        vote = Counter(plist).most_common(1)[0][0]
        cap_preds.append(vote)
        cap_targets.append(capture_targets[cid])

    c_acc = accuracy_score(cap_targets, cap_preds)
    c_f1  = f1_score(cap_targets, cap_preds, average="macro", zero_division=0)
    c_prec, c_rec, c_f1s, c_supp = precision_recall_fscore_support(cap_targets, cap_preds, labels=list(range(4)), zero_division=0)
    cm_c_raw = confusion_matrix(cap_targets, cap_preds, labels=list(range(4)))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_c_norm = np.nan_to_num(cm_c_raw.astype('float') / cm_c_raw.sum(axis=1)[:, np.newaxis])

    v_to_f_cap = int(cm_c_raw[veg_idx, flow_idx])
    f_to_v_cap = int(cm_c_raw[flow_idx, veg_idx])

    cap_cm_path = CM_DIR / f"{arch}_tile_confusion_matrix.png"
    plot_dual_confusion_matrix(
        cm_c_raw, cm_c_norm,
        f"{arch.upper()} — Majority Vote",
        f"{arch.upper()} [MAJORITY VOTE] — Field 2 Holdout (N=818, Acc: {c_acc*100:.2f}%, Macro-F1: {c_f1:.4f})",
        cap_cm_path
    )
    print(f"  [Scale B - Majority Vote]: Acc={c_acc*100:.2f}% | Macro-F1={c_f1:.4f} | V->F={v_to_f_cap} | CM saved to {cap_cm_path.name}", flush=True)

    # Package per-class
    per_class_cap = {
        CLASSES[i]: {"precision": float(c_prec[i]), "recall": float(c_rec[i]), "f1": float(c_f1s[i]), "support": int(c_supp[i])}
        for i in range(4)
    }
    per_class_tile = {
        CLASSES[i]: {"precision": float(t_prec[i]), "recall": float(t_rec[i]), "f1": float(t_f1s[i]), "support": int(t_supp[i])}
        for i in range(4)
    }

    result_entry = {
        "item_type": "35-Tile Majority Vote (Field 2)",
        "arch": arch,
        "input_method": "tile",
        "resolution": 288,
        "epochs_trained": 5,
        "param_count": param_cnt,
        "training_time": total_time,
        "total_samples": len(cap_targets),
        "overall_accuracy": float(c_acc),
        "macro_f1": float(c_f1),
        "veg_to_flow_confusion": v_to_f_cap,
        "flow_to_veg_confusion": f_to_v_cap,
        "per_class": per_class_cap,
        "confusion_matrix_raw": cm_c_raw.tolist(),
        "confusion_matrix_norm": cm_c_norm.tolist(),
        "cm_plot_path": str(cap_cm_path),
        "raw_tile_metrics": {
            "item_type": "Raw Individual Tiles (Field 2)",
            "total_samples": len(tile_targets),
            "overall_accuracy": float(t_acc),
            "macro_f1": float(t_f1),
            "veg_to_flow_confusion": v_to_f_tile,
            "flow_to_veg_confusion": f_to_v_tile,
            "per_class": per_class_tile,
            "confusion_matrix_raw": cm_t_raw.tolist(),
            "confusion_matrix_norm": cm_t_norm.tolist(),
            "cm_plot_path": str(raw_cm_path)
        }
    }
    return result_entry


# ── Update Grid JSON ──────────────────────────────────────────────────────────
def update_grid_json(new_entries: List[Dict]):
    if EVAL_JSON.exists():
        with open(EVAL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    for entry in new_entries:
        arch = entry["arch"]
        meth = entry["input_method"]
        # Replace existing or append
        idx_match = None
        for i, item in enumerate(data):
            if item.get("arch") == arch and item.get("input_method") == meth:
                idx_match = i
                break
        if idx_match is not None:
            data[idx_match] = entry
        else:
            data.append(entry)

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nUpdated {EVAL_JSON.name} successfully with matched 5-epoch entries.", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    models_to_train = ["efficientnet_b3", "resnet50"]
    new_results = []

    for arch in models_to_train:
        model, total_time, param_cnt = train_model(arch, device, epochs=5, batch_size=64)
        entry = evaluate_dual_scale(model, arch, total_time, param_cnt, device)
        new_results.append(entry)

    update_grid_json(new_results)
    print("\n" + "=" * 80)
    print(" ALL REQUESTED 5-EPOCH RUNS & RAW TILE EVALUATIONS COMPLETED SUCCESSFULLY")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
