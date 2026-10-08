"""
train_grid_models.py

Standardized training pipeline for 4-Class Growth-Stage Model Comparison Grid:
5 Architectures (EfficientNet-B0, B1, B2, B3, ResNet-50) x 2 Input Methods (Resize 384px, Tiling).

Protocol:
- 2 Warm-up epochs: Backbone frozen, classification head only (LR=1e-3).
- 8 Fine-tuning epochs: All layers unfrozen, backbone LR=1e-4, head LR=5e-4, CosineAnnealingLR.
- Loss: Focal Loss (gamma=2.0) with square-root damped class weights.
- AMP (Automatic Mixed Precision fp16).
- Full per-epoch metrics logged to CSV and plotted to PNG.
- Best checkpoint saved per model based on Validation Macro-F1.
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
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
CURVES_DIR = OUT_DIR / "curves"
CKPTS_DIR = OUT_DIR / "checkpoints"


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


# ── Datasets ──────────────────────────────────────────────────────────────────
class CachedResizeDataset(Dataset):
    """Loads pre-cached 512x512 full captures and applies augmentations/resizing."""
    def __init__(self, samples: List[Tuple[str, int]], input_size: int = 384, is_train: bool = True):
        self.samples = samples
        self.input_size = input_size
        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache_path, label = self.samples[idx]
        img = Image.open(cache_path)
        return self.tf(img), label


CACHE_TILES_ROOT = Path("scratch/cache_tiles_288")

class FastTileDataset(Dataset):
    """Loads tiles from pre-cached 288x288 directory (or falls back to raw 1024px)."""
    def __init__(self, csv_path: Path, input_size: int = 288, is_train: bool = True):
        df = pd.read_csv(csv_path)
        sub = "train" if is_train else "val"
        cache_dir = CACHE_TILES_ROOT / sub
        use_cache = cache_dir.exists()
        paths = df["tile_file_path"].tolist()
        buckets = df["bucket"].tolist()
        self.samples = [
            (str(cache_dir / f"{Path(p).stem}_288.jpg") if use_cache else str(Path(p)), CLASS_IDX[b])
            for p, b in zip(paths, buckets) if b in CLASS_IDX
        ]
        self.input_size = input_size
        
        # If tiles are already cached at 288x288, avoid redundant PIL resize
        t_list = []
        if not (use_cache and input_size == 288):
            t_list.append(transforms.Resize((input_size, input_size)))
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
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.tf(img), label


# ── Model Construction & Freezing ─────────────────────────────────────────────
def build_model(arch: str, num_classes: int = 4) -> nn.Module:
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif arch == "efficientnet_b1":
        m = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif arch == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif arch == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feat = m.fc.in_features
        m.fc = nn.Linear(in_feat, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def freeze_backbone(model: nn.Module, arch: str):
    for p in model.parameters():
        p.requires_grad = False
    if arch.startswith("efficientnet"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif arch == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def get_param_groups(model: nn.Module, arch: str, lr_backbone: float, lr_head: float):
    if arch.startswith("efficientnet"):
        head_params = list(model.classifier.parameters())
        backbone_params = list(model.features.parameters())
    elif arch == "resnet50":
        head_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head}
    ]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Training Engine ───────────────────────────────────────────────────────────
def train_model(
    arch: str,
    method: str,
    device: torch.device,
    epochs: int = 10,
    warmup_epochs: int = 2,
    batch_size: int = None,
    res: int = None,
) -> Dict:
    model_key = f"{arch}_{method}"
    print(f"\n{'='*70}")
    print(f" TRAINING MODEL: {model_key.upper()}")
    print(f" Architecture: {arch} | Input Method: {method}")
    print(f"{'='*70}", flush=True)

    if method == "resize":
        res = res or 384
        batch_size = batch_size or 32
        # Load cached samples for resize
        manifest_df = pd.read_csv("manifest.csv", low_memory=False)
        sources_df = manifest_df[
            (manifest_df["variant"] == "D") &
            (manifest_df["tile_file_path"].isna() | (manifest_df["tile_file_path"] == ""))
        ].drop_duplicates("capture_id").set_index("capture_id")

        def get_cached_samples(split_csv):
            sdf = pd.read_csv(split_csv)
            cids = sdf["source_capture_id"].dropna().unique()
            samples = []
            for cid in cids:
                if cid in sources_df.index:
                    stg = BUCKET_MAP.get(sources_df.loc[cid, "growth_stage"])
                    if stg in CLASS_IDX:
                        cache_p = CACHE_DIR_512 / f"{cid}_D_512.jpg"
                        if cache_p.exists():
                            samples.append((str(cache_p), CLASS_IDX[stg]))
            return samples

        train_samples = get_cached_samples(SPLITS_DIR / "train.csv")
        val_samples = get_cached_samples(SPLITS_DIR / "val.csv")
        train_ds = CachedResizeDataset(train_samples, input_size=res, is_train=True)
        val_ds = CachedResizeDataset(val_samples, input_size=res, is_train=False)

    elif method == "tile":
        res = res or 288
        batch_size = batch_size or 128
        train_ds = FastTileDataset(SPLITS_DIR / "train.csv", input_size=res, is_train=True)
        val_ds = FastTileDataset(SPLITS_DIR / "val.csv", input_size=res, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f" Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,} | Batch size: {batch_size} | Res: {res}x{res}", flush=True)

    # Compute square-root damped class weights
    class_counts = [0] * len(CLASSES)
    for _, l in train_ds.samples:
        class_counts[l] += 1
    max_c = max(class_counts)
    weights = [math.sqrt(max_c / max(c, 1)) for c in class_counts]
    alpha_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    print(f" Class weights: {[f'{CLASSES[i]}: {weights[i]:.3f}' for i in range(4)]}")

    model = build_model(arch, num_classes=len(CLASSES)).to(device)
    param_cnt = count_params(model)
    print(f" Parameter count: {param_cnt:,} ({param_cnt/1e6:.2f}M)")

    # 1. Warm-up stage: Freeze backbone, train head only
    freeze_backbone(model, arch)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    best_val_f1 = -1.0
    best_state = None
    curve_rows = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        # Stage transition after warm-up
        if epoch == warmup_epochs + 1:
            print(f" [Epoch {epoch}] Unfreezing all layers for full fine-tuning...")
            unfreeze_all(model)
            param_groups = get_param_groups(model, arch, lr_backbone=1e-4, lr_head=5e-4)
            optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
            remaining_epochs = epochs - warmup_epochs
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

        # Train loop
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for imgs, labels in train_loader:
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

        if epoch > warmup_epochs:
            scheduler.step()

        train_loss /= len(train_ds)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
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
            "time_s": ep_time
        })

    total_time = time.time() - t_start
    print(f" Total training time: {total_time:.1f}s ({total_time/60:.2f} min) | Best Val F1: {best_val_f1:.4f}")

    # Save curve CSV
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CURVES_DIR / f"{model_key}_curve.csv"
    pd.DataFrame(curve_rows).to_csv(csv_path, index=False)

    # Plot learning curve
    plot_learning_curve(curve_rows, model_key, CURVES_DIR / f"{model_key}_curve.png")

    # Save best checkpoint
    CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPTS_DIR / f"{model_key}_best.pt"
    torch.save({
        "arch": arch,
        "method": method,
        "resolution": res,
        "model_state": best_state,
        "best_val_f1": best_val_f1,
        "param_count": param_cnt,
        "training_time": total_time,
        "classes": CLASSES
    }, ckpt_path)
    print(f" Saved checkpoint to: {ckpt_path}")

    return {
        "model_key": model_key,
        "arch": arch,
        "method": method,
        "resolution": res,
        "param_count": param_cnt,
        "training_time": total_time,
        "best_val_f1": best_val_f1,
        "ckpt_path": str(ckpt_path),
        "curve_path": str(csv_path)
    }


def plot_learning_curve(rows: List[Dict], title: str, out_path: Path):
    df = pd.DataFrame(rows)
    epochs = df["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs, df["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    ax1.plot(epochs, df["val_loss"], "s--", color="#ff7f0e", label="Val Loss", linewidth=2)
    ax1.set_title(f"{title} — Focal Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # F1 / Acc curve
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


def main():
    parser = argparse.ArgumentParser(description="Train Grid Models")
    parser.add_argument("--arch", choices=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "resnet50", "all"], default="all")
    parser.add_argument("--method", choices=["resize", "tile", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    archs = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "resnet50"] if args.arch == "all" else [args.arch]
    methods = ["resize", "tile"] if args.method == "all" else [args.method]

    results = []
    for m in methods:
        for a in archs:
            res = train_model(a, m, device, epochs=args.epochs, warmup_epochs=args.warmup_epochs)
            results.append(res)

    print("\n" + "="*80)
    print(" ALL GRID TRAINING RUNS COMPLETED SUCCESSFULLY")
    print("="*80)
    for r in results:
        print(f" {r['model_key']:<25} | Best Val F1: {r['best_val_f1']:.4f} | Time: {r['training_time']:.1f}s ({r['training_time']/60:.2f}m)")


if __name__ == "__main__":
    main()
