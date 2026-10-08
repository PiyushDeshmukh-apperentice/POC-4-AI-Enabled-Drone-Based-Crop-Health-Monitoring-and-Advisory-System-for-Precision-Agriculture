"""
04_full_finetune.py -- Full unfreeze continuation training for the tile model.

Resumes from checkpoint_enhanced_ann.pt (epoch 3, Val F1=0.9789) and
trains with all backbone blocks unfrozen. Implements:
  - Early stopping on Val F1 (patience = --patience epochs of no improvement)
  - Per-epoch learning curve log (learning_curve.csv + learning_curve.png)
  - Saves best checkpoint as checkpoint_full_finetune.pt
  - Final test evaluation on held-out Field 2 split

Usage:
    python 04_full_finetune.py --epochs 30 --patience 5
"""

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ── Config ───────────────────────────────────────────────────────────────────
CLASSES   = ["Seedling", "Vegetative", "Flowering", "Mature"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
ORDINAL   = CLASS_IDX

SPLITS_DIR = Path("splits")
CKPT_DIR   = Path("checkpoints")
EVAL_DIR   = Path("eval_results")

RESUME_CKPT = CKPT_DIR / "checkpoint_enhanced_ann.pt"  # ep3, F1=0.9789
OUT_CKPT    = CKPT_DIR / "checkpoint_full_finetune.pt"
CURVE_CSV   = EVAL_DIR / "learning_curve.csv"
CURVE_PNG   = EVAL_DIR / "learning_curve.png"

# ── Model ────────────────────────────────────────────────────────────────────
class EnhancedRiceClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool  = base.avgpool
        in_features   = base.classifier[1].in_features  # 1280
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 384), nn.LayerNorm(384), nn.SiLU(), nn.Dropout(0.35),
            nn.Linear(384, 128),         nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.avgpool(self.features(x)))


def unfreeze_all(model: EnhancedRiceClassifier):
    """Unfreeze every parameter in both backbone and head."""
    for p in model.parameters():
        p.requires_grad = True


# ── Loss ─────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets, reduction="none")
        pt    = torch.exp(-ce)
        return (self.alpha[targets] * ((1 - pt) ** self.gamma) * ce).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────
class FastTileDataset(Dataset):
    def __init__(self, csv_path: Path, input_size: int, is_train: bool):
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r.get("bucket") in CLASS_IDX]
        self.input_size = input_size
        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["tile_file_path"])
        img.draft("RGB", (self.input_size, self.input_size))
        return self.tf(img.convert("RGB")), CLASS_IDX[r["bucket"]]


def compute_class_weights(csv_path: Path) -> torch.Tensor:
    counts = {c: 0 for c in CLASSES}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("bucket") in counts:
                counts[r["bucket"]] += 1
    mx = max(counts.values())
    return torch.tensor([math.sqrt(mx / max(counts[c], 1)) for c in CLASSES], dtype=torch.float32)


# ── Epoch Runner ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training, scaler=None):
    model.train(training)
    loss_sum, correct, preds_all, labels_all = 0.0, 0, [], []
    total = len(loader)
    with torch.set_grad_enabled(training):
        for b, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # AMP autocast: uses float16 on CUDA Tensor Core ops for speed
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            p = logits.argmax(1)
            loss_sum  += loss.item() * len(labels)
            correct   += (p == labels).sum().item()
            preds_all .extend(p.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            if (b + 1) % 50 == 0 or (b + 1) == total:
                print(f"  {'Train' if training else 'Val  '} {b+1:>4}/{total} | "
                      f"loss={loss_sum/len(labels_all):.4f} acc={correct/len(labels_all):.3f}",
                      end="\r" if (b + 1) < total else "\n", flush=True)

    n = len(labels_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return loss_sum / n, correct / n, float(macro_f1)


# ── Learning Curve Plot ───────────────────────────────────────────────────────
def save_curve(history):
    epochs   = [h["epoch"] for h in history]
    tr_loss  = [h["train_loss"] for h in history]
    va_loss  = [h["val_loss"]   for h in history]
    tr_f1    = [h["train_f1"]   for h in history]
    va_f1    = [h["val_f1"]     for h in history]
    tr_acc   = [h["train_acc"]  for h in history]
    va_acc   = [h["val_acc"]    for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, tr_loss, "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, va_loss, "r-s", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, tr_acc, "b-o",  label="Train Acc", markersize=4)
    axes[1].plot(epochs, va_acc, "r-s",  label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0.85, 1.01)

    axes[2].plot(epochs, tr_f1, "b-o", label="Train Macro F1", markersize=4)
    axes[2].plot(epochs, va_f1, "r-s", label="Val Macro F1",   markersize=4)
    axes[2].set_title("Macro F1 per Epoch"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1"); axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0.85, 1.01)

    plt.suptitle("Full-Unfreeze Fine-Tune Learning Curve (Tile Model)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=150, bbox_inches="tight")
    plt.close()


# ── Test Evaluation ───────────────────────────────────────────────────────────
def evaluate_test(model, test_dl, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            p = model(imgs.to(device)).argmax(1)
            preds_all .extend(p.cpu().tolist())
            labels_all.extend(labels.tolist())

    acc      = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    cm       = confusion_matrix(labels_all, preds_all, labels=list(range(len(CLASSES))))
    pr, rc, f1s, sup = precision_recall_fscore_support(labels_all, preds_all,
                                                        labels=list(range(len(CLASSES))))

    print("\n" + "="*65)
    print("FULL-UNFREEZE TILE MODEL  |  TEST SET (HELD-OUT FIELD 2)")
    print("="*65)
    print(f"Overall Accuracy : {acc*100:.2f}%")
    print(f"Macro F1 Score   : {macro_f1:.4f}")
    print("-"*65)
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-"*65)
    for i, c in enumerate(CLASSES):
        print(f"{c:<14} {pr[i]:>10.4f} {rc[i]:>10.4f} {f1s[i]:>10.4f} {sup[i]:>10,}")
    print("="*65)

    print("\nConfusion Matrix (rows=True, cols=Pred):")
    header = f"{'':>14}" + "".join(f"{c:>14}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>14}" + "".join(f"{v:>14,}" for v in row))

    # Save normalized confusion matrix plot
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Full-Unfreeze Tile Model — Normalized Confusion Matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm_norm[i,j]*100:.1f}%", ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix_full_finetune.png", dpi=300)
    plt.close()
    print(f"\nConfusion matrix saved to: {EVAL_DIR / 'confusion_matrix_full_finetune.png'}")
    return acc, macro_f1


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",      type=int,   default=30,   help="Max epochs to run")
    ap.add_argument("--patience",    type=int,   default=5,    help="Early-stop patience (epochs of no Val F1 improvement)")
    ap.add_argument("--batch-size",  type=int,   default=64)
    ap.add_argument("--lr-head",     type=float, default=5e-5, help="LR for ANN head (lower than initial, continuing run)")
    ap.add_argument("--lr-backbone", type=float, default=5e-6, help="LR for fully-unfrozen backbone")
    ap.add_argument("--min-delta",   type=float, default=1e-4, help="Minimum F1 improvement to reset patience counter")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # cuDNN auto-tuner: finds fastest conv algorithm for fixed input sizes
        torch.backends.cudnn.benchmark = True
    CKPT_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)

    # ── Build model & load saved weights ─────────────────────────────────────
    model = EnhancedRiceClassifier(num_classes=len(CLASSES)).to(device)

    if RESUME_CKPT.exists():
        ckpt_data = torch.load(RESUME_CKPT, map_location=device)
        model.load_state_dict(ckpt_data["model"])
        start_f1  = ckpt_data.get("val_f1", 0.0)
        start_ep  = ckpt_data.get("epoch",  0)
        input_size = ckpt_data.get("input_size", 288)
        print(f"Resumed from {RESUME_CKPT.name}  (epoch {start_ep}, Val F1={start_f1:.4f})")
    else:
        raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CKPT}")

    # Fully unfreeze all backbone + head
    unfreeze_all(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device} | Input size: {input_size}x{input_size}")
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = FastTileDataset(SPLITS_DIR / "train.csv", input_size, is_train=True)
    val_ds   = FastTileDataset(SPLITS_DIR / "val.csv",   input_size, is_train=False)
    test_ds  = FastTileDataset(SPLITS_DIR / "test.csv",  input_size, is_train=False)
    # num_workers > 0 enables parallel data loading (big speedup on CPU-bound pipelines).
    # pin_memory=True speeds up CPU→GPU transfers when using a CUDA device.
    _nw = 4 if device.type == "cuda" else 2
    _pm = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=_nw, pin_memory=_pm, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=_nw, pin_memory=_pm, persistent_workers=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=_nw, pin_memory=_pm, persistent_workers=True)
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # ── Optimizer, scheduler, loss ────────────────────────────────────────────
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(),
         "lr": args.lr_head, "weight_decay": 1e-4},
        {"params": [p for p in model.features.parameters() if p.requires_grad],
         "lr": args.lr_backbone, "weight_decay": 1e-4},
    ])
    # Cosine anneal from current LRs down to near-zero over all remaining epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    class_weights = compute_class_weights(SPLITS_DIR / "train.csv").to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    print(f"Focal Loss alpha: {[round(w, 4) for w in class_weights.cpu().tolist()]}")

    # AMP GradScaler: prevents float16 underflow during backward pass
    scaler = torch.amp.GradScaler(device=device.type) if device.type == "cuda" else None
    print(f"AMP GradScaler: {'enabled' if scaler else 'disabled (CPU)'}")

    # ── Training loop with early stopping ────────────────────────────────────
    best_val_f1  = start_f1
    best_epoch   = start_ep
    patience_ctr = 0
    history      = []

    print(f"\nMax epochs: {args.epochs} | Patience: {args.patience} | Min-delta: {args.min_delta}")
    print(f"{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'TrF1':>6}  {'VaLoss':>7}  {'VaAcc':>6}  {'VaF1':>6}  {'LR-head':>8}  {'Patience':>8}  {'Time':>5}  {'Status'}")
    print("-" * 100)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_dl, criterion, optimizer, device, training=True,  scaler=scaler)
        va_loss, va_acc, va_f1 = run_epoch(model, val_dl,   criterion, None,      device, training=False, scaler=None)
        scheduler.step()

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

        improved = va_f1 > best_val_f1 + args.min_delta
        if improved:
            best_val_f1, best_epoch = va_f1, ep
            patience_ctr = 0
            torch.save({
                "epoch": ep, "model": model.state_dict(),
                "val_f1": va_f1, "val_acc": va_acc,
                "classes": CLASSES, "input_size": input_size,
                "train_loss": tr_loss, "val_loss": va_loss,
            }, OUT_CKPT)
            status = f"** BEST (saved) **"
        else:
            patience_ctr += 1
            status = f"patience {patience_ctr}/{args.patience}"

        history.append({
            "epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
        })

        print(f"{ep:>3}  {tr_loss:>7.4f}  {tr_acc:>6.4f}  {tr_f1:>6.4f}  "
              f"{va_loss:>7.4f}  {va_acc:>6.4f}  {va_f1:>6.4f}  "
              f"{cur_lr:>8.2e}  {patience_ctr:>8d}  {elapsed:>4.0f}s  {status}", flush=True)

        # Save learning curve CSV every epoch; regenerate PNG only every 5 epochs
        # to avoid matplotlib overhead blocking training.
        with open(CURVE_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader(); w.writerows(history)
        if ep % 5 == 0 or patience_ctr >= args.patience:
            save_curve(history)

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping: Val F1 has not improved by >{args.min_delta} "
                  f"for {args.patience} consecutive epochs.")
            break

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f} at Epoch {best_epoch}")
    print(f"Learning curve saved to: {CURVE_CSV} | {CURVE_PNG}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\nLoading best checkpoint ({OUT_CKPT.name}) for final test evaluation...")
    best = torch.load(OUT_CKPT, map_location=device)
    model.load_state_dict(best["model"])
    evaluate_test(model, test_dl, device)


if __name__ == "__main__":
    main()
