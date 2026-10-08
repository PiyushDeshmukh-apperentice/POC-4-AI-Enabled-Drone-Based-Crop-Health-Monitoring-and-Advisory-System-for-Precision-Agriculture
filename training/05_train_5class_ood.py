"""
05_train_5class_ood.py -- 5-Class fine-tuning with 'Not_Rice' negative class.

Extends the 4-class tile model (checkpoint_full_finetune.pt) to 5 classes:
  0: Seedling, 1: Vegetative, 2: Flowering, 3: Mature, 4: Not_Rice

Features:
  - Head expansion preserving learned 4-class weights.
  - WeightedRandomSampler to balance 144 not_rice tiles with 33k rice tiles (~6-8% batch presence).
  - 2-Phase schedule:
      Phase 1 (Warm-up, 2 epochs): Train head only to establish 5th class prototype.
      Phase 2 (Fine-tune, up to 15 epochs): Full unfreeze with Cosine Annealing.
  - Full 5-class confusion matrix on Field 2 holdout + held-out OOD test set.
  - Generates learning curves and OOD tier rejection breakdown.
"""

import argparse
import csv
import json
import math
import random
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

# ── Config ───────────────────────────────────────────────────────────────────
CLASSES   = ["Seedling", "Vegetative", "Flowering", "Mature", "Not_Rice"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

SPLITS_DIR   = Path("splits")
CKPT_DIR     = Path("checkpoints")
EVAL_DIR     = Path("eval_results")
NOT_RICE_DIR = Path("../not_rice_class")

RESUME_CKPT = CKPT_DIR / "checkpoint_full_finetune.pt"
OUT_CKPT    = CKPT_DIR / "checkpoint_5class_finetune.pt"
CURVE_CSV   = EVAL_DIR / "learning_curve_5class.csv"
CURVE_PNG   = EVAL_DIR / "learning_curve_5class.png"

# ── Model ────────────────────────────────────────────────────────────────────
class EnhancedRiceClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base = models.efficientnet_b0()
        self.features = base.features
        self.avgpool  = base.avgpool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 384), nn.LayerNorm(384), nn.SiLU(), nn.Dropout(0.35),
            nn.Linear(384, 128),  nn.LayerNorm(128),  nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.avgpool(self.features(x)))


def expand_head_from_4class(model_5class, ckpt_4class_path, device):
    """Load 4-class weights into 5-class model, initializing the 5th class cleanly."""
    ckpt = torch.load(ckpt_4class_path, map_location=device)
    state_4 = ckpt["model"]

    # Filter out the final linear layer (head.9.weight and head.9.bias)
    compatible_state = {k: v for k, v in state_4.items() if not k.startswith("head.9")}
    model_5class.load_state_dict(compatible_state, strict=False)

    old_w = state_4["head.9.weight"]  # [4, 128]
    old_b = state_4["head.9.bias"]    # [4]

    with torch.no_grad():
        model_5class.head[-1].weight[:4] = old_w
        model_5class.head[-1].bias[:4]   = old_b
        # Initialize 5th class (Not_Rice)
        nn.init.normal_(model_5class.head[-1].weight[4:], mean=0.0, std=0.01)
        nn.init.zeros_(model_5class.head[-1].bias[4:])

    print("Transferred 4-class weights to 5-class model (head expanded).")


# ── Loss ─────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha[targets] * ((1 - pt) ** self.gamma) * ce).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────
class FiveClassDataset(Dataset):
    def __init__(self, sample_list, input_size: int, is_train: bool):
        self.samples = sample_list  # list of (filepath, class_idx)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label_idx = self.samples[idx]
        img = Image.open(fpath)
        img.draft("RGB", (self.input_size, self.input_size))
        return self.tf(img.convert("RGB")), label_idx


def build_sample_lists():
    """Build train, val, and test lists combining rice splits with not_rice tiles."""
    # 1. Load rice CSVs
    rice_train, rice_val, rice_test = [], [], []
    with open(SPLITS_DIR / "train.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("bucket") in CLASS_IDX:
                rice_train.append((Path(r["tile_file_path"]), CLASS_IDX[r["bucket"]]))

    with open(SPLITS_DIR / "val.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("bucket") in CLASS_IDX:
                rice_val.append((Path(r["tile_file_path"]), CLASS_IDX[r["bucket"]]))

    with open(SPLITS_DIR / "test.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("bucket") in CLASS_IDX:
                rice_test.append((Path(r["tile_file_path"]), CLASS_IDX[r["bucket"]]))

    # 2. Load not_rice manifest
    manifest_path = NOT_RICE_DIR / "not_rice_manifest.json"
    not_rice_train_all, not_rice_test = [], []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        for m in manifest:
            img_path = NOT_RICE_DIR / m["file"]
            if img_path.exists():
                if m["split"] == "train":
                    not_rice_train_all.append((img_path, CLASS_IDX["Not_Rice"]))
                else:
                    not_rice_test.append((img_path, CLASS_IDX["Not_Rice"]))

    # Split not_rice train into 80% train / 20% val for validation checking
    random.seed(42)
    random.shuffle(not_rice_train_all)
    n_val = max(10, int(len(not_rice_train_all) * 0.2))
    not_rice_val = not_rice_train_all[:n_val]
    not_rice_train = not_rice_train_all[n_val:]

    train_samples = rice_train + not_rice_train
    val_samples   = rice_val + not_rice_val
    test_samples  = rice_test + not_rice_test

    print(f"5-Class Samples:")
    print(f"  Train: {len(train_samples):,} (Rice: {len(rice_train):,}, Not_Rice: {len(not_rice_train)})")
    print(f"  Val  : {len(val_samples):,} (Rice: {len(rice_val):,}, Not_Rice: {len(not_rice_val)})")
    print(f"  Test : {len(test_samples):,} (Rice: {len(rice_test):,}, Not_Rice: {len(not_rice_test)})")

    return train_samples, val_samples, test_samples


def build_sampler(train_samples, target_not_rice_prob=0.07):
    """Create WeightedRandomSampler so Not_Rice appears in ~7% of batch items."""
    targets = [s[1] for s in train_samples]
    n_total = len(targets)
    n_not_rice = sum(1 for t in targets if t == CLASS_IDX["Not_Rice"])
    n_rice = n_total - n_not_rice

    w_rice = (1.0 - target_not_rice_prob) / n_rice
    w_not_rice = target_not_rice_prob / n_not_rice

    weights = [w_not_rice if t == CLASS_IDX["Not_Rice"] else w_rice for t in targets]
    sampler = WeightedRandomSampler(weights, num_samples=n_total, replacement=True)
    print(f"WeightedRandomSampler: Not_Rice upweighted to ~{target_not_rice_prob*100:.1f}% per batch.")
    return sampler


# ── Epoch Runner ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training, scaler=None):
    model.train(training)
    loss_sum, correct, preds_all, labels_all = 0.0, 0, [], []
    total = len(loader)

    with torch.set_grad_enabled(training):
        for b, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_test(model, test_dl, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            p = model(imgs.to(device)).argmax(1)
            preds_all.extend(p.cpu().tolist())
            labels_all.extend(labels.tolist())

    acc      = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    cm       = confusion_matrix(labels_all, preds_all, labels=list(range(len(CLASSES))))
    pr, rc, f1s, sup = precision_recall_fscore_support(labels_all, preds_all,
                                                        labels=list(range(len(CLASSES))), zero_division=0)

    print("\n" + "="*75)
    print("5-CLASS TILE MODEL  |  TEST SET (HELD-OUT FIELD 2 + OOD TEST)")
    print("="*75)
    print(f"Overall Accuracy : {acc*100:.2f}%")
    print(f"Macro F1 Score   : {macro_f1:.4f}")
    print("-"*75)
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-"*75)
    for i, c in enumerate(CLASSES):
        print(f"{c:<14} {pr[i]:>10.4f} {rc[i]:>10.4f} {f1s[i]:>10.4f} {sup[i]:>10,}")
    print("="*75)

    print("\nConfusion Matrix (rows=True, cols=Pred):")
    header = f"{'':>14}" + "".join(f"{c:>14}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>14}" + "".join(f"{v:>14,}" for v in row))

    # Save normalized confusion matrix
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("5-Class Model — Normalized Confusion Matrix")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm_norm[i,j]*100:.1f}%\n({cm[i,j]})", ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "confusion_matrix_5class.png", dpi=300)
    plt.close()
    print(f"\nConfusion matrix saved to: {EVAL_DIR / 'confusion_matrix_5class.png'}")
    return acc, macro_f1


def save_curve(history):
    epochs   = [h["epoch"] for h in history]
    tr_loss  = [h["train_loss"] for h in history]
    va_loss  = [h["val_loss"]   for h in history]
    tr_f1    = [h["train_f1"]   for h in history]
    va_f1    = [h["val_f1"]     for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, tr_loss, "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, va_loss, "r-s", label="Val Loss", markersize=4)
    axes[0].set_title("Loss per Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, tr_f1, "b-o", label="Train Macro F1", markersize=4)
    axes[1].plot(epochs, va_f1, "r-s", label="Val Macro F1", markersize=4)
    axes[1].set_title("Macro F1 per Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0.85, 1.01)

    plt.suptitle("5-Class Fine-Tuning Learning Curve", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(CURVE_PNG, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-epochs", type=int,   default=2,    help="Phase 1: Head-only warm-up epochs")
    ap.add_argument("--finetune-epochs", type=int, default=10,   help="Phase 2: Full finetune epochs")
    ap.add_argument("--patience",        type=int, default=4,    help="Early-stop patience")
    ap.add_argument("--batch-size",      type=int, default=32)
    ap.add_argument("--lr-head",         type=float, default=1e-4)
    ap.add_argument("--lr-backbone",     type=float, default=5e-6)
    ap.add_argument("--num-workers",     type=int,   default=0,    help="DataLoader workers (0 recommended on Windows to prevent WinError 1455)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    CKPT_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)

    input_size = 288
    train_samples, val_samples, test_samples = build_sample_lists()

    # Datasets & Loaders
    train_ds = FiveClassDataset(train_samples, input_size, is_train=True)
    val_ds   = FiveClassDataset(val_samples,   input_size, is_train=False)
    test_ds  = FiveClassDataset(test_samples,  input_size, is_train=False)

    sampler = build_sampler(train_samples, target_not_rice_prob=0.07)
    _nw = args.num_workers
    _pm = device.type == "cuda"

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=_nw, pin_memory=_pm)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,   num_workers=_nw, pin_memory=_pm)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,   num_workers=_nw, pin_memory=_pm)

    # Initialize 5-class model and load from checkpoint_full_finetune.pt
    model = EnhancedRiceClassifier(num_classes=5).to(device)
    if RESUME_CKPT.exists():
        expand_head_from_4class(model, RESUME_CKPT, device)
    else:
        raise FileNotFoundError(f"Base 4-class checkpoint not found: {RESUME_CKPT}")

    # Loss: moderate focal weights
    alpha = torch.tensor([1.2, 1.2, 1.0, 1.2, 2.0], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    scaler = torch.amp.GradScaler(device=device.type) if device.type == "cuda" else None

    # ── Phase 1: Head-Only Warm-up ────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: HEAD-ONLY WARM-UP (Backbone Frozen)")
    print("="*60)
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    opt_phase1 = torch.optim.AdamW(model.head.parameters(), lr=args.lr_head, weight_decay=1e-4)

    for ep in range(1, args.warmup_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_dl, criterion, opt_phase1, device, training=True, scaler=scaler)
        va_loss, va_acc, va_f1 = run_epoch(model, val_dl,   criterion, None,       device, training=False, scaler=None)
        elapsed = time.time() - t0
        print(f"Warm-Up Ep {ep}/{args.warmup_epochs} | TrLoss={tr_loss:.4f} TrF1={tr_f1:.4f} | VaLoss={va_loss:.4f} VaF1={va_f1:.4f} | {elapsed:.0f}s")

    # Clean up Phase 1 optimizer memory before full unfreeze
    del opt_phase1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Phase 2: Full Fine-Tune ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: FULL UNFREEZE FINE-TUNE")
    print("="*60)
    for p in model.parameters():
        p.requires_grad = True

    opt_phase2 = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": args.lr_head * 0.5, "weight_decay": 1e-4},
        {"params": model.features.parameters(), "lr": args.lr_backbone, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_phase2, T_max=args.finetune_epochs, eta_min=1e-7)

    best_val_f1 = va_f1
    best_epoch = 0
    patience_ctr = 0
    history = []

    print(f"{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'TrF1':>6}  {'VaLoss':>7}  {'VaAcc':>6}  {'VaF1':>6}  {'Patience':>8}  {'Time':>5}  {'Status'}")
    print("-" * 90)

    for ep in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_dl, criterion, opt_phase2, device, training=True, scaler=scaler)
        va_loss, va_acc, va_f1 = run_epoch(model, val_dl,   criterion, None,       device, training=False, scaler=None)
        scheduler.step()
        elapsed = time.time() - t0

        improved = va_f1 > best_val_f1 + 1e-4
        if improved:
            best_val_f1, best_epoch = va_f1, ep
            patience_ctr = 0
            torch.save({
                "epoch": ep, "model": model.state_dict(),
                "val_f1": va_f1, "val_acc": va_acc,
                "classes": CLASSES, "input_size": input_size,
            }, OUT_CKPT)
            status = "** BEST (saved) **"
        else:
            patience_ctr += 1
            status = f"patience {patience_ctr}/{args.patience}"

        history.append({
            "epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,
        })
        print(f"{ep:>3}  {tr_loss:>7.4f}  {tr_acc:>6.4f}  {tr_f1:>6.4f}  {va_loss:>7.4f}  {va_acc:>6.4f}  {va_f1:>6.4f}  {patience_ctr:>8d}  {elapsed:>4.0f}s  {status}")

        if ep % 3 == 0 or patience_ctr >= args.patience:
            with open(CURVE_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                w.writeheader(); w.writerows(history)
            save_curve(history)

        if patience_ctr >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
            break

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    print(f"\nLoading best checkpoint ({OUT_CKPT.name}) for evaluation on held-out test set...")
    if OUT_CKPT.exists():
        best = torch.load(OUT_CKPT, map_location=device)
        model.load_state_dict(best["model"])
    evaluate_test(model, test_dl, device)


if __name__ == "__main__":
    main()
