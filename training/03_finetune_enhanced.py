"""
Enhanced Fine-Tuning Script with Deep Backbone Unfreezing, ANN Multi-Layer Head,
Higher Resolution Textures, and Focal Loss to resolve Vegetative vs Flowering confusion.

Preserves all existing checkpoints and saves separately to:
  checkpoints/checkpoint_enhanced_ann.pt

Usage:
  python 03_finetune_enhanced.py --epochs 10 --batch-size 64 --input-size 288
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
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ── Config ──────────────────────────────────────────────────────────────────
CLASSES = ["Seedling", "Vegetative", "Flowering", "Mature"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
ORDINAL = CLASS_IDX

SPLITS_DIR = Path("splits")
CKPT_DIR = Path("checkpoints")
EVAL_DIR = Path("eval_results")
PREV_CKPT = CKPT_DIR / "checkpoint_round1.pt"
OUT_CKPT = CKPT_DIR / "checkpoint_enhanced_ann.pt"

# ── Focal Loss with Sqrt Class Weights ──────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss down-weights easy examples (Seedling/Mature >99%) and
    concentrates 90% of gradients on the hard Vegetative vs Flowering boundary.
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# ── Enhanced Model with Multi-Layer ANN Head ─────────────────────────────────
class EnhancedRiceClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Load EfficientNet-B0 backbone
        base = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool
        
        in_features = base.classifier[1].in_features # 1280
        
        # Multi-Layer ANN Classification Head (MLP with LayerNorm & Dropout)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(384, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.head(x)
        return x

def set_unfreeze_strategy(model: EnhancedRiceClassifier):
    """
    Freeze lower feature layers (edges/colors),
    Unfreeze top MBConv blocks (6, 7, 8) to learn rice panicle & leaf textures.
    """
    # Freeze lower blocks 0 through 5
    for i in range(6):
        for p in model.features[i].parameters():
            p.requires_grad = False

    # Unfreeze top blocks 6, 7, 8
    for i in range(6, len(model.features)):
        for p in model.features[i].parameters():
            p.requires_grad = True

    # ANN Head is fully trainable
    for p in model.head.parameters():
        p.requires_grad = True

# ── Fast Dataset with Texture Augmentation ──────────────────────────────────
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
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5), # Enhance panicle texture
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
        img = img.convert("RGB")
        return self.tf(img), CLASS_IDX[r["bucket"]]

# ── Training & Evaluation Functions ──────────────────────────────────────────
def compute_class_weights(csv_path: Path) -> torch.Tensor:
    counts = {c: 0 for c in CLASSES}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("bucket") in counts:
                counts[r["bucket"]] += 1
    max_c = max(counts.values()) if counts else 1
    weights = [math.sqrt(max_c / max(counts[c], 1)) for c in CLASSES]
    return torch.tensor(weights, dtype=torch.float32)

def run_epoch(model, loader, criterion, optimizer, device, training, desc=""):
    model.train(training)
    loss_sum, correct, preds_all, labels_all = 0.0, 0, [], []
    total_batches = len(loader)

    with torch.set_grad_enabled(training):
        for b_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            p = logits.argmax(1)
            loss_sum += loss.item() * len(labels)
            correct += (p == labels).sum().item()
            preds_all.extend(p.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == total_batches:
                cur_acc = correct / len(labels_all) if labels_all else 0
                cur_loss = loss_sum / len(labels_all) if labels_all else 0
                prefix = f"  [{desc}]" if desc else ""
                print(f"{prefix} Batch {b_idx + 1:>4}/{total_batches} ({(b_idx+1)/total_batches*100:.0f}%) | Loss: {cur_loss:.4f} | Acc: {cur_acc:.3f}", end="\r" if (b_idx + 1) < total_batches else "\n", flush=True)

    n = len(labels_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    msd = sum(abs(ORDINAL[CLASSES[p]] - ORDINAL[CLASSES[l]]) for p, l in zip(preds_all, labels_all)) / max(n, 1)
    return loss_sum / n, correct / n, float(macro_f1), msd, preds_all, labels_all

def plot_and_report_results(model, test_dl, device, out_dir: Path):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(device)
            p = model(imgs).argmax(1)
            all_preds.extend(p.cpu().tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    msd = sum(abs(ORDINAL[CLASSES[p]] - ORDINAL[CLASSES[l]]) for p, l in zip(all_preds, all_labels)) / len(all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
    p, r, f1, s = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(len(CLASSES))))

    print("\n" + "="*60)
    print("ENHANCED MODEL EVALUATION REPORT: TEST SET (FIELD 2)")
    print("="*60)
    print(f"Overall Accuracy:      {acc*100:.2f}%")
    print(f"Macro F1 Score:        {macro_f1:.4f}")
    print(f"Mean Stage Distance:   {msd:.4f}")
    print("-"*60)
    print(f"{'Class':<14} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-"*60)
    for i, c in enumerate(CLASSES):
        print(f"{c:<14} {p[i]:<10.4f} {r[i]:<10.4f} {f1[i]:<10.4f} {s[i]:<8,}")
    print("="*60)

    # Plot normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=1.0)
    ax.set_title('Normalized Confusion Matrix (Recall %) - ENHANCED MODEL')
    fig.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASSES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASSES)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")

    ax.set_ylabel('True Growth Stage')
    ax.set_xlabel('Predicted Growth Stage')
    plt.tight_layout()
    plot_path = out_dir / "confusion_matrix_enhanced_test.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save Markdown report
    report_file = out_dir / "eval_report_enhanced_test.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Enhanced Rice Model Evaluation Report (Test Split)\n\n")
        f.write(f"- **Architecture**: `EfficientNet-B0 + Multi-Layer ANN Head (Blocks 6-8 Unfrozen)`\n")
        f.write(f"- **Overall Accuracy**: **{acc*100:.2f}%**\n")
        f.write(f"- **Macro F1**: **{macro_f1:.4f}**\n")
        f.write(f"- **Mean Absolute Stage Distance (MSD)**: **{msd:.4f}**\n\n")
        f.write("### Per-Class Performance Table\n\n")
        f.write("| Growth Stage | Precision | Recall | F1-Score | Support (Tiles) |\n")
        f.write("|---|---|---|---|---|\n")
        for i, c in enumerate(CLASSES):
            f.write(f"| **{c}** | {p[i]:.4f} | {r[i]:.4f} | {f1[i]:.4f} | {s[i]:,} |\n")
        f.write(f"\n![Confusion Matrix](confusion_matrix_enhanced_test.png)\n")
    print(f"Report saved to: {report_file.resolve()}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",       type=int, default=10)
    ap.add_argument("--batch-size",   type=int, default=64)
    ap.add_argument("--input-size",   type=int, default=288, help="Higher resolution to capture panicle details")
    ap.add_argument("--lr-head",      type=float, default=2e-4)
    ap.add_argument("--lr-backbone",  type=float, default=1e-5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Resolution: {args.input_size}x{args.input_size}")
    CKPT_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)

    # 1. Datasets & Loaders
    train_ds = FastTileDataset(SPLITS_DIR / "train.csv", args.input_size, is_train=True)
    val_ds   = FastTileDataset(SPLITS_DIR / "val.csv",   args.input_size, is_train=False)
    test_ds  = FastTileDataset(SPLITS_DIR / "test.csv",  args.input_size, is_train=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # 2. Build Enhanced Model & Load Pretrained Backbone
    model = EnhancedRiceClassifier(num_classes=len(CLASSES)).to(device)
    
    # Try loading backbone weights from checkpoint_round1 if available
    if PREV_CKPT.exists():
        print(f"Initializing backbone from {PREV_CKPT.name}...")
        ckpt = torch.load(PREV_CKPT, map_location=device)
        state_dict = ckpt["model"]
        # Load backbone features
        backbone_dict = {k.replace("features.", ""): v for k, v in state_dict.items() if k.startswith("features.")}
        model.features.load_state_dict(backbone_dict, strict=False)
        print("Backbone initialized from Round 1 weights.")

    # Apply unfreezing strategy (Unfreeze blocks 6, 7, 8 + ANN Head)
    set_unfreeze_strategy(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,}")

    # 3. Optimizer with Differential Learning Rates & Focal Loss
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": args.lr_head, "weight_decay": 1e-4},
        {"params": [p for n, p in model.features.named_parameters() if p.requires_grad], "lr": args.lr_backbone, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    class_weights = compute_class_weights(SPLITS_DIR / "train.csv").to(device)
    print(f"Focal Loss Alpha Weights: {[round(w, 4) for w in class_weights.cpu().tolist()]}")
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # 4. Fine-Tuning Loop
    best_val_f1, best_epoch = 0.0, 0
    print(f"\n{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'TrF1':>6}  {'VaLoss':>7}  {'VaAcc':>6}  {'VaF1':>6}  {'VaMSD':>6}  {'Time':>5}")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, _, _, _ = run_epoch(model, train_dl, criterion, optimizer, device, training=True, desc=f"Ep {ep} Train")
        va_loss, va_acc, va_f1, va_msd, _, _ = run_epoch(model, val_dl, criterion, None, device, training=False, desc=f"Ep {ep} Val  ")
        scheduler.step()

        print(f"{ep:>3}  {tr_loss:>7.4f}  {tr_acc:>6.3f}  {tr_f1:>6.3f}  {va_loss:>7.4f}  {va_acc:>6.3f}  {va_f1:>6.3f}  {va_msd:>6.3f}  {time.time()-t0:>4.0f}s")

        if va_f1 > best_val_f1:
            best_val_f1, best_epoch = va_f1, ep
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "val_f1": va_f1,
                "val_acc": va_acc,
                "classes": CLASSES,
                "input_size": args.input_size,
            }, OUT_CKPT)
            print(f"    --> Saved best checkpoint: {OUT_CKPT.name} (Val F1: {va_f1:.4f})")

    print(f"\nTraining Complete! Best Val F1: {best_val_f1:.4f} at Epoch {best_epoch}")

    # 5. Evaluate on Held-Out Field 2 Test Set
    best_ckpt = torch.load(OUT_CKPT, map_location=device)
    model.load_state_dict(best_ckpt["model"])
    plot_and_report_results(model, test_dl, device, EVAL_DIR)

if __name__ == "__main__":
    main()
