"""
Comprehensive evaluation script for rice paddy growth-stage classifier checkpoints.
Generates:
1. Confusion Matrix heatmap PNG (raw counts & normalized)
2. Per-class metrics table (Precision, Recall, F1, Support)
3. Ordinal error metrics (Mean Absolute Stage Distance)
4. Evaluation summary report (markdown / text)

Usage:
  python 02_evaluate.py --checkpoint checkpoints/checkpoint_round1.pt --split test
  python 02_evaluate.py --checkpoint checkpoints/checkpoint_round1.pt --split val
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["Seedling", "Vegetative", "Flowering", "Mature"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
ORDINAL = CLASS_IDX

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class EvalDataset(Dataset):
    def __init__(self, csv_path: Path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r.get("bucket") in CLASS_IDX]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["tile_file_path"])
        img.draft("RGB", (224, 224))
        img = img.convert("RGB")
        return EVAL_TF(img), CLASS_IDX[r["bucket"]], r["tile_file_path"]


class EnhancedRiceClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.efficientnet_b0()
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[1].in_features
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
        return self.head(self.avgpool(self.features(x)))

def build_model_for_ckpt(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"]
    input_size = ckpt.get("input_size", 224)
    
    if any(k.startswith("head.") for k in state_dict.keys()):
        model = EnhancedRiceClassifier(len(CLASSES))
    else:
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
        
    model.load_state_dict(state_dict)
    return model, input_size


def plot_confusion_matrices(cm, out_dir: Path, split_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title(f'Confusion Matrix (Counts) - {split_name.upper()}')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASSES))
    ax1.set_xticks(tick_marks)
    ax1.set_xticklabels(CLASSES, rotation=30)
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(CLASSES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, f"{cm[i, j]:,}",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    ax1.set_ylabel('True Growth Stage')
    ax1.set_xlabel('Predicted Growth Stage')

    # Normalized (% of True Class)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=1.0)
    ax2.set_title(f'Normalized Confusion Matrix (Recall %) - {split_name.upper()}')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(CLASSES, rotation=30)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(CLASSES)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(j, i, f"{cm_norm[i, j]*100:.1f}%",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")

    ax2.set_ylabel('True Growth Stage')
    ax2.set_xlabel('Predicted Growth Stage')

    plt.tight_layout()
    plot_path = out_dir / f"confusion_matrix_{split_name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix heatmap saved: {plot_path.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test", "train"], default="test")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--splits-dir", type=Path, default=Path("splits"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(exist_ok=True)

    # Load Model & Input Size
    model, input_size = build_model_for_ckpt(args.checkpoint)
    model = model.to(device)
    model.eval()

    is_enhanced = isinstance(model, EnhancedRiceClassifier)
    prefix = "enhanced_" if is_enhanced else ""
    print(f"Evaluating {args.checkpoint.name} ({'Enhanced ANN' if is_enhanced else 'Baseline Linear'}) at {input_size}x{input_size} on {args.split} split using {device}...")

    # Load Split Dataset with correct resolution
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    split_csv = args.splits_dir / f"{args.split}.csv"
    with open(split_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("bucket") in CLASS_IDX]

    class DynamicDataset(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, idx):
            r = rows[idx]
            img = Image.open(r["tile_file_path"])
            img.draft("RGB", (input_size, input_size))
            img = img.convert("RGB")
            return tf(img), CLASS_IDX[r["bucket"]]

    dl = DataLoader(DynamicDataset(), batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dl:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    msd = sum(abs(ORDINAL[CLASSES[p]] - ORDINAL[CLASSES[l]]) for p, l in zip(all_preds, all_labels)) / len(all_preds)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
    p, r, f1, s = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(len(CLASSES))))

    print("\n" + "="*60)
    print(f"EVALUATION REPORT: {args.split.upper()} SET ({'ENHANCED MODEL' if is_enhanced else 'BASELINE MODEL'})")
    print("="*60)
    print(f"Total Tiles Evaluated: {len(all_labels):,}")
    print(f"Accuracy:              {acc*100:.2f}%")
    print(f"Macro F1 Score:        {macro_f1:.4f}")
    print(f"Weighted F1 Score:     {weighted_f1:.4f}")
    print(f"Mean Stage Distance:   {msd:.4f} (Ordinal stage hops error)")
    print("-"*60)
    print(f"{'Class':<14} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-"*60)
    for i, c in enumerate(CLASSES):
        print(f"{c:<14} {p[i]:<10.4f} {r[i]:<10.4f} {f1[i]:<10.4f} {s[i]:<8,}")
    print("="*60)

    # Plot & save confusion matrix
    plot_confusion_matrices(cm, args.out_dir, f"{prefix}{args.split}")

    # Save detailed markdown report
    report_file = args.out_dir / f"eval_report_{prefix}{args.split}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Model Evaluation Report ({args.split.upper()} Split - {'ENHANCED MODEL' if is_enhanced else 'BASELINE'})\n\n")
        f.write(f"- **Checkpoint**: `{args.checkpoint.name}`\n")
        f.write(f"- **Overall Accuracy**: **{acc*100:.2f}%**\n")
        f.write(f"- **Macro F1**: **{macro_f1:.4f}**\n")
        f.write(f"- **Weighted F1**: **{weighted_f1:.4f}**\n")
        f.write(f"- **Mean Absolute Stage Distance (MSD)**: **{msd:.4f}**\n\n")
        f.write("### Per-Class Performance Table\n\n")
        f.write("| Growth Stage | Precision | Recall | F1-Score | Support (Tiles) |\n")
        f.write("|---|---|---|---|---|\n")
        for i, c in enumerate(CLASSES):
            f.write(f"| **{c}** | {p[i]:.4f} | {r[i]:.4f} | {f1[i]:.4f} | {s[i]:,} |\n")
        f.write(f"\n![Confusion Matrix](confusion_matrix_{prefix}{args.split}.png)\n")
    print(f"Markdown report saved: {report_file.resolve()}")


if __name__ == "__main__":
    main()
