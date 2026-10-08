import os
import sys
import math
import time
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CLASSES = ["nursery_stage", "vegetative_stage", "booting_stage", "flowering_stage", "mature_stage"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ResizedImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: Path, input_size: int = 384, is_train: bool = True):
        self.samples = []
        for _, row in df.iterrows():
            if row["growth_stage"] in CLASS_IDX:
                self.samples.append((
                    str(root_dir / row["file_path"]),
                    CLASS_IDX[row["growth_stage"]],
                    str(row["field_plot"])
                ))
        self.input_size = input_size
        t_list = []
        if is_train:
            t_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ])
        t_list.extend([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.tf = transforms.Compose(t_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, flight = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label, flight, path

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

def get_loss_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df["growth_stage"].value_counts()
    c_list = [counts.get(c, 1) for c in CLASSES]
    max_c = max(c_list)
    weights = [math.sqrt(max_c / c) for c in c_list]
    return torch.tensor(weights, dtype=torch.float32)

def build_model(arch: str) -> nn.Module:
    if arch == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, NUM_CLASSES)
    elif arch == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_feat = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feat, NUM_CLASSES)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m

def freeze_backbone(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True

def plot_dual_confusion_matrix(cm_raw: np.ndarray, cm_norm: np.ndarray, title: str, subtitle: str, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.35)
    
    im1 = ax1.imshow(cm_raw, cmap="Blues")
    ax1.set_title(f"{title}\nRaw Sample Counts", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xticks(range(NUM_CLASSES))
    ax1.set_yticks(range(NUM_CLASSES))
    ax1.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax1.set_yticklabels(CLASSES, fontsize=9)
    
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_raw[i, j]
            color = "white" if val > cm_raw.max() * 0.5 else "black"
            ax1.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=9.5, fontweight="bold")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    ax2.set_title(f"{title}\nRow-Normalized Recall (%)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xticks(range(NUM_CLASSES))
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax2.set_yticklabels(CLASSES, fontsize=9)
    
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
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
    if "val_loss" in df.columns and not df["val_loss"].isna().all():
        ax1.plot(epochs, df["val_loss"], "s--", color="#ff7f0e", label="Val Loss", linewidth=2)
    ax1.set_title(f"{title} — Focal Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.plot(epochs, df["train_f1"], "o-", color="#2ca02c", label="Train Macro-F1", linewidth=2)
    if "val_f1" in df.columns and not df["val_f1"].isna().all():
        ax2.plot(epochs, df["val_f1"], "s--", color="#d62728", label="Val Macro-F1", linewidth=2)
        ax2.plot(epochs, df["val_acc"], "^:", color="#9467bd", label="Val Accuracy", linewidth=1.5, alpha=0.7)
    ax2.set_title(f"{title} — Macro-F1 & Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def flight_bootstrap_ci(df: pd.DataFrame, n: int = 2000, seed: int = 0):
    g = df.groupby("field_plot")["correct"].agg(["sum", "count"])
    if len(g) == 0:
        return (np.nan, np.nan)
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    c_sum = c[idx].sum(1)
    s_sum = s[idx].sum(1)
    valid = c_sum > 0
    scores = np.zeros(n)
    scores[valid] = s_sum[valid] / c_sum[valid]
    return np.percentile(scores, [2.5, 97.5])

def train_model(args, arch: str, device: torch.device) -> Dict:
    print(f"\n{'='*80}\n TRAINING {arch.upper()} (RESIZE 384x384) SEED {args.seed}\n{'='*80}")
    
    df = pd.read_csv(args.split_csv)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    run_val = len(val_df) > 0

    root_dir = Path(args.root)
    train_ds = ResizedImageDataset(train_df, root_dir, input_size=384, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    if run_val:
        val_ds = ResizedImageDataset(val_df, root_dir, input_size=384, is_train=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = build_model(arch).to(device)
    weights = get_loss_weights(train_df).to(device)
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    scaler = torch.amp.GradScaler('cuda')

    param_cnt = sum(p.numel() for p in model.parameters())

    # Warm-up epoch
    freeze_backbone(model)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)

    curve_rows = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()

        if epoch == 2:
            unfreeze_all(model)
            backbone_p = list(model.features.parameters())
            head_p = list(model.classifier.parameters())
            optimizer = optim.AdamW([
                {"params": backbone_p, "lr": 1e-4},
                {"params": head_p, "lr": 5e-4},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 1, eta_min=1e-6)

        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for imgs, labels, _, _ in train_loader:
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
        
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": np.nan, "val_acc": np.nan, "val_f1": np.nan
        }

        if run_val:
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for imgs, labels, _, _ in val_loader:
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
            row.update({"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
            
            print(f" Ep {epoch:2d}/{args.epochs:2d} | TrLoss: {train_loss:.4f} TrAcc: {train_acc*100:5.2f}% TrF1: {train_f1:.4f} | VaLoss: {val_loss:.4f} VaAcc: {val_acc*100:5.2f}% VaF1: {val_f1:.4f}")
        else:
            print(f" Ep {epoch:2d}/{args.epochs:2d} | TrLoss: {train_loss:.4f} TrAcc: {train_acc*100:5.2f}% TrF1: {train_f1:.4f}")

        row["epoch_time_s"] = time.time() - ep_t0
        curve_rows.append(row)

    total_time = time.time() - t_start
    print(f" Total training time: {total_time:.1f}s")
    
    # Save final epoch weights (no val checkpointing)
    run_name = Path(args.split_csv).stem
    out_dir = Path(args.out_dir)
    ckpts_dir = out_dir / "checkpoints"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(curve_rows).to_csv(curves_dir / f"{arch}_{run_name}_s{args.seed}_curve.csv", index=False)
    plot_learning_curve(curve_rows, f"{arch.upper()} {run_name} (Seed {args.seed})", curves_dir / f"{arch}_{run_name}_s{args.seed}_curve.png")

    ckpt_path = ckpts_dir / f"{arch}_{run_name}_s{args.seed}_final.pt"
    torch.save({
        "arch": arch,
        "method": "resize",
        "resolution": 384,
        "model_state": model.state_dict(),
        "seed": args.seed
    }, ckpt_path)

    return model, total_time, param_cnt

def evaluate_model(args, model: nn.Module, arch: str, total_time: float, param_cnt: int, device: torch.device):
    df = pd.read_csv(args.split_csv)
    test_df = df[df["split"] == "test"]
    if len(test_df) == 0:
        print("No test data found, skipping evaluation.")
        return None
        
    print(f"\nEvaluating {arch.upper()} on Test Set ({len(test_df)} images)...")
    root_dir = Path(args.root)
    test_ds = ResizedImageDataset(test_df, root_dir, input_size=384, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model.eval()
    preds, targets, flights, files, probs = [], [], [], [], []

    with torch.no_grad():
        for imgs, labels, batch_flights, batch_paths in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                p = F.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
            flights.extend(batch_flights)
            files.extend(batch_paths)
            probs.extend(p.cpu().tolist())

    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    cm_raw = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis])
        
    run_name = Path(args.split_csv).stem
    out_dir = Path(args.out_dir)
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    cm_path = cm_dir / f"{arch}_{run_name}_s{args.seed}_confusion_matrix.png"
    plot_dual_confusion_matrix(
        cm_raw, cm_norm,
        f"{arch.upper()} — Resize 384px",
        f"Test Set (N={len(targets)}, Acc: {acc*100:.2f}%, Macro-F1: {macro_f1:.4f})",
        cm_path
    )

    # Flight-level reporting
    res_df = pd.DataFrame({"field_plot": flights, "target": targets, "pred": preds})
    res_df["correct"] = (res_df["target"] == res_df["pred"]).astype(int)
    
    # Check if control split (flights in both train/test)
    is_control = "control" in run_name
    ci_lower, ci_upper = np.nan, np.nan
    if not is_control:
        ci_lower, ci_upper = flight_bootstrap_ci(res_df, seed=args.seed)
        ci_str = f"[{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]"
    else:
        ci_str = "N/A (Control)"
        
    print(f"\n  Test Overall: Acc={acc*100:.2f}% {ci_str} | Macro-F1={macro_f1:.4f}")
    
    print("\n  Per-flight accuracy:")
    flight_g = res_df.groupby("field_plot")["correct"].agg(["sum", "count"])
    for f, row in flight_g.iterrows():
        print(f"    {f}: {row['sum']}/{row['count']} ({(row['sum']/row['count'])*100:.1f}%)")
        
    # Save per-image predictions CSV
    csv_out_path = out_dir / f"{arch}_{run_name}_s{args.seed}_predictions.csv"
    import csv
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_path", "field_plot", "true", "pred"] + [f"p_{c}" for c in CLASSES])
        for file_path, flight, tgt, prd, prob in zip(files, flights, targets, preds, probs):
            w.writerow([file_path, flight, CLASSES[tgt], CLASSES[prd]] + [f"{x:.5f}" for x in prob])

    entry = {
        "arch": arch,
        "method": "resize",
        "split_name": run_name,
        "seed": args.seed,
        "param_count": param_cnt,
        "training_time": total_time,
        "overall_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "ci_lower": float(ci_lower) if not np.isnan(ci_lower) else None,
        "ci_upper": float(ci_upper) if not np.isnan(ci_upper) else None,
    }
    return entry

def update_json(out_dir: Path, new_entries: List[Dict]):
    eval_json = out_dir / "evaluation_results.json"
    data = []
    if eval_json.exists():
        with open(eval_json, "r") as f:
            data = json.load(f)
            
    for entry in new_entries:
        match_idx = next((i for i, item in enumerate(data) 
                          if item["arch"] == entry["arch"] and 
                             item.get("split_name") == entry["split_name"] and
                             item.get("seed", 42) == entry["seed"]), None)
        if match_idx is not None:
            data[match_idx] = entry
        else:
            data.append(entry)
            
    with open(eval_json, "w") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--root", default=r"e:\paddy_test\test_500_rgb_images")
    parser.add_argument("--out-dir", default=r"e:\paddy_test\training\full_dataset_b2_b3\resizing_384")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--arch", default="efficientnet_b2", help="efficientnet_b2 or efficientnet_b3 or all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    models_to_train = ["efficientnet_b2", "efficientnet_b3"] if args.arch == "all" else [args.arch]
    results = []
    for arch in models_to_train:
        model, t_time, p_cnt = train_model(args, arch, device)
        entry = evaluate_model(args, model, arch, t_time, p_cnt, device)
        if entry:
            results.append(entry)

    update_json(Path(args.out_dir), results)
    print("\nResizing runs completed!")

if __name__ == "__main__":
    main()
