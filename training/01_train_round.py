"""
Round-based incremental training for rice paddy growth-stage classifier.

Usage:
  # First time (creates fixed val/test splits):
  python 00_make_splits.py

  # Each round as new data arrives:
  python 01_train_round.py --round 1
  python 01_train_round.py --round 2 --replay-ratio 0.4
  python 01_train_round.py --round 3 --replay-ratio 0.4
"""
import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ── Constants ────────────────────────────────────────────────────────────────
CLASSES   = ["Seedling", "Vegetative", "Flowering", "Mature"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
ORDINAL   = CLASS_IDX

BUCKET_MAP = {
    "Nursery_Stage":    "Seedling",
    "Vegetative_Stage": "Vegetative",
    "Booting_Stage":    "Flowering",
    "Flowering_Stage":  "Flowering",
    "Mature_Stage":     "Mature",
    "Mature_stage":     "Mature",
}

MANIFEST   = Path("../manifest.csv")
SPLITS_DIR = Path("splits")
CKPT_DIR   = Path("checkpoints")
STATE_FILE = CKPT_DIR / "round_state.json"
LOG_FILE   = CKPT_DIR / "training_log.csv"

TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ──────────────────────────────────────────────────────────────────
class TileDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        img.draft("RGB", (224, 224))
        img = img.convert("RGB")
        return self.transform(img), label


# ── Manifest helpers ─────────────────────────────────────────────────────────
def load_manifest_tiles(manifest_path: Path) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("tile_file_path")]

def tile_to_sample(row: dict) -> tuple[str, int] | None:
    bucket = BUCKET_MAP.get(row["growth_stage"])
    if bucket not in CLASS_IDX:
        return None
    return row["tile_file_path"], CLASS_IDX[bucket]

def get_field(row: dict, source_map: dict) -> str:
    src = source_map.get(row.get("source_capture_id", ""))
    if src:
        fp = src.get("field_plot", "")
        if fp.startswith("F1"): return "F1"
        if fp.startswith("F2"): return "F2"
    return "unknown"

def compute_class_weights(samples: list[tuple[str, int]]) -> torch.Tensor:
    counts = defaultdict(int)
    for _, label in samples:
        counts[label] += 1
    max_c = max(counts.values()) if counts else 1
    weights = [math.sqrt(max_c / max(counts.get(i, 1), 1)) for i in range(len(CLASSES))]
    return torch.tensor(weights, dtype=torch.float32)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m

def set_freeze(model: nn.Module, round_num: int):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if round_num == 1:
        pass  # head only
    elif round_num == 2:
        for p in model.features[8].parameters(): p.requires_grad = True
    elif round_num == 3:
        for blk in [7, 8]:
            for p in model.features[blk].parameters(): p.requires_grad = True
    else:
        for p in model.parameters(): p.requires_grad = True

def make_optimizer(model: nn.Module, round_num: int):
    head_lr = 1e-3 if round_num == 1 else 1e-4
    backbone_lr = 0 if round_num == 1 else 1e-5

    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(groups), head_lr, backbone_lr

# ── Metrics ───────────────────────────────────────────────────────────────────
def mean_stage_dist(preds: list, labels: list) -> float:
    return sum(abs(ORDINAL[CLASSES[p]] - ORDINAL[CLASSES[l]])
               for p, l in zip(preds, labels)) / max(len(preds), 1)

def per_class_f1(preds, labels) -> dict:
    scores = f1_score(labels, preds, average=None, labels=list(range(len(CLASSES))), zero_division=0)
    return {CLASSES[i]: round(float(scores[i]), 4) for i in range(len(CLASSES))}

# ── Train / eval ──────────────────────────────────────────────────────────────
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
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            p = logits.argmax(1)
            loss_sum += loss.item() * len(labels)
            correct  += (p == labels).sum().item()
            preds_all.extend(p.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == total_batches:
                cur_acc = correct / len(labels_all) if labels_all else 0
                cur_loss = loss_sum / len(labels_all) if labels_all else 0
                prefix = f"  [{desc}]" if desc else ""
                print(f"{prefix} Batch {b_idx + 1:>4}/{total_batches} ({(b_idx+1)/total_batches*100:.0f}%) | Loss: {cur_loss:.4f} | Acc: {cur_acc:.3f}", end="\r" if (b_idx + 1) < total_batches else "\n", flush=True)


    n = len(labels_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    msd = mean_stage_dist(preds_all, labels_all)
    return loss_sum / n, correct / n, float(macro_f1), msd, preds_all, labels_all


# ── Round state ───────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"rounds": [], "best_val_f1": 0.0, "best_round": 0,
            "used_capture_ids": []}

def save_state(state: dict):
    CKPT_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))

def log_round(row: dict):
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new: w.writeheader()
        w.writerow(row)

def print_comparison(state: dict, current: dict):
    rounds = state["rounds"] + [current]
    print(f"\n{'Round':>5}  {'NewTiles':>9}  {'Replay':>7}  {'ValAcc':>7}  "
          f"{'ValF1':>7}  {'ValMSD':>7}  {'Best?':>6}")
    prev_f1 = 0.0
    for r in rounds:
        flag = "✓ BEST" if r["val_f1"] >= state["best_val_f1"] else \
               ("↓ DROP" if r["val_f1"] < prev_f1 else "")
        print(f"{r['round']:>5}  {r['new_tiles']:>9}  {r['replay_tiles']:>7}  "
              f"{r['val_acc']:>7.4f}  {r['val_f1']:>7.4f}  {r['val_msd']:>7.4f}  {flag}")
        prev_f1 = r["val_f1"]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round",        type=int,   required=True)
    ap.add_argument("--epochs",       type=int,   default=8)
    ap.add_argument("--replay-ratio", type=float, default=0.4,
                    help="Fraction of previous tiles to replay (round 2+)")
    ap.add_argument("--batch-size",   type=int,   default=64)
    ap.add_argument("--workers",      type=int,   default=0)
    ap.add_argument("--force",        action="store_true",
                    help="Save checkpoint even if val F1 drops")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Round: {args.round}")
    CKPT_DIR.mkdir(exist_ok=True)

    # ── Load state & manifest ──
    state = load_state()
    used_caps = set(state["used_capture_ids"])

    all_tile_rows = load_manifest_tiles(MANIFEST)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    source_map = {r["capture_id"]: r for r in all_rows
                  if not r.get("tile_file_path") and r.get("capture_id")}

    # Split F1 (train pool) vs F2 (test, already in splits/test.csv)
    f1_tiles = [r for r in all_tile_rows if get_field(r, source_map) == "F1"]

    # ── Fixed val set (created once by 00_make_splits.py) ──
    if not (SPLITS_DIR / "val.csv").exists():
        raise FileNotFoundError("splits/val.csv not found — run 00_make_splits.py first")
    with open(SPLITS_DIR / "val.csv", newline="", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    val_samples = [s for r in val_rows if (s := tile_to_sample(r))]

    # ── Build train samples ──
    new_tiles  = [r for r in f1_tiles if r.get("source_capture_id") not in used_caps
                  and tile_to_sample(r)]
    prev_tiles = [r for r in f1_tiles if r.get("source_capture_id") in used_caps
                  and tile_to_sample(r)]

    if not new_tiles:
        print("No new tiles found beyond previous rounds. Nothing to train.")
        return

    new_samples  = [tile_to_sample(r) for r in new_tiles]
    prev_samples = [tile_to_sample(r) for r in prev_tiles]

    replay_count = int(len(prev_samples) * args.replay_ratio) if prev_samples else 0
    replay_samples = random.sample(prev_samples, min(replay_count, len(prev_samples)))
    train_samples = new_samples + replay_samples
    random.shuffle(train_samples)

    print(f"New tiles:    {len(new_samples)}")
    print(f"Replay tiles: {len(replay_samples)} ({args.replay_ratio:.0%} of {len(prev_samples)} prev)")
    print(f"Val tiles:    {len(val_samples)}")

    # ── Class weights from current cumulative distribution ──
    all_train_so_far = prev_samples + new_samples
    class_weights = compute_class_weights(all_train_so_far).to(device)
    print(f"Class weights: {dict(zip(CLASSES, [round(w, 4) for w in class_weights.cpu().tolist()]))}")

    # ── DataLoaders ──
    train_dl = DataLoader(TileDataset(train_samples, TRAIN_TF),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=False)
    val_dl   = DataLoader(TileDataset(val_samples, VAL_TF),
                          batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=False)

    # ── Model: load checkpoint or init ──
    model = build_model().to(device)
    prev_ckpt = CKPT_DIR / f"checkpoint_round{args.round - 1}.pt"
    if args.round > 1 and prev_ckpt.exists():
        ckpt = torch.load(prev_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded: {prev_ckpt.name}  (prev val_f1={ckpt.get('val_f1', '?'):.4f})")
    else:
        print("Starting from ImageNet weights (round 1)")

    set_freeze(model, args.round)
    optimizer, head_lr, backbone_lr = make_optimizer(model, args.round)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  "
          f"(LR head={head_lr}, backbone={backbone_lr})")

    # ── Training loop ──
    best_val_f1 = state["best_val_f1"]
    best_epoch_f1, best_epoch = 0.0, 0

    print(f"\n{'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'TrF1':>6}  "
          f"{'VaLoss':>7}  {'VaAcc':>6}  {'VaF1':>6}  {'VaMSD':>6}  {'t':>5}")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, _, _, _ = run_epoch(
            model, train_dl, criterion, optimizer, device, training=True, desc=f"Ep {ep} Train")
        va_loss, va_acc, va_f1, va_msd, va_preds, va_labels = run_epoch(
            model, val_dl, criterion, None, device, training=False, desc=f"Ep {ep} Val  ")
        scheduler.step()


        print(f"{ep:>3}  {tr_loss:>7.4f}  {tr_acc:>6.3f}  {tr_f1:>6.3f}  "
              f"{va_loss:>7.4f}  {va_acc:>6.3f}  {va_f1:>6.3f}  {va_msd:>6.3f}  "
              f"{time.time()-t0:>4.0f}s")

        if va_f1 > best_epoch_f1:
            best_epoch_f1, best_epoch = va_f1, ep
            best_va_acc, best_va_msd = va_acc, va_msd
            best_preds, best_labels = va_preds, va_labels
            torch.save({"epoch": ep, "round": args.round, "model": model.state_dict(),
                        "val_f1": va_f1, "val_acc": va_acc, "val_msd": va_msd,
                        "classes": CLASSES},
                       CKPT_DIR / f"_tmp_round{args.round}.pt")

    print(f"\nBest val macro-F1 this round: {best_epoch_f1:.4f} at epoch {best_epoch}")
    print(f"Per-class F1: {per_class_f1(best_preds, best_labels)}")

    # ── Save checkpoint only if improves global best (or --force) ──
    improved = best_epoch_f1 > best_val_f1
    ckpt_path = CKPT_DIR / f"checkpoint_round{args.round}.pt"

    if improved or args.force:
        tmp = CKPT_DIR / f"_tmp_round{args.round}.pt"
        if tmp.exists(): tmp.rename(ckpt_path)
        if improved:
            state["best_val_f1"] = best_epoch_f1
            state["best_round"]  = args.round
        print(f"{'✓ Saved' if improved else '⚠ Forced save'}: {ckpt_path.name}")
    else:
        prev_best = state["best_val_f1"]
        print(f"✗ Val F1 {best_epoch_f1:.4f} did not beat global best {prev_best:.4f} — checkpoint NOT saved.")
        print(f"  Try: --replay-ratio {min(args.replay_ratio + 0.1, 0.7):.1f} or add more epochs.")

    # ── Update state ──
    new_cap_ids = list({r.get("source_capture_id") for r in new_tiles})
    state["used_capture_ids"] = list(used_caps | set(new_cap_ids))

    round_record = {
        "round":        args.round,
        "new_tiles":    len(new_samples),
        "replay_tiles": len(replay_samples),
        "val_acc":      round(best_va_acc, 4),
        "val_f1":       round(best_epoch_f1, 4),
        "val_msd":      round(best_va_msd, 4),
        "epochs":       args.epochs,
        "lr_head":      head_lr,
        "lr_backbone":  backbone_lr,
        "replay_ratio": args.replay_ratio,
        "saved":        improved or args.force,
    }
    state["rounds"].append(round_record)
    save_state(state)
    log_round(round_record)

    print_comparison(state, round_record)
    print(f"\nState -> {STATE_FILE}")
    print(f"Log   -> {LOG_FILE}")

if __name__ == "__main__":
    main()
