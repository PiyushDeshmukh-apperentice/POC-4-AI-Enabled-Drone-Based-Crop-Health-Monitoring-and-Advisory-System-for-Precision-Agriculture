"""
rigorous_f1_validation_test.py

Empirical Validation & Stress-Testing Suite to verify whether the ~98% Macro-F1 
scores achieved on the held-out Field 2 test set reflect genuine rice canopy phenology
or are artifacts of overfitting, color shortcuts, or dataset constraints.

Battery of 5 Scientific Tests:
1. Trivial Baseline Test: Classical ML (Random Forest & Logistic Regression) on pure color 
   & spectral vegetation indices (RGB/HSV stats, ExG, GLI) - determines how much color alone explains.
2. Grayscale Ablation Test: Evaluate trained CNNs on 100% grayscale images (zero chromaticity) 
   to prove if models rely on spatial morphology, leaf tillering, and panicle textures.
3. Photometric & Environmental Stress Test: Apply 6 severe physical perturbations 
   (Brightness +/-30%, Contrast +/-30%, Motion Blur, Sensor Noise) to test field resilience.
4. Geographic Disaggregation Test: Disaggregate F1 across all 10 Field 2 plots (F2_01 to F2_10) 
   to verify spatial consistency across independent parcels.
5. Phenological Transition Analysis: Detailed chronological audit of misclassified captures 
   to verify if errors represent biological boundary overlap vs stochastic failure.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

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
VAL_OUTPUT_DIR = OUT_DIR / "f1_validation_results"
VAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Model Builder ─────────────────────────────────────────────────────────────
def build_model(arch: str, num_classes: int = 4) -> nn.Module:
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "efficientnet_b1":
        m = models.efficientnet_b1(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "efficientnet_b2":
        m = models.efficientnet_b2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "efficientnet_b3":
        m = models.efficientnet_b3(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def load_model(arch: str, device: torch.device):
    ckpt_path = CKPTS_DIR / f"{arch}_resize_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    data = torch.load(ckpt_path, map_location=device)
    model = build_model(arch, len(CLASSES))
    model.load_state_dict(data["model_state"])
    model = model.to(device).eval()
    return model


# ── Dataset Loading ───────────────────────────────────────────────────────────
def load_dataset_records():
    manifest_df = pd.read_csv("manifest.csv", low_memory=False)
    sources_df = manifest_df[
        (manifest_df["variant"] == "D") &
        (manifest_df["tile_file_path"].isna() | (manifest_df["tile_file_path"] == ""))
    ].drop_duplicates("capture_id").set_index("capture_id")

    train_split = pd.read_csv(SPLITS_DIR / "train.csv")
    test_split = pd.read_csv(SPLITS_DIR / "test.csv")

    train_cids = sorted(train_split["source_capture_id"].dropna().unique())
    test_cids = sorted(test_split["source_capture_id"].dropna().unique())

    train_samples = []
    for cid in train_cids:
        if cid in sources_df.index:
            stg = BUCKET_MAP.get(sources_df.loc[cid, "growth_stage"])
            if stg in CLASS_IDX:
                p = CACHE_DIR_512 / f"{cid}_D_512.jpg"
                if p.exists():
                    train_samples.append({
                        "cid": cid,
                        "path": str(p),
                        "label_str": stg,
                        "label_idx": CLASS_IDX[stg],
                        "plot": sources_df.loc[cid, "field_plot"],
                        "timestamp": str(sources_df.loc[cid, "timestamp"]),
                        "flight_date": str(sources_df.loc[cid, "timestamp"])[:8]
                    })

    test_samples = []
    for cid in test_cids:
        if cid in sources_df.index:
            stg = BUCKET_MAP.get(sources_df.loc[cid, "growth_stage"])
            if stg in CLASS_IDX:
                p = CACHE_DIR_512 / f"{cid}_D_512.jpg"
                if p.exists():
                    test_samples.append({
                        "cid": cid,
                        "path": str(p),
                        "label_str": stg,
                        "label_idx": CLASS_IDX[stg],
                        "plot": sources_df.loc[cid, "field_plot"],
                        "timestamp": str(sources_df.loc[cid, "timestamp"]),
                        "flight_date": str(sources_df.loc[cid, "timestamp"])[:8]
                    })

    return train_samples, test_samples


# ── TEST 1: Color & Spectral Feature Extraction & Baseline ────────────────────
def extract_color_features(img_path: str) -> np.ndarray:
    """
    Extracts a 14-dimensional color and spectral vegetation index vector:
    Mean & Std of R, G, B (6)
    Mean & Std of H, S, V (6)
    Excess Green Index: ExG = 2*G - R - B (1)
    Green Leaf Index: GLI = (2*G - R - B) / (2*G + R + B + 1e-6) (1)
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # RGB stats
    r_mean, r_std = float(np.mean(r)), float(np.std(r))
    g_mean, g_std = float(np.mean(g)), float(np.std(g))
    b_mean, b_std = float(np.mean(b)), float(np.std(b))

    # HSV stats
    img_hsv = img.convert("HSV")
    arr_hsv = np.asarray(img_hsv, dtype=np.float32) / 255.0
    h = arr_hsv[:, :, 0]
    s = arr_hsv[:, :, 1]
    v = arr_hsv[:, :, 2]
    h_mean, h_std = float(np.mean(h)), float(np.std(h))
    s_mean, s_std = float(np.mean(s)), float(np.std(s))
    v_mean, v_std = float(np.mean(v)), float(np.std(v))

    # Agronomic indices
    exg = float(np.mean(2.0 * g - r - b))
    denom = 2.0 * g + r + b + 1e-6
    gli = float(np.mean((2.0 * g - r - b) / denom))

    return np.array([
        r_mean, r_std, g_mean, g_std, b_mean, b_std,
        h_mean, h_std, s_mean, s_std, v_mean, v_std,
        exg, gli
    ], dtype=np.float32)


def run_test1_color_baseline(train_samples, test_samples):
    print("\n" + "="*70)
    print("TEST 1: TRIVIAL COLOR & SPECTRAL VEGETATION BASELINE")
    print("Hypothesis: If high F1 is just a color shortcut, a color-only model should match ~98% F1.")
    print("="*70)

    print("Extracting 14-D color/spectral features for Train (Field 1)...")
    X_train = np.array([extract_color_features(s["path"]) for s in train_samples])
    y_train = np.array([s["label_idx"] for s in train_samples])

    print("Extracting 14-D color/spectral features for Test (Field 2)...")
    X_test = np.array([extract_color_features(s["path"]) for s in test_samples])
    y_test = np.array([s["label_idx"] for s in test_samples])

    # Standardize
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-6
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # Model A: Random Forest (100 trees)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average="macro")

    # Model B: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds, average="macro")

    print(f"\n[Random Forest (Color Only)] Accuracy: {rf_acc*100:.2f}% | Macro-F1: {rf_f1*100:.2f}%")
    print(classification_report(y_test, rf_preds, target_names=CLASSES, digits=4))

    print(f"\n[Logistic Regression (Color Only)] Accuracy: {lr_acc*100:.2f}% | Macro-F1: {lr_f1*100:.2f}%")
    print(classification_report(y_test, lr_preds, target_names=CLASSES, digits=4))

    # Feature Importance in RF
    feat_names = ["R_mean", "R_std", "G_mean", "G_std", "B_mean", "B_std",
                  "H_mean", "H_std", "S_mean", "S_std", "V_mean", "V_std",
                  "ExG", "GLI"]
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_feats = {feat_names[i]: float(importances[i]) for i in sorted_idx[:5]}

    results = {
        "random_forest": {
            "accuracy": float(rf_acc),
            "macro_f1": float(rf_f1),
            "top_features": top_feats,
            "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist()
        },
        "logistic_regression": {
            "accuracy": float(lr_acc),
            "macro_f1": float(lr_f1),
            "confusion_matrix": confusion_matrix(y_test, lr_preds).tolist()
        }
    }
    return results


# ── PyTorch Dataset for Perturbation Testing ──────────────────────────────────
class PerturbationDataset(Dataset):
    def __init__(self, samples: List[Dict], perturbation_fn=None, input_size: int = 384):
        self.samples = samples
        self.perturbation_fn = perturbation_fn
        self.input_size = input_size
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.perturbation_fn is not None:
            img = self.perturbation_fn(img)
        img = img.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # If grayscale PIL image, shape might be (H, W) or (H, W, 3)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = self.norm(tensor)
        return tensor, s["label_idx"], s["cid"], s["plot"], s["flight_date"]


def evaluate_torch_model(model: nn.Module, dataset: Dataset, device: torch.device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds, all_targets, all_cids, all_plots, all_dates = [], [], [], [], []
    with torch.no_grad():
        for imgs, lbls, cids, plots, dates in loader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(lbls.tolist())
            all_cids.extend(cids)
            all_plots.extend(plots)
            all_dates.extend(dates)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds).tolist()
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "preds": all_preds,
        "targets": all_targets,
        "cids": all_cids,
        "plots": all_plots,
        "dates": all_dates,
        "cm": cm
    }


# ── TEST 2: Grayscale Ablation Test ───────────────────────────────────────────
def run_test2_grayscale(test_samples, models_dict, device):
    print("\n" + "="*70)
    print("TEST 2: GRAYSCALE ABLATION (ZERO CHROMATICITY / TEXTURE & MORPHOLOGY ONLY)")
    print("Hypothesis: If models relied purely on color lookup, removing color causes collapse (~25%).")
    print("="*70)

    def to_grayscale(img: Image.Image) -> Image.Image:
        # Convert to L (luminance: 0.299R + 0.587G + 0.114B) then back to RGB
        return img.convert("L").convert("RGB")

    gray_dataset = PerturbationDataset(test_samples, perturbation_fn=to_grayscale, input_size=384)

    results = {}
    for arch, model in models_dict.items():
        eval_res = evaluate_torch_model(model, gray_dataset, device)
        results[arch] = {
            "accuracy": eval_res["accuracy"],
            "macro_f1": eval_res["macro_f1"],
            "cm": eval_res["cm"]
        }
        print(f"[{arch:<15}] Grayscale Macro-F1: {eval_res['macro_f1']*100:.2f}% | Accuracy: {eval_res['accuracy']*100:.2f}%")

    return results


# ── TEST 3: Environmental Photometric Stress Test ─────────────────────────────
def run_test3_photometric_stress(test_samples, models_dict, device):
    print("\n" + "="*70)
    print("TEST 3: ENVIRONMENTAL PHOTOMETRIC STRESS TESTING")
    print("Hypothesis: Tests resilience to sun angles, cloud shadows, motion blur, and sensor grain.")
    print("="*70)

    perturbations = {
        "Clean (Baseline)": None,
        "High Brightness (+30%)": lambda img: ImageEnhance.Brightness(img).enhance(1.30),
        "Low Brightness (-30%)": lambda img: ImageEnhance.Brightness(img).enhance(0.70),
        "High Contrast (+30%)": lambda img: ImageEnhance.Contrast(img).enhance(1.30),
        "Low Contrast (-30%)": lambda img: ImageEnhance.Contrast(img).enhance(0.70),
        "Motion Blur (Gaussian Blur)": lambda img: img.filter(ImageFilter.GaussianBlur(radius=2.0)),
        "Sensor Noise (Gaussian Grain)": lambda img: Image.fromarray(
            np.clip(np.asarray(img, dtype=np.float32) + np.random.normal(0, 20, np.asarray(img).shape), 0, 255).astype(np.uint8)
        )
    }

    results = {arch: {} for arch in models_dict.keys()}

    for p_name, p_fn in perturbations.items():
        print(f"\nEvaluating Stress Condition: {p_name}...")
        ds = PerturbationDataset(test_samples, perturbation_fn=p_fn, input_size=384)
        for arch, model in models_dict.items():
            res = evaluate_torch_model(model, ds, device)
            results[arch][p_name] = {
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"]
            }
            print(f"  {arch:<15} -> Macro-F1: {res['macro_f1']*100:.2f}% | Acc: {res['accuracy']*100:.2f}%")

    return results


# ── TEST 4: Geographic Disaggregation (Plot-by-Plot) ──────────────────────────
def run_test4_spatial_disaggregation(clean_eval_b3):
    print("\n" + "="*70)
    print("TEST 4: GEOGRAPHIC SPATIAL DISAGGREGATION ACROSS FIELD 2 PLOTS")
    print("Hypothesis: Performance should be consistent across all 10 independent parcels.")
    print("="*70)

    df = pd.DataFrame({
        "cid": clean_eval_b3["cids"],
        "plot": clean_eval_b3["plots"],
        "flight_date": clean_eval_b3["dates"],
        "target": clean_eval_b3["targets"],
        "pred": clean_eval_b3["preds"]
    })
    df["correct"] = df["target"] == df["pred"]

    plot_stats = {}
    for plot, group in df.groupby("plot"):
        acc = group["correct"].mean()
        f1 = f1_score(group["target"], group["pred"], average="macro", zero_division=0)
        n = len(group)
        plot_stats[plot] = {
            "total_samples": int(n),
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "errors": int((~group["correct"]).sum())
        }
        print(f"Plot {plot:<6} | Samples: {n:3d} | Acc: {acc*100:6.2f}% | Macro-F1: {f1*100:6.2f}% | Errors: {(~group['correct']).sum()}")

    return plot_stats, df


# ── TEST 5: Phenological Transition Error Root-Cause Analysis ─────────────────
def run_test5_transition_error_analysis(df, test_samples):
    print("\n" + "="*70)
    print("TEST 5: PHENOLOGICAL TRANSITION ANALYSIS (MISCLASSIFICATION AUDIT)")
    print("Hypothesis: Errors occur exclusively at the late-vegetative/booting biological boundary.")
    print("="*70)

    errors = df[df["target"] != df["pred"]].copy()
    errors["target_name"] = [CLASSES[t] for t in errors["target"]]
    errors["pred_name"] = [CLASSES[p] for p in errors["pred"]]

    print(f"Total Errors in EfficientNet-B3: {len(errors)} / {len(df)} captures ({len(errors)/len(df)*100:.2f}%)")
    print("\nError Breakdown:")
    for (t_name, p_name), g in errors.groupby(["target_name", "pred_name"]):
        print(f"  True: {t_name:<12} -> Predicted: {p_name:<12} | Count: {len(g)}")
        dates = g["flight_date"].value_counts().to_dict()
        plots = g["plot"].value_counts().to_dict()
        print(f"    Flight Dates: {dates}")
        print(f"    Plots:        {plots}")

    error_records = errors[["cid", "plot", "flight_date", "target_name", "pred_name"]].to_dict(orient="records")
    return error_records


# ── Plotting Utilities ────────────────────────────────────────────────────────
def plot_validation_summary(test1_res, test2_res, test3_res, plot_stats):
    print("\nGenerating comprehensive validation visual summary plots...")
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # 1. Color Baseline vs CNNs
    ax1 = fig.add_subplot(gs[0, 0])
    models_list = ["Color RF", "Color LogReg", "ResNet-50", "EffNet-B0", "EffNet-B1", "EffNet-B2", "EffNet-B3"]
    f1s = [
        test1_res["random_forest"]["macro_f1"] * 100,
        test1_res["logistic_regression"]["macro_f1"] * 100,
        test3_res["resnet50"]["Clean (Baseline)"]["macro_f1"] * 100,
        test3_res["efficientnet_b0"]["Clean (Baseline)"]["macro_f1"] * 100,
        test3_res["efficientnet_b1"]["Clean (Baseline)"]["macro_f1"] * 100,
        test3_res["efficientnet_b2"]["Clean (Baseline)"]["macro_f1"] * 100,
        test3_res["efficientnet_b3"]["Clean (Baseline)"]["macro_f1"] * 100,
    ]
    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71", "#1abc9c", "#9b59b6", "#2c3e50"]
    bars = ax1.bar(models_list, f1s, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Macro-F1 Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Color-Only Baselines vs Deep Spatial CNNs (Field 2 Holdout)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.setp(ax1.get_xticklabels(), rotation=25, ha="right")

    # 2. Grayscale Ablation Retention
    ax2 = fig.add_subplot(gs[0, 1])
    archs = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "resnet50"]
    labels = ["EffNet-B0", "EffNet-B1", "EffNet-B2", "EffNet-B3", "ResNet-50"]
    rgb_f1s = [test3_res[a]["Clean (Baseline)"]["macro_f1"] * 100 for a in archs]
    gray_f1s = [test2_res[a]["macro_f1"] * 100 for a in archs]
    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(x - w/2, rgb_f1s, w, label="Full RGB Color", color="#27ae60", edgecolor="black", alpha=0.85)
    ax2.bar(x + w/2, gray_f1s, w, label="100% Grayscale (Zero Color)", color="#7f8c8d", edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontweight="bold")
    ax2.set_ylabel("Macro-F1 Score (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Grayscale Ablation: Structural Morphology Retention", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for i in range(len(labels)):
        ax2.text(x[i] + w/2, gray_f1s[i] + 1.5, f"{gray_f1s[i]:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 3. Photometric Stress Degradation (EfficientNet-B3 vs ResNet-50)
    ax3 = fig.add_subplot(gs[1, 0])
    conditions = list(test3_res["efficientnet_b3"].keys())
    b3_f1s = [test3_res["efficientnet_b3"][c]["macro_f1"] * 100 for c in conditions]
    res_f1s = [test3_res["resnet50"][c]["macro_f1"] * 100 for c in conditions]
    xc = np.arange(len(conditions))
    wc = 0.35
    ax3.bar(xc - wc/2, b3_f1s, wc, label="EfficientNet-B3", color="#2980b9", edgecolor="black", alpha=0.85)
    ax3.bar(xc + wc/2, res_f1s, wc, label="ResNet-50", color="#d35400", edgecolor="black", alpha=0.85)
    ax3.set_xticks(xc)
    cond_labels_clean = [c.replace(" (Baseline)", "").replace(" (Gaussian Blur)", "").replace(" (Gaussian Grain)", "") for c in conditions]
    ax3.set_xticklabels(cond_labels_clean, rotation=25, ha="right", fontsize=9)
    ax3.set_ylabel("Macro-F1 Score (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Stress Testing: Environmental Perturbation Resilience", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 105)
    ax3.legend(loc="lower left", frameon=True)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)

    # 4. Plot-by-Plot Stability in Field 2
    ax4 = fig.add_subplot(gs[1, 1])
    sorted_plots = sorted(plot_stats.keys())
    plot_f1s = [plot_stats[p]["macro_f1"] * 100 for p in sorted_plots]
    plot_accs = [plot_stats[p]["accuracy"] * 100 for p in sorted_plots]
    xp = np.arange(len(sorted_plots))
    ax4.plot(xp, plot_f1s, marker="o", linewidth=2.5, color="#8e44ad", label="Macro-F1")
    ax4.plot(xp, plot_accs, marker="s", linewidth=2.0, linestyle="--", color="#16a085", label="Accuracy")
    ax4.set_xticks(xp)
    ax4.set_xticklabels(sorted_plots, fontweight="bold")
    ax4.set_ylabel("Metric (%)", fontsize=11, fontweight="bold")
    ax4.set_title("Spatial Generalization Across All 10 Field 2 Plots (EffNet-B3)", fontsize=12, fontweight="bold")
    ax4.set_ylim(85, 102)
    ax4.legend(loc="lower right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plot_out = VAL_OUTPUT_DIR / "f1_rigorous_validation_dashboard.png"
    plt.savefig(plot_out, bbox_inches="tight")
    plt.close()
    print(f"Validation dashboard saved to: {plot_out}")


# ── MAIN EXECUTION ────────────────────────────────────────────────────────────
def main():
    start_t = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("\nLoading dataset records...")
    train_samples, test_samples = load_dataset_records()
    print(f"Train samples (Field 1): {len(train_samples)}")
    print(f"Test samples (Field 2 Holdout): {len(test_samples)}")

    # Load all 5 PyTorch models
    print("\nLoading trained CNN checkpoints...")
    archs = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "resnet50"]
    models_dict = {a: load_model(a, device) for a in archs}

    # TEST 1: Color baseline
    test1_results = run_test1_color_baseline(train_samples, test_samples)

    # TEST 2: Grayscale ablation
    test2_results = run_test2_grayscale(test_samples, models_dict, device)

    # TEST 3: Photometric stress test
    test3_results = run_test3_photometric_stress(test_samples, models_dict, device)

    # Clean evaluation for B3
    clean_ds = PerturbationDataset(test_samples, perturbation_fn=None, input_size=384)
    clean_eval_b3 = evaluate_torch_model(models_dict["efficientnet_b3"], clean_ds, device)

    # TEST 4: Spatial disaggregation
    test4_results, df_preds = run_test4_spatial_disaggregation(clean_eval_b3)

    # TEST 5: Phenological transition analysis
    test5_results = run_test5_transition_error_analysis(df_preds, test_samples)

    # Save summary plots
    plot_validation_summary(test1_results, test2_results, test3_results, test4_results)

    # Save complete JSON report
    final_output = {
        "device": str(device),
        "total_train_samples": len(train_samples),
        "total_test_samples": len(test_samples),
        "test1_color_baseline": test1_results,
        "test2_grayscale_ablation": test2_results,
        "test3_photometric_stress": test3_results,
        "test4_plot_disaggregation": test4_results,
        "test5_transition_error_analysis": test5_results,
        "total_runtime_seconds": time.time() - start_t
    }

    json_path = VAL_OUTPUT_DIR / "rigorous_validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved complete JSON validation report to: {json_path}")
    print(f"Total testing completed in {time.time() - start_t:.1f}s.")


if __name__ == "__main__":
    main()
