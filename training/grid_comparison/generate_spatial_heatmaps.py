"""
generate_spatial_heatmaps.py

Generates:
1. Spatial Growth-Stage Region Heatmaps across the 35-tile grid on full UAV captures ($5280x3956$).
   - Overlays semi-transparent colored bounding boxes on the original drone photo.
   - Stage Colors: Seedling (Cyan), Vegetative (Green), Flowering (Orange), Mature (Gold).
   - Localizes WHERE each specific growth stage is seen across field parcels.
   - Displays 2D parcel grid matrix, per-stage field canopy distribution (%), and parcel coordinates.
2. Grad-CAM Feature Activation Heatmaps showing visual canopy textures (panicles, tillers, foliage).
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CLASSES = ["Seedling", "Vegetative", "Flowering", "Mature"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

STAGE_COLORS = {
    "Seedling":   (0.0, 0.71, 0.85, 0.45),  # Cyan
    "Vegetative": (0.18, 0.55, 0.34, 0.45), # Forest Green
    "Flowering":  (0.97, 0.50, 0.00, 0.45), # Orange
    "Mature":     (1.00, 0.82, 0.40, 0.45), # Gold
}

BORDER_COLORS = {
    "Seedling":   "#0077b6",
    "Vegetative": "#1b4332",
    "Flowering":  "#d90429",
    "Mature":     "#b07d62",
}

TILE_SIZE = 1024
STRIDE = 768

OUT_DIR = Path("training/grid_comparison")
CKPTS_DIR = OUT_DIR / "checkpoints"
HEATMAP_DIR = OUT_DIR / "spatial_heatmaps"
GRADCAM_DIR = OUT_DIR / "gradcam_heatmaps"
SPLITS_DIR = Path("training/splits")


def get_tile_bboxes(width: int = 5280, height: int = 3956) -> List[Tuple[int, int, int, int, int, int]]:
    """Returns list of (x0, y0, x1, y1, row_idx, col_idx) for 35 tiles."""
    x_starts = list(range(0, width - TILE_SIZE + 1, STRIDE))
    if x_starts[-1] + TILE_SIZE < width:
        x_starts.append(width - TILE_SIZE)

    y_starts = list(range(0, height - TILE_SIZE + 1, STRIDE))
    if y_starts[-1] + TILE_SIZE < height:
        y_starts.append(height - TILE_SIZE)

    bboxes = []
    for r_idx, y in enumerate(y_starts):
        for c_idx, x in enumerate(x_starts):
            bboxes.append((x, y, x + TILE_SIZE, y + TILE_SIZE, r_idx, c_idx))
    return bboxes


def build_model(arch: str, num_classes: int = 4) -> nn.Module:
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "efficientnet_b1":
        m = models.efficientnet_b1(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def load_best_model(ckpt_path: Path, device: torch.device):
    data = torch.load(ckpt_path, map_location=device)
    arch = data["arch"]
    model = build_model(arch, len(CLASSES))
    model.load_state_dict(data["model_state"])
    model = model.to(device).eval()
    return model, arch, data.get("resolution", 288)


# ── Spatial Region Heatmap Generator ──────────────────────────────────────────
def generate_spatial_region_map(
    image_path: Path,
    capture_id: str,
    true_stage: str,
    model: nn.Module,
    device: torch.device,
    input_size: int,
    out_path: Path,
):
    orig_img = Image.open(image_path).convert("RGB")
    w, h = orig_img.size
    bboxes = get_tile_bboxes(w, h)

    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tile_tensors = []
    for (x0, y0, x1, y1, r, c) in bboxes:
        crop = orig_img.crop((x0, y0, x1, y1))
        tile_tensors.append(tf(crop))

    batch = torch.stack(tile_tensors).to(device)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

    preds = probs.argmax(axis=1)
    stage_names = [CLASSES[p] for p in preds]

    # Create figure: Left = UAV capture with overlay, Right = 5x7 matrix + stats
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 0.7], wspace=0.15)

    # 1. Overlay on raw UAV Photo
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(orig_img)
    ax_img.set_title(f"Field UAV Capture: {capture_id} (True Stage: {true_stage})\n"
                     f"35-Tile Spatial Grid (1024x1024 px, 25% overlap) — Growth Stage Localization",
                     fontsize=12, fontweight="bold", pad=10)

    for (x0, y0, x1, y1, r, c), stage, prob in zip(bboxes, stage_names, probs):
        conf = prob[CLASS_IDX[stage]] * 100
        # Add colored bounding box
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=2, edgecolor=BORDER_COLORS[stage],
                                 facecolor=STAGE_COLORS[stage])
        ax_img.add_patch(rect)
        # Add label tag
        lbl = f"[{r},{c}] {stage[:3]}\n{conf:.1f}%"
        ax_img.text(x0 + 40, y0 + 150, lbl,
                    fontsize=8.5, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=BORDER_COLORS[stage], alpha=0.85, ec="none"))

    ax_img.set_xlim(0, w)
    ax_img.set_ylim(h, 0)
    ax_img.axis("off")

    # 2. Right: 5x7 Discrete Matrix + Canopy Distribution Breakdown
    ax_mat = fig.add_subplot(gs[1])
    # Build 5x7 color grid
    grid_data = np.zeros((5, 7))
    for (_, _, _, _, r, c), pred in zip(bboxes, preds):
        grid_data[r, c] = pred

    # Map to colormap
    color_map_vals = np.array([[0, 180, 216], [45, 106, 79], [247, 127, 0], [255, 209, 102]]) / 255.0
    grid_rgb = np.zeros((5, 7, 3))
    for r in range(5):
        for c in range(7):
            grid_rgb[r, c] = color_map_vals[int(grid_data[r, c])]

    ax_mat.imshow(grid_rgb, aspect="equal")
    ax_mat.set_title("5×7 Spatial Phenology Grid Map\nPer-Parcel Growth Stage Classification",
                     fontsize=11, fontweight="bold", pad=10)
    ax_mat.set_xticks(range(7))
    ax_mat.set_yticks(range(5))
    ax_mat.set_xticklabels([f"C{i}" for i in range(7)], fontsize=9)
    ax_mat.set_yticklabels([f"R{i}" for i in range(5)], fontsize=9)

    for (x0, y0, x1, y1, r, c), stage in zip(bboxes, stage_names):
        ax_mat.text(c, r, stage[:4], ha="center", va="center",
                    color="white" if stage != "Mature" else "black",
                    fontsize=8.5, fontweight="bold")

    # Add summary text below matrix
    counts = Counter(stage_names)
    majority_stage = counts.most_common(1)[0][0]
    total_tiles = len(stage_names)

    summary_text = (
        f"\nFIELD CANOPY COMPOSITION:\n"
        f"  - Seedling   : {counts.get('Seedling', 0):2d} parcels ({counts.get('Seedling', 0)/total_tiles*100:5.1f}%)\n"
        f"  - Vegetative : {counts.get('Vegetative', 0):2d} parcels ({counts.get('Vegetative', 0)/total_tiles*100:5.1f}%)\n"
        f"  - Flowering  : {counts.get('Flowering', 0):2d} parcels ({counts.get('Flowering', 0)/total_tiles*100:5.1f}%)\n"
        f"  - Mature     : {counts.get('Mature', 0):2d} parcels ({counts.get('Mature', 0)/total_tiles*100:5.1f}%)\n"
        f"----------------------------------------------------\n"
        f"  ★ Majority Vote Stage : {majority_stage.upper()} ({counts[majority_stage]}/{total_tiles} parcels)\n"
        f"  ★ Ground Truth Stage  : {true_stage.upper()} {'[MATCH]' if majority_stage == true_stage else '[MISMATCH]'}\n"
    )
    fig.text(0.66, 0.12, summary_text, fontsize=10, family="monospace",
             bbox=dict(boxstyle="square,pad=0.6", fc="#f8f9fa", ec="#ced4da", lw=1.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved Spatial Region Heatmap to: {out_path.name}")


# ── Grad-CAM Feature Attention Heatmap ────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


def generate_gradcam_gallery(
    model: nn.Module,
    device: torch.device,
    sample_tiles: List[Tuple[str, str]],
    input_size: int,
    out_path: Path,
):
    # Find last convolutional layer for Grad-CAM
    if hasattr(model, "features"):
        target_layer = model.features[-1]
    elif hasattr(model, "layer4"):
        target_layer = model.layer4[-1]
    else:
        target_layer = list(model.children())[-2]

    cam_engine = GradCAM(model, target_layer)

    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    fig, axes = plt.subplots(len(sample_tiles), 3, figsize=(11, 2.8 * len(sample_tiles)))
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    for i, (tile_path, stage_name) in enumerate(sample_tiles):
        pil_img = Image.open(tile_path).convert("RGB")
        tensor = tf(pil_img).unsqueeze(0).to(device)
        tensor.requires_grad = True

        target_idx = CLASS_IDX[stage_name]
        cam_map = cam_engine.generate(tensor, target_idx)

        # Raw tile
        crop_disp = pil_img.resize((input_size, input_size))
        axes[i, 0].imshow(crop_disp)
        axes[i, 0].set_title(f"Raw Canopy Crop\nTrue: {stage_name}", fontsize=9, fontweight="bold")
        axes[i, 0].axis("off")

        # Grad-CAM heatmap
        axes[i, 1].imshow(cam_map, cmap="jet")
        axes[i, 1].set_title(f"Grad-CAM Attention\nTarget: {stage_name}", fontsize=9, fontweight="bold")
        axes[i, 1].axis("off")

        # Overlay
        axes[i, 2].imshow(crop_disp)
        axes[i, 2].imshow(cam_map, cmap="jet", alpha=0.5)
        axes[i, 2].set_title(f"Canopy Focus Overlay\n({stage_name} Features)", fontsize=9, fontweight="bold")
        axes[i, 2].axis("off")

    cam_engine.remove_hooks()
    plt.suptitle("Grad-CAM Canopy Feature Attention Across Rice Growth Stages", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM Gallery to: {out_path.name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    # Use best available tile checkpoint
    ckpts = list(CKPTS_DIR.glob("*tile*best.pt"))
    if not ckpts:
        ckpts = list(CKPTS_DIR.glob("*best.pt"))
    if not ckpts:
        print("No trained checkpoints found yet.")
        return

    best_ckpt = ckpts[0]
    print(f"Using model checkpoint: {best_ckpt.name}")
    model, arch, input_size = load_best_model(best_ckpt, device)

    # Find sample test full captures representing each growth stage
    manifest_df = pd.read_csv("manifest.csv", low_memory=False)
    test_split = pd.read_csv(SPLITS_DIR / "test.csv")
    test_cids = test_split["source_capture_id"].dropna().unique()

    sources_df = manifest_df[
        (manifest_df["variant"] == "D") &
        (manifest_df["capture_id"].isin(test_cids)) &
        (manifest_df["tile_file_path"].isna() | (manifest_df["tile_file_path"] == ""))
    ].drop_duplicates("capture_id")

    BUCKET_MAP_LOCAL = {
        "Nursery_Stage": "Seedling",
        "Vegetative_Stage": "Vegetative",
        "Booting_Stage": "Flowering",
        "Flowering_Stage": "Flowering",
        "Mature_Stage": "Mature",
        "Mature_stage": "Mature",
    }

    selected_samples = {}
    for _, r in sources_df.iterrows():
        b = BUCKET_MAP_LOCAL.get(r["growth_stage"])
        if b and b not in selected_samples:
            p = Path(r["file_path"])
            if p.exists():
                selected_samples[b] = (p, r["capture_id"], b)
        if len(selected_samples) == 4:
            break

    print(f"\nGenerating Spatial Region Heatmaps for {len(selected_samples)} representative test scenes...")
    for stage, (img_p, cid, stg) in selected_samples.items():
        out_f = HEATMAP_DIR / f"spatial_region_map_{stage.lower()}_{cid}.png"
        generate_spatial_region_map(img_p, cid, stg, model, device, input_size, out_f)

    # Sample tiles for Grad-CAM
    sample_tiles = []
    for stage in CLASSES:
        stage_tiles = test_split[test_split["bucket"] == stage]["tile_file_path"].tolist()
        for t in stage_tiles:
            if Path(t).exists():
                sample_tiles.append((t, stage))
                break

    print(f"\nGenerating Grad-CAM Attention Gallery for {len(sample_tiles)} growth stages...")
    out_cam = GRADCAM_DIR / "gradcam_canopy_attention_gallery.png"
    generate_gradcam_gallery(model, device, sample_tiles, input_size, out_cam)


if __name__ == "__main__":
    main()
