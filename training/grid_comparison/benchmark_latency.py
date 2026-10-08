"""
benchmark_latency.py

Unified, controlled latency benchmark across all 10 models on the same GPU hardware.
Uses a single standardized benchmark methodology:
- 15 warm-up iterations.
- 50 timed iterations with torch.cuda.synchronize().
- Measures:
  (a) Direct Resize: wall-clock time for 1 single 384x384 image pass (ms).
  (b) Spatial Tiling: wall-clock time for all 35 tiles (288x288) + majority vote aggregation (ms).
- Reports: Mean (ms), Std (ms), Throughput (FPS), Speedup ratio.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

OUT_DIR = Path("training/grid_comparison")
CKPTS_DIR = OUT_DIR / "checkpoints"


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


def benchmark_single_model(
    arch: str,
    method: str,
    device: torch.device,
    warmup_runs: int = 15,
    timed_runs: int = 50,
    batch_tiles: int = 35,
) -> Dict:
    model = build_model(arch, num_classes=4).to(device).eval()

    # Load checkpoint weights if available
    ckpt_p = CKPTS_DIR / f"{arch}_{method}_best.pt"
    if ckpt_p.exists():
        state = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(state["model_state"])

    # Prepare input tensor
    if method == "resize":
        # 1 single 384x384 photo
        input_tensor = torch.randn(1, 3, 384, 384, device=device)
    elif method == "tile":
        # Exactly 35 tiles of 288x288
        input_tensor = torch.randn(batch_tiles, 3, 288, 288, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            with torch.amp.autocast('cuda'):
                out = model(input_tensor)
                if method == "tile":
                    preds = out.argmax(1).cpu().tolist()
                    _ = Counter(preds).most_common(1)[0][0]
            torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(timed_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast('cuda'):
                out = model(input_tensor)
                if method == "tile":
                    preds = out.argmax(1).cpu().tolist()
                    _ = Counter(preds).most_common(1)[0][0]

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat

    return {
        "arch": arch,
        "method": method,
        "input_shape": list(input_tensor.shape),
        "mean_latency_ms": mean_lat,
        "std_latency_ms": std_lat,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "fps": fps
    }


def run_master_benchmark(device: torch.device) -> List[Dict]:
    archs = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "resnet50"]
    methods = ["resize", "tile"]

    print("\n" + "="*80)
    print(" EXECUTING UNIFIED LATENCY BENCHMARK (ONE PASS, SAME HARDWARE)")
    print(f" Device: {device} ({torch.cuda.get_device_name(0)})")
    print("="*80, flush=True)

    results = []
    for m in methods:
        for a in archs:
            res = benchmark_single_model(a, m, device)
            results.append(res)
            tag = "1-Pass Direct Resize" if m == "resize" else "35-Tile Grid + Majority Vote"
            print(f" {a.upper():<16} | {tag:<30} | {res['mean_latency_ms']:6.2f} ms ± {res['std_latency_ms']:4.2f} ms | {res['fps']:5.2f} FPS", flush=True)

    # Save results JSON
    out_json = OUT_DIR / "latency_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot comparison bar chart
    plot_latency_comparison(results, OUT_DIR / "latency_comparison.png")
    print(f"\nLatency benchmark saved to {out_json} and {OUT_DIR / 'latency_comparison.png'}")
    return results


def plot_latency_comparison(results: List[Dict], out_path: Path):
    archs = ["B0", "B1", "B2", "B3", "ResNet50"]
    resize_lats = [r["mean_latency_ms"] for r in results if r["method"] == "resize"]
    tile_lats = [r["mean_latency_ms"] for r in results if r["method"] == "tile"]

    x = np.arange(len(archs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width/2, resize_lats, width, label='Direct Resize (384px, 1 pass)', color='#2ca02c')
    rects2 = ax.bar(x + width/2, tile_lats, width, label='Spatial Tiling (35 tiles + vote)', color='#1f77b4')

    ax.set_ylabel('Latency per UAV Capture (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Controlled Single-Image Latency Benchmark: Direct Resize vs. Spatial Tiling', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_master_benchmark(device)


if __name__ == "__main__":
    main()
