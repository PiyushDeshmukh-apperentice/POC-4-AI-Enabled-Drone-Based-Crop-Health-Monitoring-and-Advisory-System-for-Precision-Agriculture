import os
import argparse
from pathlib import Path
import torch
import pandas as pd
from typing import Dict, List
import json

# Import functions from 03_train_resizing_384.py
import sys
sys.path.append(r"e:\paddy_test\training\full_dataset_b2_b3")
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader("train_resizing", r"e:\paddy_test\training\full_dataset_b2_b3\03_train_resizing_384.py").load_module()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--root", default=r"e:\paddy_test\test_500_rgb_images")
    parser.add_argument("--out-dir", default=r"e:\paddy_test\training\full_dataset_b2_b3\resizing_384")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8) # Lower batch size to prevent OOM
    args = parser.parse_args()

    mod.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    run_name = Path(args.split_csv).stem
    ckpt_path = Path(args.out_dir) / "checkpoints" / f"{args.arch}_{run_name}_s{args.seed}_final.pt"
    print(f"Loading checkpoint {ckpt_path}...")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = mod.build_model(args.arch).to(device)
    model.load_state_dict(ckpt["model_state"])
    
    # We don't have total_time and param_cnt exactly, but we can reconstruct param_cnt
    param_cnt = sum(p.numel() for p in model.parameters())
    # Try to find total time from curve CSV
    curve_csv = Path(args.out_dir) / "curves" / f"{args.arch}_{run_name}_s{args.seed}_curve.csv"
    total_time = 0.0
    if curve_csv.exists():
        curve_df = pd.read_csv(curve_csv)
        total_time = curve_df["epoch_time_s"].sum()
        
    entry = mod.evaluate_model(args, model, args.arch, total_time, param_cnt, device)
    
    if entry:
        mod.update_json(Path(args.out_dir), [entry])
        print("Successfully evaluated and updated JSON.")

if __name__ == "__main__":
    main()
