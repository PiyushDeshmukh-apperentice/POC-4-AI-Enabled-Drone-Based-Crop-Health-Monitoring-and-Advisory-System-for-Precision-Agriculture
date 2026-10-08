#!/usr/bin/env python3
"""
02c_split_control.py - Within-field RANDOM image-level split (control)

Deliberately reproduces the leaky setup, on ONE field. Compare:
  A  old mixed-field random split   (leaky,     same field(s), both fields' data)
  B  this control                   (leaky,     same field,    one field)
  C  LOFO F1 -> F2                  (leak-free, cross-field)
B vs C mixes leakage-removed and field-shift; it cannot separate them.

Usage:
    python 02c_split_control.py --field F1 --out split_control_F1.csv
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_rgb.csv")
    ap.add_argument("--out", default="split_control.csv")
    ap.add_argument("--field", required=True, help="Field prefix, e.g. F1")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["field"] = r["field_plot"].split("_")[0]
        r["growth_stage_raw"] = r["growth_stage"]
        r["growth_stage"] = r["growth_stage"].strip().lower()   # same normalisation as LOFO
    rows = [r for r in rows if r["field"] == args.field]
    if not rows:
        sys.exit(f"No rows for field {args.field}")

    rng = random.Random(args.seed)
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["growth_stage"]].append(r)

    for stage in sorted(by_stage):
        rs = sorted(by_stage[stage], key=lambda r: r["file_path"])  # deterministic base order
        rng.shuffle(rs)
        n_test = max(1, round(args.test_frac * len(rs)))
        if len(rs) - n_test < 1:
            sys.exit(f"Stage '{stage}' has too few images ({len(rs)}) to split.")
        for i, r in enumerate(rs):
            r["split"] = "test" if i < n_test else "train"

    # Show that the leak is present by design
    flights = defaultdict(set)
    for r in rows:
        flights[r["field_plot"]].add(r["split"])
    both = sorted(k for k, v in flights.items() if len(v) > 1)
    print(f"Flights with images in BOTH train and test: {len(both)} of {len(flights)} "
          f"(leaky by design)")

    cols = ["file_path", "capture_id", "field", "field_plot", "session", "date",
            "growth_stage", "growth_stage_raw", "split"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    first = {}
    for r in rows:
        first[r["growth_stage"]] = min(first.get(r["growth_stage"], "9999"), r["date"] or "9999")
    stages = sorted(first, key=lambda s: first[s])
    table = defaultdict(Counter)
    for r in rows:
        table[r["growth_stage"]][r["split"]] += 1
    print(f"\n{'Stage':<26}{'train':>8}{'test':>8}{'total':>8}")
    for st in stages:
        c = table[st]
        print(f"{st:<26}{c['train']:>8}{c['test']:>8}{sum(c.values()):>8}")
    tot = Counter(r["split"] for r in rows)
    print(f"{'ALL':<26}{tot['train']:>8}{tot['test']:>8}{len(rows):>8}")
    print(f"\nControl split written to {args.out}")


if __name__ == "__main__":
    main()
