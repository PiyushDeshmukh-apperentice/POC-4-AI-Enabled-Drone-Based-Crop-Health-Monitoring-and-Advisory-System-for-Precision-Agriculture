#!/usr/bin/env python3
"""
02_split_lofo.py - Leave-one-field-out split with flight-level validation

  test  = every image from --test-field (e.g. F2)
  val   = whole flights held out from the other field, stratified by stage
  train = remaining flights of the other field

NOTE: 'field_plot' in the manifest is really a FLIGHT id (F1_03 = field 1,
flight 3). The field is the prefix before the first underscore.

Usage:
    python 02_split_lofo.py --test-field F2 --out split_test_F2.csv
    python 02_split_lofo.py --test-field F1 --out split_test_F1.csv
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict

SPLITS = ["train", "val", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_rgb.csv")
    ap.add_argument("--out", default="split.csv")
    ap.add_argument("--test-field", required=True, help="Field prefix, e.g. F2")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Approx. fraction of each stage's train-field images "
                         "held out as val (whole flights)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("Manifest empty.")

    for r in rows:
        r["field"] = r["field_plot"].split("_")[0]
        r["growth_stage_raw"] = r["growth_stage"]
        r["growth_stage"] = r["growth_stage"].strip().lower()   # merges Mature_Stage / Mature_stage

    fields = Counter(r["field"] for r in rows)
    print("Images per field:", dict(fields))
    if args.test_field not in fields:
        sys.exit(f"--test-field {args.test_field} not found. Options: {list(fields)}")

    # Show what the case-normalisation merged
    raw_variants = defaultdict(set)
    for r in rows:
        raw_variants[r["growth_stage"]].add(r["growth_stage_raw"])
    print("\nClass names after normalisation (raw variants merged):")
    for k in sorted(raw_variants):
        v = sorted(raw_variants[k])
        print(f"  {k:<24} <- {v}" + ("   [MERGED]" if len(v) > 1 else ""))

    # Sanity: each flight should belong to exactly one stage
    stages_per_flight = defaultdict(set)
    for r in rows:
        stages_per_flight[r["field_plot"]].add(r["growth_stage"])
    multi = {k: v for k, v in stages_per_flight.items() if len(v) > 1}
    if multi:
        print(f"[WARN] flights spanning multiple stages: {multi}")

    pool = [r for r in rows if r["field"] != args.test_field]
    if not pool:
        sys.exit("No training-field images left.")

    # stage -> flight -> rows  (train-field pool only)
    by_stage = defaultdict(lambda: defaultdict(list))
    for r in pool:
        by_stage[r["growth_stage"]][r["field_plot"]].append(r)

    rng = random.Random(args.seed)
    val_flights = set()
    for stage in sorted(by_stage):
        fl = by_stage[stage]
        names = sorted(fl)
        rng.shuffle(names)
        target = args.val_frac * sum(len(v) for v in fl.values())
        got, remaining = 0, len(names)
        for n in names:
            if remaining <= 1 or got >= target:   # always keep >=1 flight in train
                break
            val_flights.add(n)
            got += len(fl[n])
            remaining -= 1
        if args.val_frac > 0 and got == 0:
            print(f"[WARN] stage '{stage}' has only {len(names)} flight(s) in the "
                  f"train field; nothing held out for val.")

    for r in rows:
        if r["field"] == args.test_field:
            r["split"] = "test"
        elif r["field_plot"] in val_flights:
            r["split"] = "val"
        else:
            r["split"] = "train"

    # ---- Integrity checks ----
    fields_by_split = defaultdict(set)
    flights_by_split = defaultdict(set)
    for r in rows:
        fields_by_split[r["split"]].add(r["field"])
        flights_by_split[r["split"]].add(r["field_plot"])
    assert args.test_field not in (fields_by_split["train"] | fields_by_split["val"]), \
        "test field leaked into train/val"
    assert not (flights_by_split["train"] & flights_by_split["val"]), "flight in train and val"
    assert not (flights_by_split["train"] & flights_by_split["test"]), "flight in train and test"
    assert not (flights_by_split["val"] & flights_by_split["test"]), "flight in val and test"
    print("\n[OK] no field or flight appears in more than one split.")

    # ---- Output ----
    cols = ["file_path", "capture_id", "field", "field_plot", "session", "date",
            "growth_stage", "growth_stage_raw", "split"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ---- Report (stages in chronological order) ----
    first = {}
    for r in rows:
        first[r["growth_stage"]] = min(first.get(r["growth_stage"], "9999"),
                                       r["date"] or "9999")
    stages = sorted(first, key=lambda s: first[s])
    table = defaultdict(Counter)
    for r in rows:
        table[r["growth_stage"]][r["split"]] += 1

    print(f"\n{'Stage':<26}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'total':>8}")
    for st in stages:
        c = table[st]
        print(f"{st:<26}" + "".join(f"{c[s]:>8}" for s in SPLITS) + f"{sum(c.values()):>8}")
    tot = Counter(r["split"] for r in rows)
    print(f"{'ALL':<26}" + "".join(f"{tot[s]:>8}" for s in SPLITS) + f"{len(rows):>8}")
    print("Actual ratios:", {s: round(tot[s] / len(rows), 3) for s in SPLITS})

    for st in stages:
        for s in ("train", "test"):
            if table[st][s] == 0:
                print(f"[WARN] stage '{st}' has NO images in {s}; "
                      f"this field/direction can't evaluate it.")
    print(f"\nVal flights: {sorted(val_flights)}")
    print(f"Split written to {args.out}")


if __name__ == "__main__":
    main()
