import json
from pathlib import Path

EVAL_JSON_RESIZE = Path(r"e:\paddy_test\training\full_dataset_b2_b3\resizing_384\evaluation_results.json")
EVAL_JSON_VOTING = Path(r"e:\paddy_test\training\full_dataset_b2_b3\voting_288\evaluation_results.json")
REPORT_MD = Path(r"e:\paddy_test\training\full_dataset_b2_b3\final_report.md")

def main():
    data = []
    for path in [EVAL_JSON_RESIZE, EVAL_JSON_VOTING]:
        if path.exists():
            with open(path, "r") as f:
                data.extend(json.load(f))
        else:
            print(f"Warning: {path} not found.")

    if not data:
        print("No evaluation data found. Run training scripts first.")
        return

    # Sort by architecture, then by method
    data.sort(key=lambda x: (x["arch"], x["method"]))

    md_content = "# Full Dataset 8,486 Images - Evaluation Report\n\n"
    md_content += "| Architecture | Input Method | Resolution | Overall Acc | Macro-F1 | Veg→Flow | Flow→Veg | Params (M) | Train Time |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|\n"

    for d in data:
        arch = d["arch"].upper()
        method = d["method"].capitalize()
        res = f"{d['resolution']}x{d['resolution']}"
        acc = f"{d['overall_accuracy']*100:.2f}%"
        f1 = f"{d['macro_f1']:.4f}"
        v2f = d["veg_to_flow_confusion"]
        f2v = d["flow_to_veg_confusion"]
        params = f"{d['param_count'] / 1e6:.2f}M"
        time_m = f"{d['training_time'] / 60:.1f} min"
        
        md_content += f"| **{arch}** | {method} | {res} | {acc} | **{f1}** | {v2f} | {f2v} | {params} | {time_m} |\n"

    print("\n--- RESULTS ---")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nSaved report to {REPORT_MD}")

if __name__ == "__main__":
    main()
