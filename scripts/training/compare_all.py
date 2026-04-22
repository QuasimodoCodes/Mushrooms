"""
Compare all trained models across the whole project.

Run from the project root:
    python Herman/compare.py

Scans docs/cnn_runs/ and docs/herman_runs/ for results.csv files,
extracts the best val top-1 from each, and prints a ranked summary table.
Also includes the YOLO runs (read from their results.csv if present).
"""

import csv
import os
import glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

# Known YOLO results (hard-coded from docs since YOLO uses a different CSV format)
YOLO_KNOWN = {
    "YOLOv8n-cls  (baseline)":  {"top1": 86.8, "top5": 97.9, "params_m": 2.7,  "size_mb": 3.4},
    "YOLOv26n-cls (production)":{"top1": 88.1, "top5": 98.4, "params_m": 1.74, "size_mb": 3.6},
}


def best_from_csv(csv_path: str) -> dict:
    """
    Read a results.csv and return the row with the highest metrics/accuracy_top1.
    Returns None if file is missing or empty.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        best_row = None
        best_top1 = -1.0
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t1 = float(row.get("metrics/accuracy_top1", 0))
                if t1 > best_top1:
                    best_top1 = t1
                    best_row = row
        return best_row
    except Exception:
        return None


def scan_runs(base_dir: str) -> list[dict]:
    """Return list of {name, top1, top5, best_epoch} dicts for all runs in base_dir."""
    results = []
    if not os.path.isdir(base_dir):
        return results

    for run_folder in sorted(os.listdir(base_dir)):
        csv_path = os.path.join(base_dir, run_folder, "results.csv")
        row = best_from_csv(csv_path)
        if row:
            results.append({
                "name":       run_folder,
                "top1":       float(row.get("metrics/accuracy_top1", 0)),
                "top5":       float(row.get("metrics/accuracy_top5", 0)),
                "best_epoch": int(float(row.get("epoch", 0))),
            })
    return results


def main():
    print("\n" + "=" * 75)
    print("  Mushroom Classifier — Full Model Comparison")
    print("=" * 75)

    all_models = []

    # YOLO (hard-coded known results)
    for name, info in YOLO_KNOWN.items():
        all_models.append({
            "source": "YOLO",
            "name":   name,
            "top1":   info["top1"],
            "top5":   info["top5"],
            "note":   f"{info['params_m']}M params, {info['size_mb']} MB",
        })

    # EfficientNet / CNN runs
    cnn_runs = scan_runs(os.path.join(_ROOT, "docs", "cnn_runs"))
    for r in cnn_runs:
        all_models.append({
            "source": "CNN",
            "name":   r["name"],
            "top1":   r["top1"],
            "top5":   r["top5"],
            "note":   f"best epoch {r['best_epoch']}",
        })

    # Herman runs (ViT + ConvNeXt)
    herman_runs = scan_runs(os.path.join(_ROOT, "docs", "herman_runs"))
    for r in herman_runs:
        all_models.append({
            "source": "Herman",
            "name":   r["name"],
            "top1":   r["top1"],
            "top5":   r["top5"],
            "note":   f"best epoch {r['best_epoch']}",
        })

    # Sort by top-1 descending
    all_models.sort(key=lambda x: x["top1"], reverse=True)

    # Print table
    print(f"\n  {'Rank':<5} {'Source':<8} {'Top-1':>6}  {'Top-5':>6}  {'Run / Model'}")
    print("  " + "-" * 70)
    for i, m in enumerate(all_models, 1):
        flag = "  ← BEST" if i == 1 else ""
        print(
            f"  {i:<5} {m['source']:<8} {m['top1']:>5.1f}%  {m['top5']:>5.1f}%  "
            f"{m['name']}  ({m['note']}){flag}"
        )

    print("\n  Paths:")
    print(f"    CNN runs    → docs/cnn_runs/")
    print(f"    Herman runs → docs/herman_runs/")
    print()

    if not cnn_runs and not herman_runs:
        print("  No trained runs found yet. Train a model first, then re-run this script.\n")


if __name__ == "__main__":
    main()
