"""
Evaluate a trained YOLOv10n-cls on the held-out test set.

Run from the project root AFTER training:
    python scripts/training/yolo10/evaluate_yolo10.py

Mirrors scripts/training/yolo/compare_pt_vs_tflite.py structure
but focuses on top-1/top-5 reporting against YOLOv26 baseline.
"""

import os
import time
import torch
from ultralytics import YOLO


def evaluate(model_path: str, data_dir: str, label: str) -> dict:
    print(f"\n{'='*50}")
    print(f"  Evaluating: {label}")
    print(f"  Weights  : {os.path.basename(model_path)}")
    print(f"{'='*50}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nTrain first with train_yolo10.py")

    model = YOLO(model_path, task="classify")
    t0 = time.time()
    results = model.val(data=data_dir, imgsz=224, batch=32, verbose=False)
    elapsed = time.time() - t0

    top1 = results.results_dict.get("metrics/accuracy_top1", 0)
    top5 = results.results_dict.get("metrics/accuracy_top5", 0)

    print(f"  Top-1 Accuracy : {top1*100:.2f}%")
    print(f"  Top-5 Accuracy : {top5*100:.2f}%")
    print(f"  Eval time      : {elapsed:.1f}s")
    print(f"  Model size     : {os.path.getsize(model_path) / 1e6:.1f} MB")

    return {"label": label, "top1": top1, "top5": top5, "time": elapsed,
            "size_mb": os.path.getsize(model_path) / 1e6}


def main():
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

    data_dir    = os.path.join(_ROOT, "data", "dataset_split")
    weights_dir = os.path.join(_ROOT, "docs", "yolo10_runs", "yolo10_classifier_v1", "weights")
    best_pt     = os.path.join(weights_dir, "best.pt")

    stats = evaluate(best_pt, data_dir, "YOLOv10n-cls")

    print("\n" + "="*50)
    print("  SUMMARY")
    print("="*50)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<15}: {v:.4f}")
        else:
            print(f"  {k:<15}: {v}")


if __name__ == "__main__":
    main()
