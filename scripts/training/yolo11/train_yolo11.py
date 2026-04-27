"""
Train YOLOv11n-cls on the mushroom dataset.

Follows the exact same structure as scripts/training/yolo/train_yolo.py
but uses the YOLOv11 nano classification backbone.

YOLOv11 vs YOLOv8/v26
-----------------------
YOLOv11 (Wang et al., 2024) introduces NMS-free training via dual-label
assignment (one-to-many for training, one-to-one for inference). For
classification tasks this means:

  - More efficient inference (no post-processing bottleneck)
  - Comparable or slightly better accuracy on fine-grained tasks
  - Same Ultralytics API — drop-in replacement for YOLOv8

Run from the project root:
    python scripts/training/yolo11/train_yolo11.py

Results are saved to docs/yolo11_runs/yolo11_classifier_v1/
"""

import os
import torch
from ultralytics import YOLO


def main():
    print("==========================================")
    print("  Starting YOLOv11 Classification Training")
    print("==========================================")

    # ── 1. Hardware check ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        print(f">> SUCCESS: GPU detected — {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(">> WARNING: No GPU detected. Falling back to slow CPU training.")

    # ── 2. Load pretrained YOLOv11n classification backbone ──────────────────
    # 'yolov11n-cls.pt' is downloaded automatically from Ultralytics HUB on
    # first run (requires internet). The nano variant has ~2.3M parameters —
    # similar size to YOLOv8n-cls, but trained with the v11 dual-assignment head.
    print("\nLoading YOLOv11n-cls pretrained weights...")
    model = YOLO("yolo11n-cls.pt")

    # ── 3. Dataset path ───────────────────────────────────────────────────────
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    data_dir = os.path.join(_ROOT, "data", "dataset_split")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset not found at: {data_dir}\n"
            "Run the dataset setup script first."
        )

    # ── 4. Training ───────────────────────────────────────────────────────────
    print(f"\nTraining on dataset: {data_dir}")
    print("Watch 'loss' decrease — early stopping triggers after 11 flat epochs.\n")

    results = model.train(
        data=data_dir,
        epochs=50,
        imgsz=224,
        device=device,
        exist_ok=True,

        # ── Regularisation (same as YOLOv26 run for fair comparison) ─────────
        patience=1,
        cos_lr=True,           # Cosine LR decay — helps escape local optima
        dropout=0.0,           # YOLOv11 cls head already has BN; no extra dropout

        # ── Output ────────────────────────────────────────────────────────────
        project=os.path.join(_ROOT, "docs", "yolo11_runs"),
        name="yolo11_classifier_v1",
    )

    print("\n==========================================")
    print("  YOLOv11 Training Complete!")
    print("==========================================")
    print(f"  Best val top-1 : {results.results_dict.get('metrics/accuracy_top1', 'N/A')}")
    print(f"  Outputs saved  : docs/yolo11_runs/yolo11_classifier_v1/")


if __name__ == "__main__":
    main()
