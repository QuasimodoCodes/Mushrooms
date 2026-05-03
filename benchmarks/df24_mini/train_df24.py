from ultralytics import YOLO
import os
import torch
from pathlib import Path

def main():
    print("=========================================")
    print("  DF24-Mini Benchmark — YOLOv26n Fine-tune")
    print("=========================================")

    if torch.cuda.is_available():
        device = 'cuda'
        print(">> GPU detected — training hardware-accelerated.")
    else:
        device = 'cpu'
        print(">> WARNING: No GPU detected. CPU training will be very slow.")

    BASE     = Path(__file__).parent
    DATA_DIR = BASE / "data" / "yolo_ready"
    OUT_DIR  = BASE / "runs"

    if not DATA_DIR.exists():
        print(f"ERROR: Dataset not found at {DATA_DIR}")
        print("Run prepare_dataset.py first.")
        return

    print(f"\nDataset: {DATA_DIR}")
    print("Loading yolo26n-cls.pt...")
    model = YOLO('yolo26n-cls.pt')

    results = model.train(
        data=str(DATA_DIR),
        epochs=50,
        imgsz=224,
        device=device,
        exist_ok=True,
        patience=10,
        cos_lr=True,
        project=str(OUT_DIR),
        name="yolo26n_df24mini"
    )

    print("\nTraining complete!")
    print(f"Weights and metrics saved to: {OUT_DIR / 'yolo26n_df24mini'}")

if __name__ == "__main__":
    main()
