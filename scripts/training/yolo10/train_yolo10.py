from ultralytics import YOLO
import os
import torch

def main():
    print("==========================================")
    print("  Starting YOLOv10 Classification Training")
    print("==========================================")

    if torch.cuda.is_available():
        device = "cuda"
        print(f">> GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(">> WARNING: No GPU detected.")

    print("\nLoading YOLOv10n-cls pretrained weights...")
    model = YOLO("yolov10n-cls.pt")

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    data_dir = os.path.join(_ROOT, "data", "dataset_split")

    print(f"\nTraining on dataset: {data_dir}")

    results = model.train(
        data=data_dir,
        epochs=50,
        imgsz=224,
        device=device,
        exist_ok=True,
        patience=10,
        cos_lr=True,
        project=os.path.join(_ROOT, "docs", "yolo10_runs"),
        name="yolo10_classifier_v1",
    )

    print("\nTraining Complete!")
    print(f"Results saved to: docs/yolo10_runs/yolo10_classifier_v1/")

if __name__ == "__main__":
    main()