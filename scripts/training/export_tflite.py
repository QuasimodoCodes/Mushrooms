from ultralytics import YOLO
import os


def main():
    print("=========================================")
    print("   Exporting YOLOv26 to TFLite Format    ")
    print("=========================================")

    # 1. Locate the trained model weights
    weights_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "docs", "yolo_runs", "yolo26_classifier_v1", "weights", "best.pt")
    )

    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at: {weights_path}")
        print("Make sure you've trained the model first using train_yolo.py")
        return

    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)

    # 2. Export to TFLite
    # Ultralytics handles the full PyTorch → ONNX → TF SavedModel → TFLite conversion pipeline.
    # imgsz must match what the model was trained on (224 for our classifier).
    print("\nStarting TFLite export (this may take a few minutes)...")
    model.export(
        format="tflite",
        imgsz=224,
    )

    print("\nExport complete!")
    print("TFLite files are in the best_saved_model/ subdirectory.")
    print("Copy them to the weights/ folder for cleaner organization:")
    saved_model_dir = os.path.join(os.path.dirname(weights_path), "best_saved_model")
    for f in os.listdir(saved_model_dir):
        if f.endswith(".tflite"):
            src = os.path.join(saved_model_dir, f)
            dst = os.path.join(os.path.dirname(weights_path), f)
            import shutil
            shutil.copy2(src, dst)
            print(f"  Copied: {f} -> weights/{f}")
    print("\nYou can now deploy these .tflite files to mobile devices or edge hardware.")


if __name__ == "__main__":
    main()
