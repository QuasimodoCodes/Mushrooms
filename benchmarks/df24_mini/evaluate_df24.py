"""
Evaluates the fine-tuned YOLOv26n on the DF24-Mini validation set.
Calculates Top-1, Top-3 accuracy and macro F1 to match the published baselines.
"""

import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

BASE      = Path(__file__).parent
VAL_DIR   = BASE / "data" / "yolo_ready" / "val"
WEIGHTS   = BASE / "runs" / "yolo26n_df24mini" / "weights" / "best.pt"


def evaluate():
    print("=========================================")
    print("  DF24-Mini Evaluation — Top-1 & Top-3   ")
    print("=========================================")

    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        return

    model = YOLO(str(WEIGHTS))
    class_names = model.names  # {0: 'Agaricus_augustus', ...}

    # Build ground truth map: folder name -> class index used by the model
    folder_to_idx = {v: k for k, v in class_names.items()}

    top1_correct = 0
    top3_correct = 0
    total = 0
    skipped = 0
    all_true = []
    all_pred = []

    species_dirs = sorted(VAL_DIR.iterdir())

    for species_dir in tqdm(species_dirs, desc="Evaluating"):
        if not species_dir.is_dir():
            continue

        true_label = species_dir.name  # e.g. "Russula_fragilis"
        if true_label not in folder_to_idx:
            skipped += 1
            continue
        true_idx = folder_to_idx[true_label]

        for img_path in species_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            results = model(str(img_path), verbose=False)
            probs   = results[0].probs

            top1_idx = int(probs.top1)
            top5_idxs = probs.top5  # list of 5 indices

            if top1_idx == true_idx:
                top1_correct += 1

            if true_idx in top5_idxs[:3]:
                top3_correct += 1

            all_true.append(true_idx)
            all_pred.append(top1_idx)
            total += 1

    if total == 0:
        print("No images evaluated.")
        return

    top1 = top1_correct / total * 100
    top3 = top3_correct / total * 100

    classes = set(all_true)
    f1_per_class = []
    for c in classes:
        tp = sum(1 for t, p in zip(all_true, all_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(all_true, all_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(all_true, all_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_per_class.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0)
    f1 = sum(f1_per_class) / len(f1_per_class) * 100

    print(f"\n{'='*55}")
    print(f"  Results on DF24-Mini validation set")
    print(f"{'='*55}")
    print(f"  Images evaluated : {total}")
    print(f"  Skipped classes  : {skipped}")
    print(f"  Top-1 Accuracy   : {top1:.2f}%")
    print(f"  Top-3 Accuracy   : {top3:.2f}%")
    print(f"  Macro F1         : {f1:.2f}%")
    print(f"{'='*55}")
    print(f"\nPublished baselines (224x224):")
    print(f"  ViT-Large/16     : 67.52% Top-1 | 84.46% Top-3 | 55.90% F1")
    print(f"  ViT-Base/16      : 65.33% Top-1 | 82.44% Top-3 | 52.28% F1")
    print(f"  SE-ResNeXt-101   : 62.42% Top-1 | 80.71% Top-3 | 50.01% F1")
    print(f"  EfficientNet-B3  : 59.31% Top-1 | 78.79% Top-3 | 47.83% F1")
    print(f"  EfficientNet-B0  : 58.58% Top-1 | 77.01% Top-3 | 46.00% F1")
    print(f"  YOLOv26n (ours)  : {top1:.2f}% Top-1 | {top3:.2f}% Top-3 | {f1:.2f}% F1")


if __name__ == "__main__":
    evaluate()
