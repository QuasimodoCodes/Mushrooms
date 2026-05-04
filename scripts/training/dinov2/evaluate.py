"""
Evaluate a trained DINOv2-Small on the held-out test set.

Run from the project root AFTER training:
    python scripts/training/dinov2/evaluate.py --run dinov2_adamw_ce_smooth

--run must match a folder name inside docs/franken_runs/.

Outputs (saved into the run folder):
    - Top-1 and Top-5 accuracy printed to terminal
    - confusion_matrix_normalized.png
    - top_errors.txt  (20 most-confused class pairs)
    - per_class_accuracy.txt
"""

import argparse
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

sys.path.insert(0, _HERE)

from dataset import get_dataloaders
from model   import build_dinov2_small

DATA_DIR = os.path.join(_ROOT, "data", "dataset_split")


def build_confusion_matrix(model, loader, num_classes, device):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu()
            for t, p in zip(labels, preds):
                matrix[t, p] += 1
    return matrix


def plot_confusion_matrix(matrix, class_names, save_path, title="DINOv2-Small"):
    norm = matrix.float()
    row_sums = norm.sum(dim=1, keepdim=True).clamp(min=1)
    norm = norm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm.numpy(), interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix (Normalized) — {title}", fontsize=14, pad=12)
    ax.set_xlabel("Predicted class index", fontsize=11)
    ax.set_ylabel("True class index", fontsize=11)

    step = max(1, len(class_names) // 20)
    ax.set_xticks(range(0, len(class_names), step))
    ax.set_yticks(range(0, len(class_names), step))
    ax.set_xticklabels(range(0, len(class_names), step), fontsize=7, rotation=90)
    ax.set_yticklabels(range(0, len(class_names), step), fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved: {save_path}")


def top_errors(matrix, class_names, n=20):
    errors = []
    for true_cls in range(len(class_names)):
        for pred_cls in range(len(class_names)):
            if true_cls != pred_cls and matrix[true_cls, pred_cls] > 0:
                errors.append((
                    matrix[true_cls, pred_cls].item(),
                    class_names[true_cls],
                    class_names[pred_cls],
                ))
    errors.sort(reverse=True)
    return errors[:n]


def per_class_accuracy(matrix, class_names):
    results = []
    for i, name in enumerate(class_names):
        total = matrix[i].sum().item()
        correct = matrix[i, i].item()
        acc = correct / total if total > 0 else 0.0
        results.append((acc, name, correct, total))
    results.sort()
    return results


def main(run_name: str):
    run_dir     = os.path.join(_ROOT, "docs", "franken_runs", run_name)
    weights_dir = os.path.join(run_dir, "weights")
    best_pt     = os.path.join(weights_dir, "best.pt")

    if not os.path.exists(best_pt):
        raise FileNotFoundError(
            f"Weights not found: {best_pt}\n"
            f"Train first: python scripts/training/dinov2/train.py"
        )

    print(f"\n{'='*55}")
    print(f"  DINOv2-Small Evaluation — {run_name}")
    print(f"{'='*55}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    _, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR, batch_size=64)
    num_classes = len(class_names)

    model = build_dinov2_small(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(best_pt, map_location=device))
    model.eval()
    print(f"Loaded weights from: {best_pt}")

    # ── Accuracy on test set ──────────────────────────────────────────────────
    total = correct_top1 = correct_top5 = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            _, top5_pred = logits.topk(5, dim=1)
            correct_top1 += (top5_pred[:, 0] == labels).sum().item()
            correct_top5 += sum(labels[i] in top5_pred[i] for i in range(len(labels)))
            total += labels.size(0)

    top1_acc = correct_top1 / total * 100
    top5_acc = correct_top5 / total * 100
    print(f"\n  Test Top-1 Accuracy : {top1_acc:.2f}%")
    print(f"  Test Top-5 Accuracy : {top5_acc:.2f}%")
    print(f"  Total test samples  : {total:,}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print("\nBuilding confusion matrix on test set...")
    matrix = build_confusion_matrix(model, test_loader, num_classes, device)

    cm_path = os.path.join(run_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix(matrix, class_names, cm_path)

    # ── Top errors ────────────────────────────────────────────────────────────
    errors_path = os.path.join(run_dir, "top_errors.txt")
    errors = top_errors(matrix, class_names, n=20)
    with open(errors_path, "w") as f:
        f.write(f"Top confusions — {run_name}\n")
        f.write(f"{'Count':<8} {'True class':<40} {'Predicted class'}\n")
        f.write("-" * 80 + "\n")
        for count, true_cls, pred_cls in errors:
            f.write(f"{count:<8} {true_cls:<40} {pred_cls}\n")
    print(f"Top errors saved:  {errors_path}")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    pca_path = os.path.join(run_dir, "per_class_accuracy.txt")
    pca = per_class_accuracy(matrix, class_names)
    with open(pca_path, "w") as f:
        f.write(f"Per-class accuracy (sorted ascending) — {run_name}\n")
        f.write(f"{'Acc':>7}  {'Correct':>7}  {'Total':>7}  Class\n")
        f.write("-" * 60 + "\n")
        for acc, name, correct, total_cls in pca:
            f.write(f"{acc*100:6.1f}%  {correct:>7}  {total_cls:>7}  {name}\n")
    print(f"Per-class accuracy : {pca_path}")

    print(f"\n  Done — all outputs in: {run_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DINOv2-Small mushroom classifier")
    parser.add_argument("--run", default="dinov2_adamw_ce_smooth",
                        help="Run folder name inside docs/franken_runs/")
    args = parser.parse_args()
    main(run_name=args.run)
