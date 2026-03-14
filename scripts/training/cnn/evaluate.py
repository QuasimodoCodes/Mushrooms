"""
Evaluate a trained EfficientNet-B0 on the held-out test set.

Run from the project root AFTER training is complete:
    python scripts/training/cnn/evaluate.py --run efficientnet_b0_adamw_ce_smooth

--run must match the folder name inside docs/cnn_runs/.

Outputs (saved into that run's folder):
    - Per-class accuracy printed to terminal
    - confusion_matrix_normalized.png
    - top_errors.txt  — the 20 most-confused class pairs
"""

import argparse
import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")  # no display required — works headless in Cloud Run / CI
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataloaders
from model import build_efficientnet_b0

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data", "dataset_split")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_confusion_matrix(model, loader, num_classes, device):
    """
    Accumulate a (num_classes × num_classes) confusion matrix over the loader.
    Entry [true, pred] is incremented for each sample.
    """
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu()
            for t, p in zip(labels, preds):
                matrix[t, p] += 1

    return matrix


def top_k_accuracy(matrix, k=5):
    """
    Derive top-k accuracy from the confusion matrix.
    NOTE: top-k > 1 cannot be recovered from a 2-D confusion matrix alone,
    so we compute top-5 directly from model logits in the main loop instead.
    This helper computes top-1 from the matrix.
    """
    correct = matrix.diag().sum().item()
    total   = matrix.sum().item()
    return 100.0 * correct / total


def plot_confusion_matrix(matrix, class_names, save_path):
    """
    Save a normalised confusion matrix heatmap.
    With 169 classes the labels are too dense to show individually —
    we render a clean heatmap so you can spot clusters of confusion.
    """
    norm = matrix.float()
    row_sums = norm.sum(dim=1, keepdim=True).clamp(min=1)
    norm = norm / row_sums  # each row sums to 1.0

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(norm.numpy(), interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix (Normalized) — EfficientNet-B0", fontsize=14, pad=12)
    ax.set_xlabel("Predicted class index", fontsize=11)
    ax.set_ylabel("True class index", fontsize=11)

    # Show every 10th tick label so the axis stays readable
    step = max(1, len(class_names) // 20)
    ax.set_xticks(range(0, len(class_names), step))
    ax.set_yticks(range(0, len(class_names), step))
    ax.set_xticklabels(range(0, len(class_names), step), fontsize=7, rotation=90)
    ax.set_yticklabels(range(0, len(class_names), step), fontsize=7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f">> Confusion matrix saved: {save_path}")


def find_top_errors(matrix, class_names, n=20):
    """Return the n most-confused off-diagonal (true, pred) pairs."""
    off_diag = matrix.clone()
    off_diag.fill_diagonal_(0)
    flat = off_diag.view(-1)
    top_indices = flat.topk(n).indices

    errors = []
    num_classes = matrix.size(0)
    for idx in top_indices:
        true_cls = idx.item() // num_classes
        pred_cls = idx.item() %  num_classes
        count    = matrix[true_cls, pred_cls].item()
        errors.append((count, class_names[true_cls], class_names[pred_cls]))

    errors.sort(reverse=True)
    return errors


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained EfficientNet-B0 run")
    parser.add_argument(
        "--run", required=True,
        help="Run folder name inside docs/cnn_runs/ (e.g. efficientnet_b0_adamw_ce_smooth)",
    )
    args = parser.parse_args()

    run_dir = os.path.join(_ROOT, "docs", "cnn_runs", args.run)
    best_pt = os.path.join(run_dir, "weights", "best.pt")

    print("=========================================")
    print("  Evaluating EfficientNet-B0 on Test Set ")
    print(f"  Run: {args.run}")
    print("=========================================")

    if not os.path.exists(best_pt):
        print(f"ERROR: No weights found at {best_pt}")
        print("       Run train.py first.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Device: {device}")

    # Data — test split only
    _, _, test_loader, class_names = get_dataloaders(DATA_DIR)
    num_classes = len(class_names)
    print(f">> Test samples: {len(test_loader.dataset):,}  |  Classes: {num_classes}")

    # Model
    print(f"\nLoading weights from: {best_pt}")
    model = build_efficientnet_b0(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(best_pt, map_location=device))
    model.eval()

    # ── Pass 1: top-1 & top-5 accuracy ──────────────────────────────────────
    correct_top1 = 0
    correct_top5 = 0
    total        = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # top-1
            preds_top1 = logits.argmax(dim=1)
            correct_top1 += preds_top1.eq(labels).sum().item()

            # top-5
            top5_preds = logits.topk(5, dim=1).indices
            correct_top5 += sum(
                labels[i].item() in top5_preds[i].tolist()
                for i in range(labels.size(0))
            )
            total += labels.size(0)

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total
    print(f"\nTest Results")
    print(f"  Top-1 Accuracy : {top1:.2f}%")
    print(f"  Top-5 Accuracy : {top5:.2f}%")
    print(f"  Total samples  : {total:,}")

    # ── Pass 2: confusion matrix ─────────────────────────────────────────────
    print("\nBuilding confusion matrix (this takes a moment)...")
    matrix = build_confusion_matrix(model, test_loader, num_classes, device)

    cm_path = os.path.join(run_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix(matrix, class_names, cm_path)

    # ── Top errors ──────────────────────────────────────────────────────────
    errors = find_top_errors(matrix, class_names, n=20)
    errors_path = os.path.join(run_dir, "top_errors.txt")
    with open(errors_path, "w") as f:
        f.write(f"Top 20 most-confused pairs (EfficientNet-B0 test set)\n")
        f.write(f"Top-1: {top1:.2f}%  |  Top-5: {top5:.2f}%\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Count':>6}  {'True class':<40}  Predicted as\n")
        f.write("-" * 60 + "\n")
        for count, true_cls, pred_cls in errors:
            f.write(f"{count:>6}  {true_cls:<40}  {pred_cls}\n")

    print(f">> Top errors saved: {errors_path}")

    # ── Per-class accuracy (terminal) ────────────────────────────────────────
    print("\nPer-class accuracy (sorted best → worst):")
    per_class = []
    for i, name in enumerate(class_names):
        n_true = matrix[i].sum().item()
        n_correct = matrix[i, i].item()
        acc = 100.0 * n_correct / n_true if n_true > 0 else 0.0
        per_class.append((acc, name, n_true))

    per_class.sort(reverse=True)
    for acc, name, n in per_class:
        bar = "#" * int(acc / 5)  # 20-char bar for 100%
        print(f"  {acc:5.1f}%  {bar:<20}  {name}  (n={n})")

    print(f"\nDone. All outputs in: {run_dir}")


if __name__ == "__main__":
    main()
