"""
Train ConvNeXt-Tiny on the mushroom dataset.

Run from the project root:
    python Herman/convnext/train.py
    python Herman/convnext/train.py --optimizer adamw --loss ce_smooth   # explicit defaults

Optimizer options : adamw | sgd | radam
Loss options      : ce | ce_smooth | focal | focal_smooth

Results are saved to docs/herman_runs/convnext_<run_name>/ so multiple experiments
coexist without overwriting each other.

ConvNeXt training notes
-----------------------
ConvNeXt behaves similarly to EfficientNet for fine-tuning:
  - AdamW works well with lr=4e-4 (slightly lower than EfficientNet's 1e-3 due
    to larger model capacity — prevents overshooting)
  - No warmup required (unlike ViT)
  - CosineAnnealingLR schedule, same as the existing CNN baseline
  - Weight decay 1e-4 (same as EfficientNet)
  - Gradient clipping at max_norm=1.0 (standard practice)
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from tqdm import tqdm

# ─── Resolve project root ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

sys.path.insert(0, os.path.join(_ROOT, "Herman"))
sys.path.insert(0, _HERE)

from shared.dataset import get_dataloaders
from shared.losses  import build_criterion, build_optimizer
from model          import build_convnext_tiny

# ─── Hyperparameters ──────────────────────────────────────────────────────────

EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 4e-4      # Slightly lower than EfficientNet (1e-3) for larger model
WEIGHT_DECAY = 1e-4
PATIENCE     = 10

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(_ROOT, "data", "dataset_split")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def accuracy(outputs, labels, topk=(1, 5)):
    """Return top-k accuracy percentages for a single batch."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


def run_epoch(model, loader, criterion, optimizer, device, training: bool, desc: str = ""):
    """One full pass over loader. Returns (avg_loss, avg_top1, avg_top5, avg_grad_norm)."""
    model.train(training)
    total_loss = total_top1 = total_top5 = total_gnorm = 0.0

    bar = tqdm(loader, desc=desc, leave=False, unit="batch",
               bar_format="{l_bar}{bar:30}{r_bar}")

    with torch.set_grad_enabled(training):
        for i, (imgs, labels) in enumerate(bar, 1):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            gnorm = 0.0
            if training:
                optimizer.zero_grad()
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                optimizer.step()
                total_gnorm += gnorm

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            total_loss += loss.item()
            total_top1 += top1
            total_top5 += top5

            postfix = {"loss": f"{total_loss/i:.4f}", "top1": f"{total_top1/i:.1f}%"}
            if training:
                postfix["gnorm"] = f"{total_gnorm/i:.3f}"
            bar.set_postfix(**postfix)

    n = len(loader)
    return total_loss / n, total_top1 / n, total_top5 / n, total_gnorm / n


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(optimizer_name: str, loss_name: str):
    run_name    = f"convnext_tiny_{optimizer_name}_{loss_name}"
    output_dir  = os.path.join(_ROOT, "docs", "herman_runs", run_name)
    weights_dir = os.path.join(output_dir, "weights")
    results_csv = os.path.join(output_dir, "results.csv")

    print("=========================================")
    print(f"  ConvNeXt-Tiny | {optimizer_name.upper()} + {loss_name.upper()}")
    print("=========================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f">> GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(">> WARNING: No GPU — training will be very slow on CPU.")

    os.makedirs(weights_dir, exist_ok=True)

    # Data
    print(f"\nLoading dataset from: {DATA_DIR}")
    train_loader, val_loader, _, class_names = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    print(f">> {len(class_names)} classes | "
          f"{len(train_loader.dataset):,} train | "
          f"{len(val_loader.dataset):,} val")

    # Model
    print("\nBuilding ConvNeXt-Tiny (ImageNet-1k pretrained)...")
    model = build_convnext_tiny(num_classes=len(class_names)).to(device)
    print(f">> Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss + optimiser + scheduler (same pattern as EfficientNet-B0 baseline)
    criterion = build_criterion(loss_name)
    optimizer = build_optimizer(optimizer_name, model.parameters(), LR, WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # CSV log
    csv_fields = ["epoch", "time", "train/loss", "train/top1", "train/top5", "train/grad_norm",
                  "val/loss", "metrics/accuracy_top1", "metrics/accuracy_top5", "lr/pg0"]
    with open(results_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # Training loop
    best_top1         = 0.0
    epochs_no_improve = 0

    print(f"\nTraining up to {EPOCHS} epochs (patience={PATIENCE})...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_top1, train_top5, grad_norm = run_epoch(
            model, train_loader, criterion, optimizer, device,
            training=True, desc=f"Epoch {epoch:>3}/{EPOCHS} train")
        val_loss, val_top1, val_top5, _ = run_epoch(
            model, val_loader, criterion, optimizer, device,
            training=False, desc=f"Epoch {epoch:>3}/{EPOCHS} val  ")
        epoch_time = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"time: {epoch_time:.0f}s | "
            f"train_loss: {train_loss:.4f} | train_top1: {train_top1:.1f}% | gnorm: {grad_norm:.3f} | "
            f"val_loss: {val_loss:.4f} | top1: {val_top1:.2f}% | top5: {val_top5:.2f}% | "
            f"lr: {current_lr:.2e}"
        )

        with open(results_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "epoch":                    epoch,
                "time":                     round(epoch_time, 1),
                "train/loss":               round(train_loss, 6),
                "train/top1":               round(train_top1, 4),
                "train/top5":               round(train_top5, 4),
                "train/grad_norm":          round(grad_norm, 6),
                "val/loss":                 round(val_loss, 6),
                "metrics/accuracy_top1":    round(val_top1, 4),
                "metrics/accuracy_top5":    round(val_top5, 4),
                "lr/pg0":                   round(current_lr, 8),
            })

        torch.save(model.state_dict(), os.path.join(weights_dir, "last.pt"))

        if val_top1 > best_top1:
            best_top1 = val_top1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))
            print(f"  >> New best! top1={best_top1:.2f}% — saved best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n>> Early stopping at epoch {epoch} ({PATIENCE} epochs no improvement).")
                break

    print(f"\nDone — {run_name}")
    print(f"  Best val top-1 : {best_top1:.2f}%")
    print(f"  Outputs        : {output_dir}")
    return best_top1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny mushroom classifier")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd", "radam"])
    parser.add_argument("--loss",      default="ce_smooth",
                        choices=["ce", "ce_smooth", "focal", "focal_smooth"])
    args = parser.parse_args()
    main(optimizer_name=args.optimizer, loss_name=args.loss)
