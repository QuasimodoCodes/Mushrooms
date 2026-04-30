"""
Train DINOv2-Small on the mushroom dataset.

Follows the exact same structure as scripts/training/vit/train.py so
results are directly comparable — same epochs, batch size, patience,
loss functions, optimizer options, CSV logging format, and output layout.

Run from the project root:
    python scripts/training/dinov2/train.py
    python scripts/training/dinov2/train.py --optimizer adamw --loss ce_smooth

Optimizer options : adamw | sgd | radam
Loss options      : ce | ce_smooth | focal | focal_smooth

Results saved to: docs/franken_runs/dinov2_<run_name>/

DINOv2-specific training notes
-------------------------------
DINOv2 is more robust than supervised ViTs — the self-supervised pretraining
produces features that are less easily disrupted. Compared to ViT-S/16:

  1. Same base LR (1e-4) — DINOv2 was designed to be fine-tuned similarly to ViT.

  2. Same linear warmup (5 epochs) + cosine decay schedule.

  3. Same backbone/head LR split (backbone at 0.1× head LR) — even though
     DINOv2 features are more stable, we still protect the backbone from
     large gradient updates during the first head-training phase.

  4. The DINOv2 hub model uses ViT-S/14 patches (not /16), so it sees
     (224 / 14)² = 256 patches instead of 196 — slightly more spatial tokens.
"""

import argparse
import csv
import os
import sys
import time

import torch
from tqdm import tqdm

# ── Resolve project root ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

sys.path.insert(0, _HERE)
# Reuse losses from vit (identical file — no duplication needed)
sys.path.insert(0, os.path.join(_HERE, "..", "vit"))

from Mushrooms.scripts.training.dinov2.dataset import get_dataloaders
from losses  import build_criterion
from dinov2_model import build_dinov2_small

# ── Hyperparameters (identical to vit/train.py for fair comparison) ───────────
EPOCHS            = 50
BATCH_SIZE        = 32
LR                = 1e-4
WEIGHT_DECAY      = 0.05
PATIENCE          = 10
WARMUP_EPOCHS     = 5
BACKBONE_LR_SCALE = 0.1

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(_ROOT, "data", "dataset_split")


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def build_dinov2_optimizer(model, optimizer_name, lr, weight_decay):
    """
    Split parameters into backbone and head groups with different LRs.

    DINOv2 stores the classification head in model.head (same as our wrapper).
    Backbone gets 0.1× LR to preserve self-supervised representations.
    """
    head_params     = list(model.head.parameters())
    head_param_ids  = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    param_groups = [
        {"params": backbone_params, "lr": lr * BACKBONE_LR_SCALE},
        {"params": head_params,     "lr": lr},
    ]

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9,
                                weight_decay=weight_decay, nesterov=True)
    elif optimizer_name == "radam":
        return torch.optim.RAdam(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")


def build_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup then cosine decay — identical to vit/train.py."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(optimizer_name: str, loss_name: str):
    run_name    = f"dinov2_{optimizer_name}_{loss_name}"
    # Save to franken_runs to appear in compare_all.py scan alongside ViT/ConvNeXt
    output_dir  = os.path.join(_ROOT, "docs", "franken_runs", run_name)
    weights_dir = os.path.join(output_dir, "weights")
    results_csv = os.path.join(output_dir, "results.csv")

    print("==========================================")
    print(f"  DINOv2-Small | {optimizer_name.upper()} + {loss_name.upper()}")
    print("==========================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f">> GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(">> WARNING: No GPU — training will be extremely slow on CPU.")

    os.makedirs(weights_dir, exist_ok=True)

    # Data
    print(f"\nLoading dataset from: {DATA_DIR}")
    train_loader, val_loader, _, class_names = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    print(f">> {len(class_names)} classes | "
          f"{len(train_loader.dataset):,} train | "
          f"{len(val_loader.dataset):,} val")

    # Model — DINOv2 downloads weights from torch.hub on first run
    print("\nBuilding DINOv2-Small (loading from facebookresearch/dinov2 hub)...")
    print("   (First run will download ~85MB weights — subsequent runs use cache)")
    model = build_dinov2_small(num_classes=len(class_names)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">> Parameters: {total_params:,}")

    # Optimiser + scheduler
    criterion = build_criterion(loss_name)
    optimizer = build_dinov2_optimizer(model, optimizer_name, LR, WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    print(f">> Backbone LR: {LR * BACKBONE_LR_SCALE:.1e}  |  Head LR: {LR:.1e}")
    print(f">> Warmup: {WARMUP_EPOCHS} epochs  |  Cosine decay: {EPOCHS - WARMUP_EPOCHS} epochs\n")

    # CSV logging — same field names as vit/train.py so compare_all.py picks it up
    csv_fields = ["epoch", "time", "train/loss", "train/top1", "train/top5", "train/grad_norm",
                  "val/loss", "metrics/accuracy_top1", "metrics/accuracy_top5",
                  "lr/backbone", "lr/head"]
    with open(results_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    best_top1         = 0.0
    epochs_no_improve = 0

    print(f"Training up to {EPOCHS} epochs (patience={PATIENCE})...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_top1, train_top5, grad_norm = run_epoch(
            model, train_loader, criterion, optimizer, device,
            training=True, desc=f"Epoch {epoch:>3}/{EPOCHS} train")
        val_loss, val_top1, val_top5, _ = run_epoch(
            model, val_loader, criterion, optimizer, device,
            training=False, desc=f"Epoch {epoch:>3}/{EPOCHS} val  ")
        epoch_time = time.time() - t0

        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head     = optimizer.param_groups[1]["lr"]
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"time: {epoch_time:.0f}s | "
            f"train_loss: {train_loss:.4f} | train_top1: {train_top1:.1f}% | gnorm: {grad_norm:.3f} | "
            f"val_loss: {val_loss:.4f} | top1: {val_top1:.2f}% | top5: {val_top5:.2f}% | "
            f"lr_backbone: {lr_backbone:.2e} | lr_head: {lr_head:.2e}"
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
                "lr/backbone":              round(lr_backbone, 8),
                "lr/head":                  round(lr_head, 8),
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
    parser = argparse.ArgumentParser(description="Train DINOv2-Small mushroom classifier")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd", "radam"])
    parser.add_argument("--loss",      default="ce_smooth",
                        choices=["ce", "ce_smooth", "focal", "focal_smooth"])
    args = parser.parse_args()
    main(optimizer_name=args.optimizer, loss_name=args.loss)
