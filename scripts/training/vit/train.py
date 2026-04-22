"""
Train ViT-S/16 on the mushroom dataset.

Run from the project root:
    python scripts/training/vit/train.py
    python scripts/training/vit/train.py --optimizer adamw --loss ce_smooth   # explicit defaults

Optimizer options : adamw | sgd | radam
Loss options      : ce | ce_smooth | focal | focal_smooth

Results are saved to docs/herman_runs/vit_<run_name>/ so multiple experiments
coexist without overwriting each other.

ViT-specific training notes
---------------------------
ViTs are more sensitive to learning rate than CNNs. Two key differences
from the EfficientNet-B0 training script:

  1. Lower base LR (1e-4 vs 1e-3) — ViT fine-tuning diverges at CNN-style LRs.

  2. Linear warmup for the first WARMUP_EPOCHS epochs, followed by cosine decay.
     Without warmup, the randomly-initialised classification head generates
     large gradients in early steps that corrupt the pretrained backbone weights.

  3. Separate LR for backbone vs head (layer-wise LR):
     - Backbone: lr * BACKBONE_LR_SCALE  (default 0.1x — gentle fine-tuning)
     - Head:     lr                       (full LR — head is randomly initialised)
     This prevents the backbone from being updated too aggressively while the
     head is still learning from scratch.
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
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

sys.path.insert(0, _HERE)

from dataset import get_dataloaders
from losses  import build_criterion, build_optimizer
from model   import build_vit_small

# ─── Hyperparameters ──────────────────────────────────────────────────────────

EPOCHS           = 50
BATCH_SIZE       = 32
LR               = 1e-4       # ViTs need a smaller LR than CNNs
WEIGHT_DECAY     = 0.05       # Higher weight decay is standard for ViT (AdamW)
PATIENCE         = 10
WARMUP_EPOCHS    = 5          # Linear LR warmup — critical for stable ViT fine-tuning
BACKBONE_LR_SCALE = 0.1       # Backbone updated at 10% of the head LR

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


def build_vit_optimizer(model, optimizer_name, lr, weight_decay):
    """
    Split parameters into backbone and head groups with different LRs.

    ViTs benefit from a lower LR on the pretrained backbone to avoid
    catastrophic forgetting of ImageNet features.
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
    """
    Linear warmup for warmup_epochs, then cosine decay to near-zero.

    Using SequentialLR to chain two schedulers:
      Phase 1: LinearLR  (start_factor=0.01 → 1.0 over warmup_epochs steps)
      Phase 2: CosineAnnealingLR (T_max = remaining epochs)
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(optimizer_name: str, loss_name: str):
    run_name    = f"vit_small_{optimizer_name}_{loss_name}"
    output_dir  = os.path.join(_ROOT, "docs", "herman_runs", run_name)
    weights_dir = os.path.join(output_dir, "weights")
    results_csv = os.path.join(output_dir, "results.csv")

    print("=========================================")
    print(f"  ViT-S/16 | {optimizer_name.upper()} + {loss_name.upper()}")
    print("=========================================")

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

    # Model
    print("\nBuilding ViT-S/16 (ImageNet-21k pretrained via timm)...")
    model = build_vit_small(num_classes=len(class_names)).to(device)
    print(f">> Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimiser (split backbone / head LRs)
    criterion = build_criterion(loss_name)
    optimizer = build_vit_optimizer(model, optimizer_name, LR, WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    print(f">> Backbone LR: {LR * BACKBONE_LR_SCALE:.1e}  |  Head LR: {LR:.1e}")
    print(f">> Warmup: {WARMUP_EPOCHS} epochs  |  Cosine decay: {EPOCHS - WARMUP_EPOCHS} epochs\n")

    # CSV log
    csv_fields = ["epoch", "time", "train/loss", "train/top1", "train/top5", "train/grad_norm",
                  "val/loss", "metrics/accuracy_top1", "metrics/accuracy_top5", "lr/backbone", "lr/head"]
    with open(results_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # Training loop
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

        # Read LRs before stepping
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
    parser = argparse.ArgumentParser(description="Train ViT-S/16 mushroom classifier")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd", "radam"])
    parser.add_argument("--loss",      default="ce_smooth",
                        choices=["ce", "ce_smooth", "focal", "focal_smooth"])
    args = parser.parse_args()
    main(optimizer_name=args.optimizer, loss_name=args.loss)
