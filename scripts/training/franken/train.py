"""
Train TaxonomicYOLO26 on the mushroom dataset.

Run from the project root:
    python scripts/training/franken/train.py

Each epoch logs both Genus Top-1 and Species Top-1 so you can monitor how
well the model is learning the coarse (genus) vs fine-grained (species) task.

Outputs saved to docs/franken_runs/taxonomic_yolo26/:
    results.csv   — per-epoch metrics
    weights/
        best.pt   — weights at best species top-1
        last.pt   — weights after most recent epoch
"""

import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataloaders
from losses  import TaxonomicLoss
from model   import TaxonomicYOLO26

# ─── Defaults ─────────────────────────────────────────────────────────────────

EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-3
PATIENCE     = 10   # early stopping on species top-1

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data", "dataset_split")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    """Return top-k accuracy percentage for a single batch."""
    with torch.no_grad():
        batch_size = labels.size(0)
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(labels.view(1, -1).expand_as(pred.t()))
        return correct[:k].reshape(-1).float().sum().mul_(100.0 / batch_size).item()


def run_epoch(model, loader, criterion, optimizer, device, training: bool, desc: str = ""):
    """
    One full pass over the dataloader.

    Returns
    -------
    avg_total_loss, avg_genus_loss, avg_species_loss,
    avg_genus_top1, avg_species_top1, avg_grad_norm
    """
    model.train(training)

    total_loss = genus_loss_sum = species_loss_sum = 0.0
    genus_top1_sum = species_top1_sum = grad_norm_sum = 0.0

    bar = tqdm(loader, desc=desc, leave=False, unit="batch",
               bar_format="{l_bar}{bar:30}{r_bar}")

    with torch.set_grad_enabled(training):
        for i, (imgs, genus_labels, species_labels) in enumerate(bar, 1):
            imgs           = imgs.to(device)
            genus_labels   = genus_labels.to(device)
            species_labels = species_labels.to(device)

            genus_logits, species_logits = model(imgs)

            loss, g_loss, s_loss = criterion(
                genus_logits, species_logits,
                genus_labels, species_labels,
            )

            gnorm = 0.0
            if training:
                optimizer.zero_grad()
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                optimizer.step()
                grad_norm_sum += gnorm

            total_loss      += loss.item()
            genus_loss_sum  += g_loss.item()
            species_loss_sum += s_loss.item()
            genus_top1_sum  += topk_accuracy(genus_logits,   genus_labels,   k=1)
            species_top1_sum += topk_accuracy(species_logits, species_labels, k=1)

            postfix = {
                "loss":    f"{total_loss / i:.4f}",
                "g_top1":  f"{genus_top1_sum / i:.1f}%",
                "sp_top1": f"{species_top1_sum / i:.1f}%",
            }
            if training:
                postfix["gnorm"] = f"{grad_norm_sum / i:.3f}"
            bar.set_postfix(**postfix)

    n = len(loader)
    return (
        total_loss      / n,
        genus_loss_sum  / n,
        species_loss_sum / n,
        genus_top1_sum  / n,
        species_top1_sum / n,
        grad_norm_sum   / n,
    )


# ─── Plotting ─────────────────────────────────────────────────────────────────

def generate_plots(results_csv: str, output_dir: str):
    """Read results.csv and save a results.png mirroring the YOLO output style."""
    epochs, tr_loss, val_loss = [], [], []
    tr_g_top1, tr_s_top1     = [], []
    val_g_top1, val_s_top1   = [], []

    with open(results_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train/total_loss"]))
            val_loss.append(float(row["val/total_loss"]))
            tr_g_top1.append(float(row["train/genus_top1"]))
            tr_s_top1.append(float(row["train/species_top1"]))
            val_g_top1.append(float(row["val/genus_top1"]))
            val_s_top1.append(float(row["metrics/species_top1"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("TaxonomicYOLO26 — Training Results", fontsize=14, fontweight="bold")

    axes[0, 0].plot(epochs, tr_loss,  label="train", color="#2196F3")
    axes[0, 0].plot(epochs, val_loss, label="val",   color="#F44336")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    axes[0, 1].plot(epochs, tr_s_top1,  label="train", color="#2196F3")
    axes[0, 1].plot(epochs, val_s_top1, label="val",   color="#F44336")
    axes[0, 1].set_title("Species Top-1 Accuracy (%)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    axes[1, 0].plot(epochs, tr_g_top1,  label="train", color="#2196F3")
    axes[1, 0].plot(epochs, val_g_top1, label="val",   color="#F44336")
    axes[1, 0].set_title("Genus Top-1 Accuracy (%)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    # Train/val gap per epoch
    gap = [t - v for t, v in zip(tr_s_top1, val_s_top1)]
    axes[1, 1].plot(epochs, gap, color="#4CAF50")
    axes[1, 1].set_title("Species Train/Val Gap (%)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(output_dir, "results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

_CHECKPOINT_FILE = "checkpoint.pt"


def _resolve_output_dir(base_name: str):
    """
    Returns (output_dir, resuming).

    1. Scans all existing versioned folders newest-first for a checkpoint.pt.
       If found → resume that run (same folder, same results.csv).
    2. Otherwise find a free versioned slot for a fresh run.
    """
    franken_dir = os.path.join(_ROOT, "docs", "franken_runs")

    # Collect all existing versioned folders for this base name
    existing = []
    candidate = os.path.join(franken_dir, base_name)
    if os.path.exists(candidate):
        existing.append(candidate)
    n = 2
    while True:
        v = os.path.join(franken_dir, f"{base_name}_{n}")
        if os.path.exists(v):
            existing.append(v)
            n += 1
        else:
            break

    # Check newest first for a resumable checkpoint
    for folder in reversed(existing):
        ckpt = os.path.join(folder, "weights", _CHECKPOINT_FILE)
        if os.path.exists(ckpt):
            return folder, True

    # No checkpoint — find a free slot for a fresh run
    if not existing:
        return candidate, False

    existing_csv = os.path.join(existing[0], "results.csv")
    has_data = (os.path.exists(existing_csv) and
                os.path.getsize(existing_csv) > 200)
    if not has_data:
        return existing[0], False

    n = 2
    while True:
        versioned = os.path.join(franken_dir, f"{base_name}_{n}")
        if not os.path.exists(versioned):
            return versioned, False
        n += 1


def main():
    base_name          = "taxonomic_yolo26"
    output_dir, resuming = _resolve_output_dir(base_name)
    run_name           = os.path.basename(output_dir)
    weights_dir = os.path.join(output_dir, "weights")
    results_csv = os.path.join(output_dir, "results.csv")

    print("=========================================")
    print("  TaxonomicYOLO26 | Genus + Species Heads")
    print("=========================================")

    # 1. Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f">> GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(">> WARNING: No GPU detected — training will be very slow on CPU.")

    os.makedirs(weights_dir, exist_ok=True)

    # 2. Data
    print(f"\nLoading dataset from: {DATA_DIR}")
    train_loader, val_loader, _, meta = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    num_genera  = meta["num_genera"]
    num_species = meta["num_species"]
    print(f">> {num_genera} genera | {num_species} species | "
          f"{len(train_loader.dataset):,} train | {len(val_loader.dataset):,} val")

    # 3. Model
    print("\nBuilding TaxonomicYOLO26 (yolo26n-cls backbone)...")
    model = TaxonomicYOLO26(num_genera=num_genera, num_species=num_species).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f">> Parameters: {total_params:,}")

    # 4. Loss + optimiser + scheduler
    #
    # Differential LR: backbone gets a tiny LR (1e-5) to preserve pretrained
    # ImageNet features while only nudging them gently. The two heads get the
    # full LR (1e-3) since they're randomly initialised and need to learn fast.
    #
    # Backbone freeze for first WARMUP_EPOCHS: let the heads find a stable
    # starting point before the backbone starts moving at all.
    WARMUP_EPOCHS = 5
    for p in model._inner.parameters():
        p.requires_grad = False
    print(f">> Backbone frozen for first {WARMUP_EPOCHS} epochs.")

    criterion = TaxonomicLoss(alpha=0.3, beta=0.7, label_smoothing=0.1)
    optimizer = torch.optim.AdamW([
        {"params": model._inner.parameters(),       "lr": 1e-5},
        {"params": model.genus_head.parameters(),   "lr": LR},
        {"params": model.species_head.parameters(),"lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. CSV log
    csv_fields = [
        "epoch", "time",
        "train/total_loss", "train/genus_loss", "train/species_loss",
        "train/genus_top1", "train/species_top1", "train/grad_norm",
        "val/total_loss",   "val/genus_loss",   "val/species_loss",
        "val/genus_top1",   "metrics/species_top1",
        "lr/pg0",
    ]

    # 6. Resume or fresh start
    best_species_top1 = 0.0
    epochs_no_improve = 0
    start_epoch       = 1

    if resuming:
        ckpt_path = os.path.join(weights_dir, _CHECKPOINT_FILE)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch       = ckpt["epoch"] + 1
        best_species_top1 = ckpt["best_species_top1"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        # Restore backbone freeze state
        if not ckpt["backbone_frozen"]:
            for p in model._inner.parameters():
                p.requires_grad = True
        print(f">> Resuming from epoch {start_epoch} (best species: {best_species_top1:.2f}%)")
    else:
        with open(results_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    print(f"\nTraining up to {EPOCHS} epochs (patience={PATIENCE})...\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        # Unfreeze backbone after warmup
        if epoch == WARMUP_EPOCHS + 1:
            for p in model._inner.parameters():
                p.requires_grad = True
            print(f">> Backbone unfrozen at epoch {epoch}.")

        (tr_loss, tr_g_loss, tr_s_loss,
         tr_g_top1, tr_s_top1, grad_norm) = run_epoch(
            model, train_loader, criterion, optimizer, device,
            training=True, desc=f"Epoch {epoch:>3}/{EPOCHS} train",
        )

        (val_loss, val_g_loss, val_s_loss,
         val_g_top1, val_s_top1, _) = run_epoch(
            model, val_loader, criterion, optimizer, device,
            training=False, desc=f"Epoch {epoch:>3}/{EPOCHS} val  ",
        )

        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{EPOCHS} | time: {epoch_time:.0f}s | "
            f"train_loss: {tr_loss:.4f} | "
            f"train_genus: {tr_g_top1:.1f}% | train_species: {tr_s_top1:.1f}% | "
            f"grad_norm: {grad_norm:.3f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_genus: {val_g_top1:.2f}% | val_species: {val_s_top1:.2f}% | "
            f"lr: {current_lr:.2e}"
        )

        with open(results_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "epoch":                  epoch,
                "time":                   round(epoch_time, 1),
                "train/total_loss":       round(tr_loss,   6),
                "train/genus_loss":       round(tr_g_loss, 6),
                "train/species_loss":     round(tr_s_loss, 6),
                "train/genus_top1":       round(tr_g_top1, 4),
                "train/species_top1":     round(tr_s_top1, 4),
                "train/grad_norm":        round(grad_norm, 6),
                "val/total_loss":         round(val_loss,   6),
                "val/genus_loss":         round(val_g_loss, 6),
                "val/species_loss":       round(val_s_loss, 6),
                "val/genus_top1":         round(val_g_top1, 4),
                "metrics/species_top1":   round(val_s_top1, 4),
                "lr/pg0":                 round(current_lr, 8),
            })

        torch.save(model.state_dict(), os.path.join(weights_dir, "last.pt"))

        # Save full checkpoint so training can be resumed if interrupted
        torch.save({
            "epoch":              epoch,
            "model":              model.state_dict(),
            "optimizer":          optimizer.state_dict(),
            "scheduler":          scheduler.state_dict(),
            "best_species_top1":  best_species_top1,
            "epochs_no_improve":  epochs_no_improve,
            "backbone_frozen":    epoch < WARMUP_EPOCHS,
        }, os.path.join(weights_dir, _CHECKPOINT_FILE))

        if val_s_top1 > best_species_top1:
            best_species_top1 = val_s_top1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))
            print(f"  >> New best! species_top1={best_species_top1:.2f}% — saved best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n>> Early stopping at epoch {epoch} ({PATIENCE} epochs no improvement).")
                break

    # Remove checkpoint — training finished cleanly, no need to resume
    ckpt_path = os.path.join(weights_dir, _CHECKPOINT_FILE)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"\nDone — {run_name}")
    print(f"  Best val species top-1 : {best_species_top1:.2f}%")
    print(f"  Outputs                : {output_dir}")

    print("\nGenerating plots...")
    generate_plots(results_csv, output_dir)


if __name__ == "__main__":
    main()
