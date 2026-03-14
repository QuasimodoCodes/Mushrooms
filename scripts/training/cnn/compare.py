"""
Run all optimizer × loss combinations, track every epoch, and produce
publication-quality comparison plots for the professor.

Usage (from project root):
    python scripts/training/cnn/compare.py          # train all + plot
    python scripts/training/cnn/compare.py --plot-only  # plot from existing runs

Outputs in docs/cnn_runs/:
    comparison_results.csv          — final top-1/top-5 for every combo
    plots/comparison_grid.png       — heatmap: optimizer × loss → top-1 accuracy
    plots/accuracy_curves.png       — val top-1 over epochs for all 12 runs
    plots/loss_curves.png           — val loss over epochs for all 12 runs
    plots/accuracy_vs_time.png      — accuracy vs training time scatter
"""

import argparse
import csv
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from train import main as train_run

# ─── Experiment grid ──────────────────────────────────────────────────────────

OPTIMIZERS = ["adamw", "sgd", "radam"]
LOSSES     = ["ce", "ce_smooth", "focal", "focal_smooth"]

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
CNN_RUNS_DIR = os.path.join(_ROOT, "docs", "cnn_runs")
PLOTS_DIR    = os.path.join(CNN_RUNS_DIR, "plots")
RESULTS_CSV  = os.path.join(CNN_RUNS_DIR, "comparison_results.csv")

# Consistent colours per optimizer across all plots
OPT_COLORS = {"adamw": "#2196F3", "sgd": "#F44336", "radam": "#4CAF50"}

# Line styles per loss function so lines are distinguishable even in greyscale
LOSS_STYLES = {"ce": "-", "ce_smooth": "--", "focal": "-.", "focal_smooth": ":"}


# ─── Training ─────────────────────────────────────────────────────────────────

def run_all_experiments():
    """Train every optimizer × loss pair. Returns list of result dicts."""
    total = len(OPTIMIZERS) * len(LOSSES)
    results = []

    for i, opt in enumerate(OPTIMIZERS):
        for j, loss in enumerate(LOSSES):
            run_num = i * len(LOSSES) + j + 1
            print(f"\n{'═'*60}")
            print(f"  Run {run_num}/{total}  |  optimizer={opt}  loss={loss}")
            print(f"{'═'*60}\n")

            t0 = time.time()
            best_top1 = train_run(optimizer_name=opt, loss_name=loss)
            elapsed   = time.time() - t0

            results.append({
                "optimizer":  opt,
                "loss":       loss,
                "best_top1":  round(best_top1, 4),
                "time_min":   round(elapsed / 60, 1),
            })

            _print_leaderboard(results)

    return results


# ─── Result loading ───────────────────────────────────────────────────────────

def load_results_from_disk():
    """
    Rebuild the results list by scanning existing run folders.
    Used when --plot-only is passed so you don't have to re-train.
    """
    results = []
    for opt in OPTIMIZERS:
        for loss in LOSSES:
            run_name    = f"efficientnet_b0_{opt}_{loss}"
            results_csv = os.path.join(CNN_RUNS_DIR, run_name, "results.csv")
            if not os.path.exists(results_csv):
                print(f"  [skip] No results found for {run_name}")
                continue

            best_top1 = 0.0
            with open(results_csv) as f:
                for row in csv.DictReader(f):
                    best_top1 = max(best_top1, float(row["metrics/accuracy_top1"]))

            results.append({
                "optimizer": opt,
                "loss":      loss,
                "best_top1": round(best_top1, 4),
                "time_min":  0.0,   # not tracked in CSV, leave as 0
            })

    return results


def read_epoch_curves(opt, loss):
    """Return (epochs, top1_list, val_loss_list) from a run's results.csv."""
    run_name    = f"efficientnet_b0_{opt}_{loss}"
    results_csv = os.path.join(CNN_RUNS_DIR, run_name, "results.csv")
    if not os.path.exists(results_csv):
        return [], [], []

    epochs, top1s, losses = [], [], []
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            top1s.append(float(row["metrics/accuracy_top1"]))
            losses.append(float(row["val/loss"]))

    return epochs, top1s, losses


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison_grid(results):
    """
    Heatmap: rows = optimizers, cols = loss functions, cells = best val top-1.
    Immediately shows which axis (optimizer vs loss) has the bigger impact.
    """
    grid = np.zeros((len(OPTIMIZERS), len(LOSSES)))
    for r in results:
        i = OPTIMIZERS.index(r["optimizer"])
        j = LOSSES.index(r["loss"])
        grid[i, j] = r["best_top1"]

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, cmap="YlGn", vmin=grid[grid > 0].min() - 1, vmax=grid.max() + 0.5)

    ax.set_xticks(range(len(LOSSES)))
    ax.set_yticks(range(len(OPTIMIZERS)))
    ax.set_xticklabels(LOSSES, fontsize=12)
    ax.set_yticklabels(OPTIMIZERS, fontsize=12)
    ax.set_xlabel("Loss Function", fontsize=13, labelpad=10)
    ax.set_ylabel("Optimizer", fontsize=13, labelpad=10)
    ax.set_title("EfficientNet-B0 — Val Top-1 Accuracy (%) by Optimizer × Loss", fontsize=13, pad=14)

    # Annotate each cell with its accuracy value
    for i in range(len(OPTIMIZERS)):
        for j in range(len(LOSSES)):
            val = grid[i, j]
            if val > 0:
                color = "white" if val > grid.mean() else "black"
                ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Top-1 Accuracy (%)", fraction=0.046, pad=0.04)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "comparison_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


def plot_accuracy_curves():
    """
    All 12 val top-1 curves on one axes.
    Colour = optimizer, line style = loss function.
    A legend explains both dimensions.
    """
    fig, ax = plt.subplots(figsize=(13, 7))

    for opt in OPTIMIZERS:
        for loss in LOSSES:
            epochs, top1s, _ = read_epoch_curves(opt, loss)
            if not epochs:
                continue
            label = f"{opt} + {loss}"
            ax.plot(epochs, top1s,
                    color=OPT_COLORS[opt],
                    linestyle=LOSS_STYLES[loss],
                    linewidth=1.8,
                    label=label)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Val Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("EfficientNet-B0 — Validation Accuracy Curves (all runs)", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "accuracy_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


def plot_loss_curves():
    """Same layout as accuracy_curves but for validation loss."""
    fig, ax = plt.subplots(figsize=(13, 7))

    for opt in OPTIMIZERS:
        for loss in LOSSES:
            epochs, _, val_losses = read_epoch_curves(opt, loss)
            if not epochs:
                continue
            label = f"{opt} + {loss}"
            ax.plot(epochs, val_losses,
                    color=OPT_COLORS[opt],
                    linestyle=LOSS_STYLES[loss],
                    linewidth=1.8,
                    label=label)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("EfficientNet-B0 — Validation Loss Curves (all runs)", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "loss_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


def plot_accuracy_vs_time(results):
    """
    Scatter plot: x = training time (min), y = best top-1 accuracy.
    Lets you see accuracy/time tradeoffs at a glance.
    Skipped when time data isn't available (--plot-only mode).
    """
    has_time = any(r["time_min"] > 0 for r in results)
    if not has_time:
        print("  [skip] accuracy_vs_time.png — no timing data (use --plot-only after a fresh run)")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for r in results:
        color  = OPT_COLORS[r["optimizer"]]
        marker = {"ce": "o", "ce_smooth": "s", "focal": "^", "focal_smooth": "D"}[r["loss"]]
        ax.scatter(r["time_min"], r["best_top1"],
                   color=color, marker=marker, s=120, zorder=3,
                   label=f"{r['optimizer']} + {r['loss']}")
        ax.annotate(f"  {r['optimizer']}\n  {r['loss']}",
                    (r["time_min"], r["best_top1"]),
                    fontsize=7, color=color)

    ax.set_xlabel("Training Time (minutes)", fontsize=12)
    ax.set_ylabel("Best Val Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("EfficientNet-B0 — Accuracy vs Training Time", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "accuracy_vs_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


def plot_final_bar(results):
    """
    Horizontal bar chart sorted best → worst.
    Makes the ranking immediately obvious for a slide or poster.
    """
    ranked = sorted(results, key=lambda r: r["best_top1"])
    labels = [f"{r['optimizer']} + {r['loss']}" for r in ranked]
    values = [r["best_top1"] for r in ranked]
    colors = [OPT_COLORS[r["optimizer"]] for r in ranked]

    fig, ax = plt.subplots(figsize=(10, max(5, len(ranked) * 0.5)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)

    # Value label on each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=10)

    ax.set_xlabel("Best Val Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("EfficientNet-B0 — Final Accuracy Ranking", fontsize=13)

    # Colour legend for optimizers
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=o) for o, c in OPT_COLORS.items()]
    ax.legend(handles=legend_elements, title="Optimizer", loc="lower right", fontsize=10)

    # Tighten x-axis around the data range
    margin = 1.5
    ax.set_xlim(max(0, min(values) - margin), max(values) + margin + 2)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "final_accuracy_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Saved: {path}")


def generate_all_plots(results):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"\nGenerating comparison plots → {PLOTS_DIR}\n")
    plot_comparison_grid(results)
    plot_accuracy_curves()
    plot_loss_curves()
    plot_accuracy_vs_time(results)
    plot_final_bar(results)


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_results_csv(results):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "loss", "best_top1", "time_min"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults table saved: {RESULTS_CSV}")


# ─── Leaderboard ──────────────────────────────────────────────────────────────

def _print_leaderboard(results):
    ranked = sorted(results, key=lambda r: r["best_top1"], reverse=True)
    print("\n  -- Leaderboard so far --------------------------------------")
    print(f"  {'#':>3}  {'Optimizer':<8}  {'Loss':<14}  {'Top-1':>6}  {'Time':>7}")
    print(f"  {'---'}  {'--------'}  {'---' + '-'*11}  {'------'}  {'-------'}")
    for rank, r in enumerate(ranked, 1):
        marker = "  <- best" if rank == 1 else ""
        print(f"  {rank:>3}  {r['optimizer']:<8}  {r['loss']:<14}  "
              f"{r['best_top1']:>5.2f}%  {r['time_min']:>6.1f}m{marker}")
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare all optimizer × loss combinations")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training — load existing run CSVs and regenerate plots only")
    args = parser.parse_args()

    os.makedirs(CNN_RUNS_DIR, exist_ok=True)

    if args.plot_only:
        print("Loading existing run results from disk...")
        results = load_results_from_disk()
        if not results:
            print("ERROR: No completed runs found. Run without --plot-only first.")
            return
    else:
        print("=" * 60)
        print(f"  EfficientNet-B0  |  {len(OPTIMIZERS)} optimizers × {len(LOSSES)} losses")
        print("=" * 60)
        results = run_all_experiments()

    save_results_csv(results)
    _print_leaderboard(results)
    generate_all_plots(results)

    best = max(results, key=lambda r: r["best_top1"])
    print(f"\nBest combination:  {best['optimizer']} + {best['loss']}  →  {best['best_top1']:.2f}% top-1")
    print(f"Evaluate it with:")
    print(f"  python scripts/training/cnn/evaluate.py "
          f"--run efficientnet_b0_{best['optimizer']}_{best['loss']}")


if __name__ == "__main__":
    main()
