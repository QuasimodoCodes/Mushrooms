"""
Compare YOLOv26 PyTorch (best.pt) vs TFLite float16 (best_float16.tflite)
on the validation set. Generates side-by-side comparison plots.
"""

from ultralytics import YOLO
import os
import time
import matplotlib.pyplot as plt
import numpy as np


def run_validation(model_path, data_dir, label):
    """Run YOLO val on the given model and return the results dict."""
    print(f"\n{'='*50}")
    print(f"  Evaluating: {label}")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"{'='*50}")

    model = YOLO(model_path, task="classify")

    start = time.time()
    results = model.val(data=data_dir, imgsz=224, batch=1, verbose=False)
    elapsed = time.time() - start

    top1 = results.results_dict.get("metrics/accuracy_top1", 0)
    top5 = results.results_dict.get("metrics/accuracy_top5", 0)

    print(f"  Top-1 Accuracy: {top1:.4f}")
    print(f"  Top-5 Accuracy: {top5:.4f}")
    print(f"  Eval time:      {elapsed:.1f}s")

    return {
        "label": label,
        "top1": top1,
        "top5": top5,
        "eval_time": elapsed,
        "file_size_mb": os.path.getsize(model_path) / (1024 * 1024),
    }


def plot_comparison(pt_stats, tflite_stats, output_dir):
    """Generate comparison bar charts and save them."""
    labels = [pt_stats["label"], tflite_stats["label"]]
    colors = ["#2196F3", "#FF9800"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("YOLOv26 PyTorch vs TFLite (float16) — Validation Set", fontsize=14, fontweight="bold")

    # 1. Top-1 Accuracy
    vals = [pt_stats["top1"] * 100, tflite_stats["top1"] * 100]
    bars = axes[0].bar(labels, vals, color=colors, width=0.5)
    axes[0].set_title("Top-1 Accuracy (%)")
    axes[0].set_ylim(0, 100)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{v:.2f}%", ha="center", fontweight="bold")

    # 2. Top-5 Accuracy
    vals = [pt_stats["top5"] * 100, tflite_stats["top5"] * 100]
    bars = axes[1].bar(labels, vals, color=colors, width=0.5)
    axes[1].set_title("Top-5 Accuracy (%)")
    axes[1].set_ylim(0, 100)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{v:.2f}%", ha="center", fontweight="bold")

    # 3. Model File Size
    vals = [pt_stats["file_size_mb"], tflite_stats["file_size_mb"]]
    bars = axes[2].bar(labels, vals, color=colors, width=0.5)
    axes[2].set_title("Model File Size (MB)")
    for bar, v in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{v:.2f}", ha="center", fontweight="bold")

    # 4. Evaluation Time
    vals = [pt_stats["eval_time"], tflite_stats["eval_time"]]
    bars = axes[3].bar(labels, vals, color=colors, width=0.5)
    axes[3].set_title("Eval Time on Val Set (seconds)")
    for bar, v in zip(bars, vals):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{v:.1f}s", ha="center", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pt_vs_tflite_comparison.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nComparison plot saved to: {plot_path}")

    # Summary table plot
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.axis("off")
    table_data = [
        ["Metric", pt_stats["label"], tflite_stats["label"], "Difference"],
        ["Top-1 Accuracy", f"{pt_stats['top1']*100:.2f}%", f"{tflite_stats['top1']*100:.2f}%", f"{(tflite_stats['top1']-pt_stats['top1'])*100:+.2f}%"],
        ["Top-5 Accuracy", f"{pt_stats['top5']*100:.2f}%", f"{tflite_stats['top5']*100:.2f}%", f"{(tflite_stats['top5']-pt_stats['top5'])*100:+.2f}%"],
        ["File Size", f"{pt_stats['file_size_mb']:.2f} MB", f"{tflite_stats['file_size_mb']:.2f} MB", f"{(tflite_stats['file_size_mb']-pt_stats['file_size_mb']):+.2f} MB"],
        ["Eval Time", f"{pt_stats['eval_time']:.1f}s", f"{tflite_stats['eval_time']:.1f}s", f"{(tflite_stats['eval_time']-pt_stats['eval_time']):+.1f}s"],
    ]
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(4):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig2.suptitle("PyTorch vs TFLite (float16) — Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    table_path = os.path.join(output_dir, "pt_vs_tflite_table.png")
    fig2.savefig(table_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Summary table saved to:  {table_path}")


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    weights_dir = os.path.join(base, "docs", "yolo_runs", "yolo26_classifier_v1", "weights")
    data_dir = os.path.join(base, "data", "dataset_split")
    output_dir = os.path.join(base, "docs", "yolo_runs", "yolo26_classifier_v1")

    pt_path = os.path.join(weights_dir, "best.pt")
    tflite_path = os.path.join(weights_dir, "best_float16.tflite")

    for p, name in [(pt_path, "best.pt"), (tflite_path, "best_float16.tflite")]:
        if not os.path.exists(p):
            print(f"ERROR: {name} not found at {p}")
            return

    pt_stats = run_validation(pt_path, data_dir, "PyTorch (best.pt)")
    tflite_stats = run_validation(tflite_path, data_dir, "TFLite float16")

    plot_comparison(pt_stats, tflite_stats, output_dir)

    print("\n" + "="*50)
    print("  DONE — All plots generated!")
    print("="*50)


if __name__ == "__main__":
    main()
