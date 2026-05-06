"""
Ablation Study — Figure Generator (v2)
=========================================
Produces two publication-quality figures from ablation/results/ablation_raw.csv:

  Figure 1 — ablation_performance.png
    Panel A: Safety Recall on dangerous species   (all 7 conditions)
    Panel B: Precision                            (all 7 conditions)
    Panel C: F1 Score                             (all 7 conditions)

  Figure 2 — ablation_speed.png
    Vision model inference speed (ms) — LLM excluded (cloud API, model-independent)
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "ablation_raw.csv"
OUT_PERF    = RESULTS_DIR / "ablation_performance.png"
OUT_SPEED   = RESULTS_DIR / "ablation_speed.png"

DANGEROUS_KEYWORDS = ["deadly", "highly toxic", "toxic", "hallucinogenic", "pathogenic"]

# (model, llm, display_label, color)  — order defines bar order left→right
CONDITIONS = [
    ("YOLOv26n",      "none",   "YOLO\n(vision only)",    "#4C72B0"),
    ("ConvNeXt-Tiny", "none",   "ConvNeXt\n(vision only)","#5B8DB8"),
    ("DINOv2-Small",  "none",   "DINOv2\n(vision only)",  "#6AACD4"),
    ("LLM Only",      "gemini", "Gemini\n(LLM only)",     "#9B59B6"),
    ("YOLOv26n",      "gemini", "YOLO\n+ Gemini",         "#DD8452"),
    ("ConvNeXt-Tiny", "gemini", "ConvNeXt\n+ Gemini",     "#E8905A"),
    ("DINOv2-Small",  "gemini", "DINOv2\n+ Gemini",       "#C44E52"),
]

SPEED_MODELS  = ["YOLOv26n", "ConvNeXt-Tiny", "DINOv2-Small"]
SPEED_COLORS  = {"YOLOv26n": "#4C72B0", "ConvNeXt-Tiny": "#5B8DB8", "DINOv2-Small": "#6AACD4"}
GROUP_LABELS  = {("vision", 0): "Vision only", ("llm", 3): "LLM only", ("combined", 4): "Vision + Gemini"}


def is_dangerous(toxicity: str) -> bool:
    return any(k in str(toxicity).lower() for k in DANGEROUS_KEYWORDS)


def load_data():
    if not CSV_PATH.exists():
        sys.exit(f"No results at {CSV_PATH}. Run run_ablation.py first.")
    df = pd.read_csv(CSV_PATH)
    if "dangerous" not in df.columns:
        df["dangerous"] = df["true_toxicity"].apply(lambda t: is_dangerous(str(t)))
    print(f"Loaded {len(df)} rows — "
          f"{df[df['dangerous']]['image'].nunique()} dangerous images, "
          f"{df[~df['dangerous']]['image'].nunique()} safe images.")
    return df


def compute_metrics(df):
    danger_df = df[df["dangerous"] == True]
    safe_df   = df[df["dangerous"] == False]
    has_safe  = len(safe_df) > 0

    rows = []
    for model, llm, label, color in CONDITIONS:
        d = danger_df[(danger_df["model"] == model) & (danger_df["llm"] == llm)]
        s = safe_df[(safe_df["model"] == model) & (safe_df["llm"] == llm)] if has_safe else pd.DataFrame()

        if d.empty:
            continue

        tp = int(d["flagged"].sum())
        fn = int((~d["flagged"]).sum())
        fp = int(s["flagged"].sum())  if not s.empty else None
        tn = int((~s["flagged"]).sum()) if not s.empty else None

        recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) * 100 if (fp is not None and (tp + fp) > 0) else None
        f1        = (2 * precision * recall / (precision + recall)) \
                    if (precision is not None and (precision + recall) > 0) else None
        fpr       = fp / (fp + tn) * 100 if (fp is not None and tn is not None and (fp + tn) > 0) else None

        rows.append({
            "model": model, "llm": llm, "label": label, "color": color,
            "recall": recall, "precision": precision, "f1": f1, "fpr": fpr,
            "n_danger": len(d), "n_safe": len(s) if not s.empty else 0,
        })
    return pd.DataFrame(rows)


def compute_speed(df):
    rows = []
    for model in SPEED_MODELS:
        g = df[(df["model"] == model) & (df["llm"] == "none") & (df["inference_ms"] > 0)]
        if g.empty:
            continue
        med     = g["inference_ms"].median()
        g_clean = g[g["inference_ms"] <= med * 3]
        rows.append({"model": model, "median": g_clean["inference_ms"].median()})
    return pd.DataFrame(rows)


def _add_group_dividers(ax, metrics_df):
    """Draw dashed vertical lines between vision-only / LLM-only / combined groups."""
    models = metrics_df["model"].tolist()
    llms   = metrics_df["llm"].tolist()

    def group(m, l):
        if m == "LLM Only":  return "llm"
        return "vision" if l == "none" else "combined"

    prev = None
    for i, (m, l) in enumerate(zip(models, llms)):
        g = group(m, l)
        if prev and g != prev:
            ax.axvline(i - 0.5, color="#bbbbbb", linewidth=1.2, linestyle="--", zorder=2)
        prev = g


def bar_panel(ax, metrics_df, col, title, ylabel):
    present = metrics_df[metrics_df[col].notna()].reset_index(drop=True)
    if present.empty:
        ax.set_visible(False)
        return

    x    = np.arange(len(present))
    bars = ax.bar(x, present[col], color=present["color"].tolist(),
                  width=0.62, zorder=3, edgecolor="white", linewidth=0.8)

    _add_group_dividers(ax, present)

    ax.set_ylim(0, 118)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(title, fontsize=11, pad=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(present["label"].tolist(), fontsize=8.5)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, present[col]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#222222")


def plot_performance(metrics_df, n_images):
    has_precision = metrics_df["precision"].notna().any()
    n_panels      = 3 if has_precision else 1
    fig, axes     = plt.subplots(1, n_panels, figsize=(5.5 * n_panels + 1, 6.2))
    if n_panels == 1:
        axes = [axes]

    # F1 leads — balances recall and precision into one honest score
    bar_panel(axes[0], metrics_df, "f1",
              "F1 Score (↑ better)\nBalances catching danger vs. over-flagging safe species",
              "F1 (%)")

    if has_precision and n_panels >= 3:
        bar_panel(axes[1], metrics_df, "recall",
                  "Safety Recall (↑ better)\nHow many dangerous species were correctly flagged?",
                  "Dangerous species flagged (%)")

        # Add FPR warning to the LLM-only x-axis label — high recall is misleading without it
        llm_row = metrics_df[metrics_df["model"] == "LLM Only"]
        if not llm_row.empty and llm_row["fpr"].notna().any():
            present  = metrics_df[metrics_df["recall"].notna()].reset_index(drop=True)
            llm_idx  = list(present["model"]).index("LLM Only")
            fpr      = llm_row["fpr"].values[0]
            labels   = [t.get_text() for t in axes[1].get_xticklabels()]
            if llm_idx < len(labels):
                labels[llm_idx] = f"Gemini\n(LLM only)\n⚠ FPR: {fpr:.0f}%"
                axes[1].set_xticklabels(labels, fontsize=8.5)
                axes[1].get_xticklabels()[llm_idx].set_color("#C0392B")

        bar_panel(axes[2], metrics_df, "fpr",
                  "False Positive Rate (↓ better)\nHow often were safe species wrongly flagged?",
                  "Safe species incorrectly flagged (%)")

    legend_patches = [
        mpatches.Patch(color="#4C72B0", label="Vision only"),
        mpatches.Patch(color="#9B59B6", label="LLM only (Gemini)"),
        mpatches.Patch(color="#DD8452", label="Vision + Gemini"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 1.03), frameon=True,
               edgecolor="#cccccc")

    fig.suptitle(
        f"Ablation Study: Vision Model × LLM Safety Evaluation  "
        f"(n≈{n_images} dangerous images/condition)",
        fontsize=12, fontweight="bold", y=1.08,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PERF, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PERF}")
    plt.close(fig)


def plot_speed(speed_df):
    if speed_df.empty:
        print("No speed data — skipping ablation_speed.png")
        return

    fig, ax  = plt.subplots(figsize=(6, 3.5))
    models   = speed_df["model"].tolist()
    medians  = speed_df["median"].tolist()
    colors   = [SPEED_COLORS.get(m, "#4C72B0") for m in models]
    y        = np.arange(len(models))

    bars = ax.barh(y, medians, color=colors, height=0.5, zorder=3, edgecolor="white")
    ax.set_xlabel("Median inference time (ms)", fontsize=11)
    ax.set_title("Vision Model Inference Speed\n"
                 "(Lower is better — LLM API latency excluded, model-independent)",
                 fontsize=11, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=10)
    ax.xaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, medians):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} ms", va="center", ha="left",
                fontsize=10, fontweight="bold", color="#222222")
    ax.set_xlim(0, max(medians) * 1.45)

    plt.tight_layout()
    fig.savefig(OUT_SPEED, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_SPEED}")
    plt.close(fig)


def main():
    df         = load_data()
    metrics_df = compute_metrics(df)
    speed_df   = compute_speed(df)

    print("\nPerformance Metrics:")
    print(metrics_df[["label", "recall", "precision", "f1",
                       "n_danger", "n_safe"]].to_string(index=False))

    n_images = int(metrics_df["n_danger"].max()) if len(metrics_df) else 0
    plot_performance(metrics_df, n_images)
    plot_speed(speed_df)


if __name__ == "__main__":
    main()
