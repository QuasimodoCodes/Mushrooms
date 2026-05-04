"""
Ablation Study — Figure Generator
===================================
Reads ablation/results/ablation_raw.csv and produces a 3-panel figure:
  Panel 1 — Safety Recall per model (vision-only vs +Gemini LLM)
  Panel 2 — High-Confidence Misclassification Recall (Gemini only, primary metric)
  Panel 3 — Median inference time per model (vision model only)

Output: ablation/results/ablation_figure.png
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "ablation_raw.csv"
CONTEXT_CSV = ROOT / "data" / "mushroom_context.csv"
OUT_PATH    = RESULTS_DIR / "ablation_figure.png"

DANGEROUS_KEYWORDS = ["deadly", "highly toxic", "toxic", "hallucinogenic", "pathogenic"]

MODEL_ORDER = ["YOLOv26n", "ConvNeXt-Tiny", "DINOv2-Small"]
MODEL_LABELS = {
    "YOLOv26n":      "YOLOv26n\n(edge/mobile)",
    "ConvNeXt-Tiny": "ConvNeXt-Tiny\n(PEFT)",
    "DINOv2-Small":  "DINOv2-Small\n(ViT-S/14)",
}

COLOR_NONE   = "#4C72B0"   # blue   — vision only
COLOR_GEMINI = "#DD8452"   # orange — +Gemini
COLOR_M2     = "#C44E52"   # red    — primary metric (HC misclassification)
COLOR_SPEED  = "#55A868"   # green  — speed bars


def is_dangerous(toxicity: str) -> bool:
    t = str(toxicity).lower()
    return any(k in t for k in DANGEROUS_KEYWORDS)


def load_context():
    ctx = pd.read_csv(CONTEXT_CSV)
    ctx["species_key"] = ctx["species_name"].str.replace(" ", "_").str.lower()
    return ctx.set_index("species_key")["toxicity_type"].to_dict()

def load_data():
    if not CSV_PATH.exists():
        sys.exit(f"No results found at {CSV_PATH}. Run ablation/run_ablation.py first.")
    df = pd.read_csv(CSV_PATH)
    # Filter to dangerous-only rows for Metric 1 & 2
    if "dangerous" in df.columns:
        df = df[df["dangerous"] == True].copy()
    print(f"Loaded {len(df)} dangerous-species rows, {df['image'].nunique()} unique images.")
    return df


def compute_recall(df):
    rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        for llm in ("none", "gemini"):
            g = sub[sub["llm"] == llm]
            if g.empty:
                continue
            recall = g["flagged"].mean() * 100
            rows.append({"model": model, "llm": llm, "recall": recall, "n": len(g)})
    return pd.DataFrame(rows)


def compute_speed(df):
    rows = []
    for model in MODEL_ORDER:
        g = df[(df["model"] == model) & (df["llm"] == "none")]
        if g.empty or "inference_ms" not in g.columns:
            continue
        # Drop warmup outliers: exclude values > 3× the median
        med = g["inference_ms"].median()
        g_clean = g[g["inference_ms"] <= med * 3]
        rows.append({
            "model":  model,
            "median": g_clean["inference_ms"].median(),
            "mean":   g_clean["inference_ms"].mean(),
            "n_dropped": len(g) - len(g_clean),
        })
    return pd.DataFrame(rows)


def compute_metric2(df, toxicity_map):
    """High-confidence misclassification recall — the primary clean metric."""
    rows = []
    pred_key = df["pred_class"].str.lower().str.replace(" ", "_")
    pred_dangerous = pred_key.map(lambda x: is_dangerous(toxicity_map.get(x, "")))
    hc_miss = df[(df["confidence"] >= 0.70) & (~pred_dangerous)]

    for model in MODEL_ORDER:
        g = hc_miss[(hc_miss["model"] == model) & (hc_miss["llm"] == "gemini")]
        if g.empty:
            continue
        recall = g["flagged"].mean() * 100
        rows.append({"model": model, "recall": recall, "n": len(g)})
    return pd.DataFrame(rows)


def plot(recall_df, speed_df, m2_df):
    has_speed = len(speed_df) > 0
    n_panels  = 2 if has_speed else 1
    ratios    = [1.6, 1] if has_speed else None

    fig, axes = plt.subplots(1, n_panels, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": ratios} if ratios else None)
    if n_panels == 1:
        axes = [axes]

    ax1 = axes[0]
    ax2 = None
    ax3 = axes[1] if has_speed else None

    # ── Panel 1: Safety Recall grouped bar ───────────────────────────────────
    models_present = [m for m in MODEL_ORDER if m in recall_df["model"].values]
    x = np.arange(len(models_present))
    width = 0.35

    none_vals   = [recall_df[(recall_df["model"] == m) & (recall_df["llm"] == "none")  ]["recall"].values[0]
                   if len(recall_df[(recall_df["model"] == m) & (recall_df["llm"] == "none")]) > 0 else 0
                   for m in models_present]
    gemini_vals = [recall_df[(recall_df["model"] == m) & (recall_df["llm"] == "gemini")]["recall"].values[0]
                   if len(recall_df[(recall_df["model"] == m) & (recall_df["llm"] == "gemini")]) > 0 else 0
                   for m in models_present]

    bars_none   = ax1.bar(x - width / 2, none_vals,   width, color=COLOR_NONE,   label="No LLM audit",         zorder=3)
    bars_gemini = ax1.bar(x + width / 2, gemini_vals, width, color=COLOR_GEMINI, label="With Gemini visual audit", zorder=3)

    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Dangerous species caught (%)", fontsize=12)
    ax1.set_title("Dangerous Species Caught by the Pipeline\n(Higher is better)", fontsize=12, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_present], fontsize=10)
    ax1.yaxis.grid(True, alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.text(0.02, 0.01,
             "\"No LLM audit\" = confidence threshold rule only\n(predictions <70% confidence auto-flagged HIGH)",
             transform=ax1.transAxes, fontsize=7.5, va="bottom", color="#666666",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.8))

    # Value labels + LLM lift annotations
    for bar, nv, gv in zip(bars_none, none_vals, gemini_vals):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, color=COLOR_NONE, fontweight="bold")
    for bar, nv, gv in zip(bars_gemini, none_vals, gemini_vals):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, color=COLOR_GEMINI, fontweight="bold")
        lift = gv - nv
        if lift > 0:
            mid_x = bar.get_x() + bar.get_width() / 2
            ax1.annotate(f"+{lift:.0f}pp", xy=(mid_x, gv + 5.5),
                         ha="center", va="bottom", fontsize=8,
                         color="#888888", style="italic")

    # ── Panel 2: Inference Speed horizontal bar ───────────────────────────────
    if ax3 is not None and len(speed_df) > 0:
        sp_models = [m for m in MODEL_ORDER if m in speed_df["model"].values]
        sp_medians = [speed_df[speed_df["model"] == m]["median"].values[0] for m in sp_models]
        y = np.arange(len(sp_models))

        bars_sp = ax3.barh(y, sp_medians, color=COLOR_SPEED, height=0.5, zorder=3)
        ax3.set_xlabel("Median inference time (ms)", fontsize=12)
        ax3.set_title("Vision Model Inference Speed\n(Lower is better, excl. LLM)", fontsize=12, pad=10)
        ax3.set_yticks(y)
        ax3.set_yticklabels([MODEL_LABELS.get(m, m) for m in sp_models], fontsize=10)
        ax3.xaxis.grid(True, alpha=0.4, zorder=0)
        ax3.set_axisbelow(True)

        for bar, val in zip(bars_sp, sp_medians):
            ax3.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f} ms", va="center", ha="left", fontsize=10, fontweight="bold")

        ax3.set_xlim(0, max(sp_medians) * 1.35)

    n_images = recall_df["n"].max() if len(recall_df) else 0
    fig.suptitle(
        f"Ablation: Vision Model × LLM Safety Evaluation  (n={n_images} images/combo)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {OUT_PATH}")
    plt.close(fig)


def main():
    df          = load_data()
    toxicity_map = load_context() if CONTEXT_CSV.exists() else {}
    recall_df   = compute_recall(df)
    speed_df    = compute_speed(df)
    m2_df       = compute_metric2(df, toxicity_map)

    print("\nMetric 1 — Safety Recall:")
    print(recall_df.to_string(index=False))

    print("\nMetric 2 — HC Misclassification Recall (primary):")
    print(m2_df.to_string(index=False))

    if len(speed_df):
        print("\nInference speed (warmup outliers excluded where >3x median):")
        print(speed_df.to_string(index=False))

    plot(recall_df, speed_df, m2_df)


if __name__ == "__main__":
    main()
