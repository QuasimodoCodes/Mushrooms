"""
Ablation Study — Vision Model x LLM Safety Evaluation
=======================================================
Evaluates every combination of vision model and LLM against a held-out test
set, measuring safety recall on toxic/deadly species.

Matrix:
    Vision models : YOLOv8n | YOLOv26n | EfficientNet-B0
    LLM configs   : None (vision only) | Llama3 | Gemma | Gemini
"""

import os, sys, json, argparse, random
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── Project paths ──────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
SCRIPTS_DIR  = ROOT / "scripts" / "training"
PIPELINE_DIR = ROOT / "services" / "brain_ui" / "pipeline"
DATA_DIR     = ROOT / "data" / "dataset_split"
CONTEXT_CSV  = ROOT / "data" / "mushroom_context.csv"
RESULTS_DIR  = Path(__file__).parent / "results"

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "cnn"))

# ── Model weights ──────────────────────────────────────────────────────────────
WEIGHTS = {
    "yolo8n":   ROOT / "docs/yolo_runs/mushroom_classifier_v1/weights/best.pt",
    "yolo26n":  ROOT / "docs/yolo_runs/yolo26_classifier_v1/weights/best.pt",
    "efficientnet": ROOT / "docs/cnn_runs/efficientnet_b0_adamw_ce_smooth/weights/best.pt",
}

# ── Toxicity categories considered dangerous ───────────────────────────────────
DANGEROUS_KEYWORDS = ["deadly", "highly toxic", "toxic", "hallucinogenic", "pathogenic"]


def is_dangerous(toxicity: str) -> bool:
    t = toxicity.lower()
    return any(k in t for k in DANGEROUS_KEYWORDS)


# ── Load context CSV ───────────────────────────────────────────────────────────
def load_context():
    df = pd.read_csv(CONTEXT_CSV)
    df["species_key"] = df["species_name"].str.replace(" ", "_").str.lower()
    return df.set_index("species_key").to_dict("index")


# ── Load vision models ─────────────────────────────────────────────────────────
def load_yolo(path):
    from ultralytics import YOLO
    model = YOLO(str(path))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def load_efficientnet(path):
    from model import build_efficientnet_b0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_efficientnet_b0()
    net.load_state_dict(torch.load(str(path), map_location=device))
    net.eval().to(device)
    class_names = sorted(os.listdir(DATA_DIR / "train"))
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return net, class_names, tfm, device


# ── Inference functions ────────────────────────────────────────────────────────
def predict_yolo(model, img_path):
    results = model(str(img_path), verbose=False)
    probs = results[0].probs
    idx = int(probs.top1)
    return model.names[idx].replace("_", " "), float(probs.top1conf)


def predict_efficientnet(model, class_names, tfm, device, img_path):
    img = Image.open(img_path).convert("RGB")
    x   = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return class_names[idx].replace("_", " "), float(probs[idx])


# ── LLM query ─────────────────────────────────────────────────────────────────
def query_llm_for_ablation(provider: str, ollama_model: str,
                            prompt: str, image_path=None):
    import llm_provider as lp
    if provider == "none":
        return "✅ PLAUSIBLE"
    elif provider == "gemini":
        return lp._query_gemini(prompt, image_path=image_path)
    elif provider in ("llama", "gemma"):
        original_model = lp.OLLAMA_MODEL
        lp.OLLAMA_MODEL = ollama_model
        result = lp._query_ollama(prompt)
        lp.OLLAMA_MODEL = original_model
        return result
    return "✅ PLAUSIBLE"


# ── Risk engine ────────────────────────────────────────────────────────────────
def get_risk(species_name, confidence, context_row, llm_verdict):
    from risk_engine import assess_risk
    ctx = {
        "toxicity_type": context_row.get("toxicity_type", "Unknown"),
        "habitat":       context_row.get("habitat", "Unknown"),
        "season":        context_row.get("season", "Unknown"),
        "region":        context_row.get("region", "Unknown"),
        "key_warnings":  context_row.get("key_warnings", "None"),
    }
    return assess_risk(species_name, confidence, ctx, llm_verdict)


# ── Build audit prompt ─────────────────────────────────────────────────────────
def build_prompt(species_name, confidence, context_row):
    from audit_layer import build_audit_prompt
    ctx = {k: context_row.get(k, "Unknown") for k in
           ["toxicity_type", "habitat", "season", "region", "key_warnings"]}
    return build_audit_prompt(species_name, confidence, ctx,
                              user_season="Unknown", user_location="Unknown")


# ── Collect test images ────────────────────────────────────────────────────────
def collect_test_images(context, samples_per_class):
    test_dir   = DATA_DIR / "test"
    image_list = []

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        species_key  = class_dir.name.replace(" ", "_").lower()
        ctx_row      = context.get(species_key, {})
        toxicity     = ctx_row.get("toxicity_type", "Unknown")
        dangerous    = is_dangerous(toxicity)

        imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        random.shuffle(imgs)
        selected = imgs[:samples_per_class]

        if not dangerous:
            continue

        for img in selected:
            image_list.append({
                "path":       img,
                "true_class": class_dir.name,
                "toxicity":   toxicity,
                "dangerous":  dangerous,
                "ctx_row":    ctx_row,
            })

    print(f"Collected {len(image_list)} test images "
          f"({sum(1 for i in image_list if i['dangerous'])} from dangerous classes)")
    return image_list


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20,
                        help="Max images per class from test set")
    parser.add_argument("--llms", nargs="+",
                        default=["none", "llama", "gemma"],
                        choices=["none", "llama", "gemma"],
                        help="LLM providers to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 55)
    print("  Ablation Study — Vision x LLM Safety Evaluation")
    print("=" * 55)
    print(f"LLMs to evaluate : {args.llms}")
    print(f"Samples per class: {args.samples}\n")

    context    = load_context()
    test_images = collect_test_images(context, args.samples)

    # Load vision models
    print("\nLoading vision models...")
    yolo8   = load_yolo(WEIGHTS["yolo8n"])   if WEIGHTS["yolo8n"].exists()   else None
    yolo26  = load_yolo(WEIGHTS["yolo26n"])  if WEIGHTS["yolo26n"].exists()  else None
    eff_b0  = load_efficientnet(WEIGHTS["efficientnet"]) if WEIGHTS["efficientnet"].exists() else None

    vision_models = {}
    if yolo8:   vision_models["YOLOv8n"]        = ("yolo", yolo8)
    if yolo26:  vision_models["YOLOv26n"]       = ("yolo", yolo26)
    if eff_b0:  vision_models["EfficientNet-B0"] = ("eff",  eff_b0)

    OLLAMA_MODELS = {"llama": "llama3.2", "gemma": "gemma4:e2b"}

    rows = []
    out_csv = RESULTS_DIR / "ablation_raw.csv"

    for i, item in enumerate(tqdm(test_images, desc="Evaluating")):
        img_path    = item["path"]
        true_class  = item["true_class"]
        toxicity    = item["toxicity"]
        ctx_row     = item["ctx_row"]

        for model_name, (model_type, model_obj) in vision_models.items():
            # Run vision model
            try:
                if model_type == "yolo":
                    pred_class, confidence = predict_yolo(model_obj, img_path)
                else:
                    net, class_names, tfm, device = model_obj
                    pred_class, confidence = predict_efficientnet(net, class_names, tfm, device, img_path)
            except Exception as e:
                continue

            pred_ctx  = context.get(pred_class.replace(" ", "_").lower(), {})

            # Skip LLM if confidence < 0.70 — risk engine always flags HIGH
            # via the confidence threshold regardless of LLM verdict.
            # LLM only matters when vision is confident but potentially wrong.
            needs_llm = confidence >= 0.70
            prompt    = build_prompt(pred_class, confidence, pred_ctx) if needs_llm else None

            for llm_name in args.llms:
                ollama_model = OLLAMA_MODELS.get(llm_name, "")
                try:
                    if not needs_llm or llm_name == "none":
                        llm_verdict = "✅ PLAUSIBLE"
                    else:
                        llm_verdict = query_llm_for_ablation(llm_name, ollama_model, prompt)
                    risk_result = get_risk(pred_class, confidence, pred_ctx, llm_verdict)
                    risk_level  = risk_result["risk_level"]
                except Exception as e:
                    llm_verdict = f"ERROR: {e}"
                    risk_level  = "ERROR"

                flagged = risk_level in ("HIGH", "CRITICAL")

                rows.append({
                    "image":        img_path.name,
                    "true_class":   true_class,
                    "true_toxicity": toxicity,
                    "model":        model_name,
                    "llm":          llm_name,
                    "pred_class":   pred_class,
                    "confidence":   round(confidence, 4),
                    "risk_level":   risk_level,
                    "flagged":      flagged,
                })

        # Save checkpoint every 1 images
        if i % 1 == 0 and rows:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Save raw results
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_csv, index=False)
    print(f"\nRaw results saved to {out_csv}")

    # ── Metric 1: Overall safety recall ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Metric 1 — Overall Safety Recall on Dangerous Species")
    print("=" * 65)
    print(f"{'Model':<20} {'LLM':<12} {'Safety Recall':>14} {'False Safe Rate':>16}")
    print("-" * 65)

    summary_rows = []

    for (model, llm), grp in results_df.groupby(["model", "llm"]):
        recall     = grp["flagged"].mean() * 100
        false_safe = 100 - recall
        print(f"{model:<20} {llm:<12} {recall:>13.1f}% {false_safe:>15.1f}%")
        summary_rows.append({
            "model": model, "llm": llm,
            "metric": "overall_safety_recall",
            "safety_recall_%": round(recall, 2),
            "false_safe_rate_%": round(false_safe, 2),
            "n_images": len(grp),
        })

    # ── Metric 2: High-confidence misclassification recall ────────────────────
    # The LLM's real contribution: catching cases where YOLO is confident
    # but wrong — predicting a safe species when the true class is dangerous.
    hc_miss = results_df[
        (results_df["confidence"] >= 0.70) &
        (~results_df["pred_class"].str.lower().str.replace(" ", "_")
         .map(lambda x: is_dangerous(context.get(x, {}).get("toxicity_type", ""))))
    ]

    print(f"\n{'='*65}")
    print("  Metric 2 — High-Confidence Misclassification Recall")
    print("  (YOLO ≥70% confident but predicted a safe species)")
    print(f"  Total cases: {len(hc_miss) // max(1, len(results_df['model'].unique()) * len(results_df['llm'].unique()))}")
    print(f"{'='*65}")
    print(f"{'Model':<20} {'LLM':<12} {'Safety Recall':>14} {'N Cases':>8}")
    print("-" * 65)

    for (model, llm), grp in hc_miss.groupby(["model", "llm"]):
        recall = grp["flagged"].mean() * 100 if len(grp) > 0 else 0.0
        print(f"{model:<20} {llm:<12} {recall:>13.1f}% {len(grp):>8}")
        summary_rows.append({
            "model": model, "llm": llm,
            "metric": "hc_misclassification_recall",
            "safety_recall_%": round(recall, 2),
            "false_safe_rate_%": round(100 - recall, 2),
            "n_images": len(grp),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to {summary_csv}")


if __name__ == "__main__":
    main()
