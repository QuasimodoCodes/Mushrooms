"""
Ablation Study — Vision Model x LLM Safety Evaluation
=======================================================
Evaluates every combination of vision model and LLM against a held-out test
set, measuring safety recall on toxic/deadly species.

Matrix:
    Vision models : YOLOv26n | ConvNeXt-Tiny (PEFT) | DINOv2-Small
    LLM configs   : None (vision only) | Gemini Flash
"""

import os, sys, json, argparse, random
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
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

# ── Model weights ──────────────────────────────────────────────────────────────
WEIGHTS = {
    "yolo26n":   ROOT / "docs/yolo_runs/yolo26_classifier_v1/weights/best.pt",
    "convnext":  ROOT / "docs/herman_runs/convnext_tiny_adamw_ce_smooth_peft/weights/best.pt",
    "dinov2":    ROOT / "docs/dinov2/weights/best.pt",
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


# ── Standard ImageNet transforms (shared by all CNN-type models) ───────────────
_IMAGENET_TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Load vision models ─────────────────────────────────────────────────────────
def load_yolo(path):
    from ultralytics import YOLO
    model = YOLO(str(path))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def _import_model_module(subdir):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"{subdir}_model", SCRIPTS_DIR / subdir / "model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_convnext(path):
    mod = _import_model_module("convnext")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = mod.build_convnext_tiny()
    net.load_state_dict(torch.load(str(path), map_location=device))
    net.eval().to(device)
    class_names = sorted(os.listdir(DATA_DIR / "train"))
    return net, class_names, _IMAGENET_TFM, device


def load_dinov2(path):
    mod = _import_model_module("dinov2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = mod.build_dinov2_small()
    net.load_state_dict(torch.load(str(path), map_location=device))
    net.eval().to(device)
    class_names = sorted(os.listdir(DATA_DIR / "train"))
    return net, class_names, _IMAGENET_TFM, device


# ── Inference functions ────────────────────────────────────────────────────────
def predict_yolo(model, img_path):
    results = model(str(img_path), verbose=False)
    probs = results[0].probs
    idx = int(probs.top1)
    return model.names[idx].replace("_", " "), float(probs.top1conf)


def predict_cnn(model, class_names, tfm, device, img_path):
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
        result = lp._query_ollama(prompt, image_path=image_path)
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


# ── Build audit prompts ────────────────────────────────────────────────────────
def build_prompt(species_name, confidence, context_row):
    from audit_layer import build_audit_prompt
    ctx = {k: context_row.get(k, "Unknown") for k in
           ["toxicity_type", "habitat", "season", "region", "key_warnings"]}
    return build_audit_prompt(species_name, confidence, ctx,
                              user_season="Unknown", user_location="Unknown")


def build_visual_prompt(species_name, confidence, context_row):
    from audit_layer import build_visual_audit_prompt
    ctx = {k: context_row.get(k, "Unknown") for k in
           ["toxicity_type", "habitat", "season", "region", "key_warnings",
            "edibility", "cap_description", "gill_description",
            "stem_description", "smell", "lookalikes"]}
    return build_visual_audit_prompt(species_name, confidence, ctx,
                                     user_season="Unknown", user_location="Unknown")


# ── Collect test images ────────────────────────────────────────────────────────
def collect_test_images(context, samples_dangerous, samples_safe):
    test_dir   = DATA_DIR / "test"
    image_list = []

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        species_key  = class_dir.name.replace(" ", "_").lower()
        ctx_row      = context.get(species_key, {})
        toxicity     = ctx_row.get("toxicity_type", "Unknown")
        dangerous    = is_dangerous(toxicity)

        n = samples_dangerous if dangerous else samples_safe
        if n == 0:
            continue

        imgs = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        random.shuffle(imgs)

        for img in imgs[:n]:
            image_list.append({
                "path":       img,
                "true_class": class_dir.name,
                "toxicity":   toxicity,
                "dangerous":  dangerous,
                "ctx_row":    ctx_row,
            })

    n_danger = sum(1 for i in image_list if i["dangerous"])
    n_safe   = sum(1 for i in image_list if not i["dangerous"])
    print(f"Collected {len(image_list)} test images "
          f"({n_danger} dangerous, {n_safe} safe)")
    return image_list


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20,
                        help="Max images per dangerous class")
    parser.add_argument("--samples-safe", type=int, default=5,
                        help="Max images per safe class (0 to skip safe species)")
    parser.add_argument("--llms", nargs="+",
                        default=["none", "gemini"],
                        choices=["none", "gemini"],
                        help="LLM providers to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 55)
    print("  Ablation Study — Vision x LLM Safety Evaluation")
    print("=" * 55)
    print(f"LLMs to evaluate  : {args.llms}")
    print(f"Samples dangerous : {args.samples}")
    print(f"Samples safe      : {args.samples_safe}\n")

    context     = load_context()
    test_images = collect_test_images(context, args.samples, args.samples_safe)

    # Load vision models
    print("\nLoading vision models...")
    yolo26   = load_yolo(WEIGHTS["yolo26n"])  if WEIGHTS["yolo26n"].exists()  else None
    convnext = load_convnext(WEIGHTS["convnext"]) if WEIGHTS["convnext"].exists() else None
    dinov2   = load_dinov2(WEIGHTS["dinov2"])     if WEIGHTS["dinov2"].exists()   else None

    vision_models = {}
    if yolo26:   vision_models["YOLOv26n"]        = ("yolo", yolo26)
    if convnext: vision_models["ConvNeXt-Tiny"]   = ("cnn",  convnext)
    if dinov2:   vision_models["DINOv2-Small"]    = ("cnn",  dinov2)

    out_csv = RESULTS_DIR / "ablation_raw.csv"

    # Resume: load already-completed images so we don't redo them
    done_images = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        # Backfill `dangerous` column for rows written before this column existed
        if "dangerous" not in existing.columns:
            existing["dangerous"] = existing["true_toxicity"].apply(
                lambda t: is_dangerous(str(t))
            )
        rows = existing.to_dict("records")
        n_combos = len(vision_models) * len(args.llms)
        counts = existing.groupby("image").size()
        done_images = set(counts[counts >= n_combos].index)
        print(f"Resuming — {len(done_images)} images already done, "
              f"{len(test_images) - len(done_images)} remaining.")
    else:
        rows = []

    gemini_dead = False  # set True on unrecoverable Gemini quota error → stop calling it

    for i, item in enumerate(tqdm(test_images, desc="Evaluating")):
        img_path    = item["path"]
        true_class  = item["true_class"]
        toxicity    = item["toxicity"]
        dangerous   = item["dangerous"]
        ctx_row     = item["ctx_row"]

        if img_path.name in done_images:
            continue

        for model_name, (model_type, model_obj) in vision_models.items():
            # Run vision model
            try:
                import time
                t0 = time.perf_counter()
                if model_type == "yolo":
                    pred_class, confidence = predict_yolo(model_obj, img_path)
                else:
                    net, class_names, tfm, device = model_obj
                    pred_class, confidence = predict_cnn(net, class_names, tfm, device, img_path)
                inference_ms = round((time.perf_counter() - t0) * 1000, 1)
            except Exception as e:
                continue

            pred_ctx  = context.get(pred_class.replace(" ", "_").lower(), {})

            # Skip LLM if confidence < 0.70 — risk engine always flags HIGH
            # via the confidence threshold regardless of LLM verdict.
            # LLM only matters when vision is confident but potentially wrong.
            needs_llm     = confidence >= 0.70
            text_prompt   = build_prompt(pred_class, confidence, pred_ctx)        if needs_llm else None
            visual_prompt = build_visual_prompt(pred_class, confidence, pred_ctx) if needs_llm else None

            for llm_name in args.llms:
                try:
                    if not needs_llm or llm_name == "none":
                        llm_verdict = "✅ PLAUSIBLE"
                    elif llm_name == "gemini" and gemini_dead:
                        llm_verdict = "SKIPPED: Gemini quota exhausted"
                        risk_level  = "QUOTA_EXHAUSTED"
                        flagged     = False
                        rows.append({
                            "image":         img_path.name,
                            "true_class":    true_class,
                            "true_toxicity": toxicity,
                            "dangerous":     dangerous,
                            "model":         model_name,
                            "llm":           llm_name,
                            "pred_class":    pred_class,
                            "confidence":    round(confidence, 4),
                            "inference_ms":  inference_ms,
                            "risk_level":    risk_level,
                            "flagged":       flagged,
                        })
                        continue
                    else:
                        img    = str(img_path) if llm_name == "gemini" else None
                        prompt = visual_prompt  if llm_name == "gemini" else text_prompt
                        llm_verdict = query_llm_for_ablation(llm_name, "", prompt, image_path=img)
                        if llm_name == "gemini" and "rate limit after 3 retries" in llm_verdict:
                            print("\n[Gemini quota exhausted — stopping Gemini calls for this run. "
                                  "Re-run to resume from this point.]")
                            gemini_dead = True
                    risk_result = get_risk(pred_class, confidence, pred_ctx, llm_verdict)
                    risk_level  = risk_result["risk_level"]
                except Exception as e:
                    llm_verdict = f"ERROR: {e}"
                    risk_level  = "ERROR"

                flagged = risk_level in ("HIGH", "CRITICAL")

                rows.append({
                    "image":         img_path.name,
                    "true_class":    true_class,
                    "true_toxicity": toxicity,
                    "dangerous":     dangerous,
                    "model":         model_name,
                    "llm":           llm_name,
                    "pred_class":    pred_class,
                    "confidence":    round(confidence, 4),
                    "inference_ms":  inference_ms,
                    "risk_level":    risk_level,
                    "flagged":       flagged,
                })

        # Save checkpoint every image
        if rows:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Save raw results
    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_csv, index=False)
    print(f"\nRaw results saved to {out_csv}")

    danger_df = results_df[results_df["dangerous"] == True]
    safe_df   = results_df[results_df["dangerous"] == False]

    # ── Metric 1: Overall safety recall (dangerous species only) ─────────────
    print("\n" + "=" * 65)
    print("  Metric 1 — Safety Recall on Dangerous Species")
    print("=" * 65)
    print(f"{'Model':<20} {'LLM':<12} {'Recall':>8} {'False Safe':>11} {'N':>6}")
    print("-" * 65)

    summary_rows = []

    for (model, llm), grp in danger_df.groupby(["model", "llm"]):
        recall     = grp["flagged"].mean() * 100
        false_safe = 100 - recall
        print(f"{model:<20} {llm:<12} {recall:>7.1f}% {false_safe:>10.1f}% {len(grp):>6}")
        summary_rows.append({
            "model": model, "llm": llm,
            "metric": "safety_recall",
            "value_%": round(recall, 2),
            "n_images": len(grp),
        })

    # ── Metric 2: High-confidence misclassification recall ────────────────────
    hc_miss = danger_df[
        (danger_df["confidence"] >= 0.70) &
        (~danger_df["pred_class"].str.lower().str.replace(" ", "_")
         .map(lambda x: is_dangerous(context.get(x, {}).get("toxicity_type", ""))))
    ]

    n_hc = len(hc_miss) // max(1, len(danger_df["model"].unique()) * len(danger_df["llm"].unique()))
    print(f"\n{'='*65}")
    print("  Metric 2 — High-Confidence Misclassification Recall")
    print(f"  (vision model >=70% confident, predicted safe, true class dangerous)")
    print(f"  Total cases per combo: {n_hc}")
    print(f"{'='*65}")
    print(f"{'Model':<20} {'LLM':<12} {'Recall':>8} {'N Cases':>8}")
    print("-" * 65)

    for (model, llm), grp in hc_miss.groupby(["model", "llm"]):
        recall = grp["flagged"].mean() * 100 if len(grp) > 0 else 0.0
        print(f"{model:<20} {llm:<12} {recall:>7.1f}% {len(grp):>8}")
        summary_rows.append({
            "model": model, "llm": llm,
            "metric": "hc_misclassification_recall",
            "value_%": round(recall, 2),
            "n_images": len(grp),
        })

    # ── Metric 4: Precision / F1 across all species ───────────────────────────
    if len(safe_df) > 0:
        print(f"\n{'='*65}")
        print("  Metric 4 — Precision / Recall / F1 (all species)")
        print(f"  ({len(danger_df['image'].nunique() if 'image' in danger_df else 0)} dangerous images, "
              f"{safe_df['image'].nunique()} safe images)")
        print(f"{'='*65}")
        print(f"{'Model':<20} {'LLM':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 65)

        for (model, llm), d_grp in danger_df.groupby(["model", "llm"]):
            s_grp = safe_df[(safe_df["model"] == model) & (safe_df["llm"] == llm)]
            tp = d_grp["flagged"].sum()
            fn = (~d_grp["flagged"]).sum()
            fp = s_grp["flagged"].sum()  if len(s_grp) else 0
            tn = (~s_grp["flagged"]).sum() if len(s_grp) else 0
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            print(f"{model:<20} {llm:<12} {precision:>9.1f}% {recall:>7.1f}% {f1:>7.1f}%")
            summary_rows.append({
                "model": model, "llm": llm,
                "metric": "precision",
                "value_%": round(precision, 2),
                "n_images": int(tp + fp),
            })
            summary_rows.append({
                "model": model, "llm": llm,
                "metric": "f1",
                "value_%": round(f1, 2),
                "n_images": int(tp + fn + fp + tn),
            })
    else:
        print("\n[Metric 4 skipped — no safe species in dataset. Re-run without --samples-safe 0]")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary saved to {summary_csv}")


if __name__ == "__main__":
    main()
