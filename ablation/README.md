# Ablation Study — Vision Model × LLM Safety Evaluation

## Objective

This ablation study evaluates every combination of vision model and LLM to quantify how much each component contributes to the system's safety. Since the YOLO + LLM pipeline is our main contribution, we need empirical evidence that the LLM layer adds measurable safety value beyond the vision model alone.

## Evaluation Matrix

| Vision Model | LLM | Description |
|:-------------|:----|:------------|
| YOLOv8n | None | Vision only baseline |
| YOLOv8n | Llama3.2 | Vision + local LLM |
| YOLOv8n | Gemma4:e2b | Vision + local LLM |
| YOLOv26n | None | Vision only baseline |
| YOLOv26n | Llama3.2 | Vision + local LLM |
| YOLOv26n | Gemma4:e2b | Vision + local LLM |
| EfficientNet-B0 | None | Vision only baseline |
| EfficientNet-B0 | Llama3.2 | Vision + local LLM |
| EfficientNet-B0 | Gemma4:e2b | Vision + local LLM |

## Metrics

### Metric 1 — Overall Safety Recall
Out of all dangerous species test images, what percentage did the pipeline correctly flag as HIGH or CRITICAL risk?

This measures the system's general safety coverage across all conditions.

- **Safety Recall** — higher is better
- **False Safe Rate** = 1 − Safety Recall — the dangerous failure rate, lower is better

### Metric 2 — High-Confidence Misclassification Recall
Out of all cases where the vision model was ≥70% confident but predicted a safe species on a dangerous true class, what percentage did the LLM escalate to HIGH or CRITICAL?

This is the metric that specifically isolates the LLM's contribution. When vision confidence is below 70%, the risk engine flags it as HIGH automatically via the confidence threshold — the LLM adds nothing in those cases. The LLM only has the opportunity to add value when the vision model is **confident but wrong**, predicting an edible species on what is actually a toxic mushroom. Metric 2 filters for exactly those cases and shows whether the LLM catches them.

## Why 800 Images?

- 40 dangerous species out of 169 total (anything with `Deadly`, `Highly Toxic`, `Toxic`, `Hallucinogenic`, or `Pathogenic` in the toxicity label)
- Up to 20 images sampled per class from the held-out test split
- 40 × 20 = 800 images maximum
- Edible and inedible classes are excluded — they cannot contribute to safety recall by definition, and including them would only slow down the evaluation with no scientific benefit

## Why the 70% Confidence Threshold?

The risk engine in `services/brain_ui/pipeline/risk_engine.py` has a hardcoded rule: if YOLO confidence is below 70%, the verdict is always escalated to HIGH regardless of what the LLM says. This means for low-confidence predictions, the LLM call is deterministic — it cannot change the outcome. Calling the LLM on those images would waste time and produce identical results to the "none" baseline. We therefore skip the LLM call when `confidence < 0.70` and use a neutral PLAUSIBLE verdict, which accurately reflects what would happen in the real system.

## Test Set

- Source: `data/dataset_split/test/` (held-out 10%, never seen during any model training)
- Scope: dangerous classes only
- Dangerous classes: 40 out of 169 total species
- Sampling: up to 20 images per class
- Seed: 42 (reproducible)

## How to Run

```bash
# Full evaluation — requires Ollama running locally with llama3.2 and gemma4:e2b
ollama serve   # in a separate terminal

python ablation/run_ablation.py

# Quick test with fewer images
python ablation/run_ablation.py --samples 5
```

## Results

### Metric 1 — Overall Safety Recall

*Updated after evaluation completes.*

| Model | LLM | Safety Recall | False Safe Rate |
|:------|:----|:-------------|:----------------|
| YOLOv8n | None | TBD | TBD |
| YOLOv8n | Llama3.2 | TBD | TBD |
| YOLOv8n | Gemma4:e2b | TBD | TBD |
| YOLOv26n | None | TBD | TBD |
| YOLOv26n | Llama3.2 | TBD | TBD |
| YOLOv26n | Gemma4:e2b | TBD | TBD |
| EfficientNet-B0 | None | TBD | TBD |
| EfficientNet-B0 | Llama3.2 | TBD | TBD |
| EfficientNet-B0 | Gemma4:e2b | TBD | TBD |

### Metric 2 — High-Confidence Misclassification Recall

| Model | LLM | Safety Recall | N Cases |
|:------|:----|:-------------|:--------|
| YOLOv8n | None | TBD | TBD |
| YOLOv8n | Llama3.2 | TBD | TBD |
| YOLOv8n | Gemma4:e2b | TBD | TBD |
| YOLOv26n | None | TBD | TBD |
| YOLOv26n | Llama3.2 | TBD | TBD |
| YOLOv26n | Gemma4:e2b | TBD | TBD |
| EfficientNet-B0 | None | TBD | TBD |
| EfficientNet-B0 | Llama3.2 | TBD | TBD |
| EfficientNet-B0 | Gemma4:e2b | TBD | TBD |

## Citation

If citing this ablation in the paper, reference:

- YOLOv8: Jocher et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
- YOLOv26 (YOLO11): Ultralytics (2024). https://github.com/ultralytics/ultralytics
- EfficientNet: Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
- Llama3.2: Meta AI (2024). https://llama.meta.com
- Gemma4: Google DeepMind (2024). https://ai.google.dev/gemma
