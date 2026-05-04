# Ablation Study — Vision Model × LLM Safety Evaluation

## Objective

Evaluate every combination of vision model and LLM to quantify how much the audit layer contributes to safety. We need empirical evidence that the LLM layer adds measurable value beyond the vision model alone, and that pairing a fast lightweight vision model with a capable LLM auditor is a viable strategy for real-time deployment.

---

## Key Findings

### Finding 1 — The LLM is the safety backbone, not an optional add-on

All three models converge to ~90–94% safety recall with Gemini, regardless of their baseline accuracy or speed. The vision model determines what species is predicted; the LLM determines whether that prediction is trusted. Removing the LLM from any configuration collapses safety recall by 60–70 percentage points.

### Finding 2 — The LLM audit layer benefits stronger models most

The naive reading is that YOLO benefits most from the LLM. The absolute lift tells the opposite story:

| Model | Without LLM audit | With Gemini | LLM Lift |
|:------|------------------:|------------:|---------:|
| YOLOv26n | 36% | 94% | **+58pp** |
| ConvNeXt-Tiny | 21% | 90% | **+69pp** |
| DINOv2-Small | 24% | 90% | **+66pp** |

Gemini does more work for ConvNeXt and DINOv2. These models are systematically overconfident — they almost always predict above the 70% confidence threshold, so the threshold rule barely fires. They rely on Gemini for nearly every dangerous species. YOLO's lower confidence means the threshold rule catches more cases automatically, so Gemini is called less often.

### Finding 3 — Overconfident models are the most dangerous without the LLM

**ConvNeXt-Tiny and DINOv2-Small are more dangerous than YOLOv26n in a pipeline without an LLM**, despite being more accurate classifiers. Their overconfidence bypasses the confidence threshold safety net almost entirely — they miss ~78–79% of dangerous species without the LLM. YOLO misses ~64% because its lower confidence at least trips the threshold rule as a partial backstop.

This is not an argument for using a weaker vision model. It is an argument that a highly accurate but overconfident vision model without an LLM audit is the most dangerous configuration — it gives users confident wrong answers with no safety net.

### Finding 4 — YOLO + Gemini is the right production configuration

YOLO reaches near-DINOv2 safety (94% vs 90%) at 15ms vs 39ms inference. The LLM API call (~500–1500ms) dominates total latency regardless of vision model, so the vision model choice is primarily about **on-device footprint** — YOLO is deployable on mobile/edge hardware without a GPU.

---

## Evaluation Matrix

| Vision Model | LLM | Description |
|:-------------|:----|:------------|
| YOLOv26n | None | Confidence threshold rule only |
| YOLOv26n | Gemini 3.1 Flash Lite | + Visual audit layer |
| ConvNeXt-Tiny (PEFT) | None | Confidence threshold rule only |
| ConvNeXt-Tiny (PEFT) | Gemini 3.1 Flash Lite | + Visual audit layer |
| DINOv2-Small | None | Confidence threshold rule only |
| DINOv2-Small | Gemini 3.1 Flash Lite | + Visual audit layer |

### Why these three vision models?

- **YOLOv26n** — optimised for edge/mobile hardware. Fastest inference, lowest parameter count, highest misclassification rate on fine-grained species.
- **ConvNeXt-Tiny (PEFT)** — modernised CNN with transformer-style design. Strong accuracy with moderate inference cost.
- **DINOv2-Small** — self-supervised ViT pretrained on 142M images. Richest features; highest accuracy but slowest inference.

### Why Llama3.2 was excluded

Produced artificially inflated safety recall (~90%). Root cause: the text audit prompt instructs the LLM to *"issue a clear safety warning regardless of plausibility"* when a species is toxic. Since the prompt contains the toxicity field from the context CSV, Llama re-reads the label and echoes it back as a warning — label lookup, not reasoning. Additionally, text-only ecological reasoning requires real location/season data; batch evaluation uses Unknown/Unknown, disabling the only signal a text-only LLM could use.

### Why Gemma4:e2b was excluded

Two issues: (1) an `image_path` forwarding bug meant Gemma never saw the images and defaulted to AGREE on every case; (2) after fixing the bug, the 2B-parameter model still produced near-identical results to the no-LLM baseline — fine-grained species discrimination requires more world knowledge than a 2B model carries.

### The LLM bottleneck finding

Local lightweight models (Llama3.2, Gemma4:e2b) are insufficient for reliable visual mushroom verification. A frontier-class multimodal model (Gemini, GPT-4V, Claude Opus) is required for the audit layer to add genuine safety value. This has deployment implications: the LLM call requires a cloud API regardless of how fast the local vision model is.

---

## Metrics

### Metric 1 — Overall Safety Recall
Out of all dangerous species test images, what percentage did the pipeline correctly flag as HIGH or CRITICAL risk?

> **Interpretation:** The "No LLM" baseline measures what the confidence threshold rule alone achieves — predictions below 70% confidence are auto-flagged HIGH regardless of which species was predicted. This means overconfident models (ConvNeXt, DINOv2) appear to have *lower* vision-only recall than YOLO, not because they are less accurate but because they rarely fall below the threshold. The vision-only bars represent pipeline safety without the LLM, not vision model accuracy.

### Metric 2 — High-Confidence Misclassification Recall
Out of cases where the vision model was ≥70% confident but predicted a safe species on a dangerous true class, what percentage did Gemini escalate to HIGH or CRITICAL?

This isolates Gemini's genuine visual contribution. The threshold rule never fires here (confidence is high). Vision-only recall for this metric is 0% by design. Gemini either visually disagrees with the prediction or it doesn't. These are the most dangerous failures in deployment — a confident wrong answer the user has no reason to question.

### Metric 3 — Inference Speed
Median inference time (ms) per image, vision model only (excludes LLM latency). Measured with `time.perf_counter()`. Warmup outliers (>3× median) excluded.

---

## Test Set

- **Source:** `data/dataset_split/test/` — held-out 10%, never seen during any model training
- **Scope:** 40 dangerous species (toxicity contains `Deadly`, `Highly Toxic`, `Toxic`, `Hallucinogenic`, or `Pathogenic`)
- **Sampling:** up to 20 images per dangerous class, seed 42 (reproducible)
- **Total available:** 169 classes, 10,549 images (~62 per class)

## Why the 70% Confidence Threshold?

The risk engine auto-escalates any prediction below 70% confidence to HIGH risk — when the model is uncertain, always warn the user. This is correct safety policy. The LLM is skipped for these cases because the outcome is deterministic regardless of what Gemini would say. The side effect is that YOLO's lower average confidence gives it higher vision-only recall than the more accurate but overconfident CNN/ViT models — see Findings 2 and 3 above.

---

## Results

*Based on n=245 images across 40 dangerous classes (partial run, seed 42).*

### Metric 1 — Dangerous Species Caught by the Pipeline

| Model | No LLM audit | With Gemini | LLM Lift |
|:------|-------------:|------------:|---------:|
| YOLOv26n | 36.1% | 93.6% | +57.5pp |
| ConvNeXt-Tiny | 20.8% | 90.1% | +69.3pp |
| DINOv2-Small | 23.8% | 90.1% | +66.3pp |

### Metric 2 — High-Confidence Misclassification Recall (Gemini only)

*Vision-only is 0% by design for all models.*

| Model | Gemini Recall | N Cases |
|:------|:-------------|--------:|
| YOLOv26n | 85.7% | 7 |
| ConvNeXt-Tiny | 100.0% | 1 |
| DINOv2-Small | — | 0 |

> N is small — these cases are rare because the models are mostly accurate. DINOv2 made zero high-confidence dangerous misidentifications across 245 images. A full 800-image run is needed for reliable Metric 2 estimates.

### Metric 3 — Inference Speed (vision model only)

| Model | Median ms | Mean ms |
|:------|----------:|--------:|
| YOLOv26n | 15.3 | 15.5 |
| ConvNeXt-Tiny | 30.1 | 31.9 |
| DINOv2-Small | 38.7 | 39.3 |

---

## How to Run

```bash
# Full run — dangerous species only (default)
python ablation/run_ablation.py

# Include safe species for precision/F1 metrics
python ablation/run_ablation.py --samples-safe 5

# Quick smoke test (1 image per class)
python ablation/run_ablation.py --samples 1 --samples-safe 1

# Resume after interruption — automatically skips completed images
python ablation/run_ablation.py
```

Requires `GEMINI_API_KEY` set in `.env`. Results are saved to `ablation/results/` (gitignored).

---

## Citation

- YOLOv26 (YOLO11): Ultralytics (2024). https://github.com/ultralytics/ultralytics
- ConvNeXt: Liu et al. (2022). A ConvNet for the 2020s. CVPR.
- DINOv2: Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. TMLR.
- Gemini: Google DeepMind (2024). https://ai.google.dev/gemini
