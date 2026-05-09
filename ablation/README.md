# Ablation Study — Vision Model × LLM Evaluation

## What This Study Does

We tested every combination of vision model and AI auditor — including the AI auditor working alone with no vision model — to find out how much each part contributes. The goal is to show that neither part works well alone, and that our chosen combination (YOLO + Gemini) is the right one for a phone-based app.

---

## Key Findings

### Finding 1 — Neither component works well alone

Vision models on their own miss 70–86% of dangerous species. Gemini alone catches 92.6% of dangerous species but also wrongly flags 70.7% of safe species as dangerous — it is too aggressive to be useful on its own. The combination is what works.

| Condition | Catches dangerous? | Wrongly flags safe? | F1 |
|:----------|-------------------:|--------------------:|---:|
| LLM Only (Gemini) | **92.6%** | **70.7%** | 81.6% |
| YOLO + Gemini | 75.9% | 18.6% | **80.0%** |
| ConvNeXt + Gemini | 73.4% | 10.3% | 81.2% |
| DINOv2 + Gemini | 74.9% | 10.3% | 81.9% |
| YOLO alone | 30.0% | 16.5% | 44.2% |
| ConvNeXt alone | 14.3% | 0.0% | 25.0% |
| DINOv2 alone | 15.3% | 0.0% | 26.5% |

### Finding 2 — Once Gemini is added, all vision models perform about the same

YOLO + Gemini, ConvNeXt + Gemini, and DINOv2 + Gemini all score within 2 F1 points of each other (80.0–81.9%). Adding Gemini closes the gap between a small cheap model and a large expensive one. This means choosing the vision model is not a model-quality decision — it is a practical one about what runs on a phone.

### Finding 3 — YOLO is safer than ConvNeXt and DINOv2 when Gemini is unavailable

If the internet is down or the API fails, the app falls back to the vision model alone. In that scenario, YOLO catches 30% of dangerous species — more than double the 14–15% from ConvNeXt and DINOv2. This is not because YOLO is more accurate. It is because ConvNeXt and DINOv2 are overconfident: they almost always give a high-confidence prediction, so our automatic low-confidence warning never triggers. YOLO is less certain more often, and uncertainty always triggers a warning.

**A highly accurate but overconfident model is the most dangerous fallback — it gives the user a confident wrong answer with no warning.**

### Finding 4 — Gemini alone is not a practical solution

Gemini alone flags 70.7% of safe mushrooms as dangerous. In practice this means the app would warn about almost every mushroom a user photographs, including common edible ones. Users would stop trusting the warnings and ignore them. High recall with this level of false alarms is not useful.

### Finding 5 — YOLO + Gemini is the right choice for production

All three Vision + Gemini combinations achieve similar scores (F1 80–82%). Within that, YOLO is the clear winner for a phone app:

| Model | F1 | Speed | Size | Needs GPU? |
|:------|---:|------:|-----:|:-----------|
| YOLO + Gemini | 80.0% | **15 ms** | **1.74M params** | No |
| ConvNeXt + Gemini | 81.2% | 30 ms | 28M params | Recommended |
| DINOv2 + Gemini | 81.9% | 39 ms | 22M params | Recommended |

The Gemini API call takes 500–1500ms regardless of which vision model is used, so the difference between 15ms and 39ms does not change the user experience. What matters is that YOLO runs on a phone without a GPU. ConvNeXt and DINOv2 need one.

---

## What We Tested

| Condition | AI Auditor | What it does |
|:----------|:-----------|:-------------|
| YOLO alone | None | Flags anything below 70% confidence |
| ConvNeXt alone | None | Flags anything below 70% confidence |
| DINOv2 alone | None | Flags anything below 70% confidence |
| Gemini alone | Gemini Flash Lite | Looks at the photo and decides danger level with no vision model help |
| YOLO + Gemini | Gemini Flash Lite | YOLO identifies the species, Gemini checks the photo and confirms |
| ConvNeXt + Gemini | Gemini Flash Lite | Same as above with ConvNeXt |
| DINOv2 + Gemini | Gemini Flash Lite | Same as above with DINOv2 |

### Why these three vision models?

- **YOLOv26n** — built for phones and low-power devices. Fastest, smallest, our production model.
- **ConvNeXt-Tiny (PEFT)** — a modern image classifier with good accuracy. Needs a GPU for fast inference.
- **DINOv2-Small** — a vision model trained on 142M images with no labels. Highest classification accuracy but slowest and GPU-dependent.

### Why Llama3.2 was excluded

Llama gave unrealistically high scores. The reason: our audit prompt tells the model to warn about toxic species and the prompt also includes the toxicity information from our database. Llama was simply reading the toxicity label and repeating it back as a warning, not actually looking at the image or reasoning about it.

### Why Gemma4:e2b was excluded

Two problems: first, a bug meant the model never actually received the images. After fixing that, the 2-billion parameter model still performed no better than having no AI auditor at all — recognising specific mushroom species requires more knowledge than a model this small carries.

### Only large frontier models work for this task

Small local models (Llama, Gemma) are not good enough for visual mushroom identification. A large capable model like Gemini, GPT-4V, or Claude is needed. This means the AI auditor always requires an internet connection, regardless of which vision model is used.

---

## How We Measure Performance

### Recall
Out of all dangerous mushroom photos, what percentage did the system correctly flag as dangerous? Higher is better.

### False Positive Rate
Out of all safe mushroom photos, what percentage did the system wrongly flag as dangerous? Lower is better. This matters because too many false alarms and users stop trusting the app.

### F1 Score
A single number that balances both of the above. High F1 means the system catches most dangerous species without constantly alarming users about safe ones. This is our main metric.

### Confident Wrong Answers (Metric 4)
The most dangerous failure mode: the vision model is confident it identified a safe species, but it is actually looking at something dangerous. What percentage of these did Gemini catch by looking at the photo? The automatic warning system never fires in these cases — only Gemini can catch them.

### Inference Speed
How long the vision model takes per photo in milliseconds. Does not include the Gemini API call, which takes the same amount of time regardless of vision model.

---

## Test Set

- **Source:** held-out test split — none of these images were used in any model training
- **Dangerous species:** 40 species (Deadly, Highly Toxic, Toxic, Hallucinogenic, or Pathogenic)
- **Safe species:** remaining species, 3 images per class
- **Total:** 300 images (203 dangerous, 97 safe), random seed 42

## Why 70% Confidence?

Our system automatically warns the user whenever the vision model is less than 70% confident, regardless of what species it predicted — low confidence means the model is guessing. The side effect is that YOLO, which is less confident on average, gets more automatic warnings than ConvNeXt or DINOv2 and therefore catches more dangerous species on its own.

---

## Results

*300 images total (203 dangerous, 97 safe), seed 42.*

### Full Results

| Condition | Catches dangerous | Wrongly flags safe | F1 | Missed dangerous |
|:----------|------------------:|-------------------:|---:|-----------------:|
| LLM Only (Gemini) | 92.6% | 70.7% | 81.6% | 7.4% |
| DINOv2 + Gemini | 74.9% | 10.3% | 81.9% | 25.1% |
| ConvNeXt + Gemini | 73.4% | 10.3% | 81.2% | 26.6% |
| YOLO + Gemini | 75.9% | 18.6% | 80.0% | 24.1% |
| YOLO alone | 30.0% | 16.5% | 44.2% | 70.0% |
| DINOv2 alone | 15.3% | 0.0% | 26.5% | 84.7% |
| ConvNeXt alone | 14.3% | 0.0% | 25.0% | 85.7% |

### Confident Wrong Answer Recovery (Gemini only)

Cases where the vision model was confident but wrong — the most dangerous failure type.

| Vision Model | Gemini caught | Cases |
|:-------------|-------------:|------:|
| YOLOv26n | 83.3% | 6 |
| ConvNeXt-Tiny | 100.0% | 1 |
| DINOv2-Small | — | 0 |

> Sample size is small because the models are mostly accurate — confident wrong answers on dangerous species are rare.

### Speed (vision model only)

| Model | Median ms |
|:------|----------:|
| YOLOv26n | 15.3 |
| ConvNeXt-Tiny | 30.1 |
| DINOv2-Small | 38.7 |

---

## How to Run

```bash
# Standard run
python ablation/run_ablation.py --max-images 300 --samples-safe 3

# Quick test (1 image per class)
python ablation/run_ablation.py --samples 1 --samples-safe 1

# Resume after interruption — skips already completed images automatically
python ablation/run_ablation.py --max-images 300 --samples-safe 3

# Generate figures
python ablation/plot_results.py
```

Requires `GEMINI_API_KEY` in `.env`. Results saved to `ablation/results/` (gitignored).

---

## Citation

- YOLOv26 (YOLO11): Ultralytics (2024). https://github.com/ultralytics/ultralytics
- ConvNeXt: Liu et al. (2022). A ConvNet for the 2020s. CVPR.
- DINOv2: Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. TMLR.
- Gemini: Google DeepMind (2024). https://ai.google.dev/gemini
