# Mushroom Safety Classification System — Project Overview

A full account of what was built, what was trained, what was found, and how each piece works.

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Dataset](#2-dataset)
3. [System Architecture](#3-system-architecture)
4. [Models](#4-models)
   - [YOLOv8n-cls — Baseline](#41-yolov8n-cls--baseline)
   - [YOLOv26n-cls — Production Model](#42-yolov26n-cls--production-model)
   - [EfficientNet-B0 — Classification-Native CNN](#43-efficientnet-b0--classification-native-cnn)
   - [TaxonomicYOLO26 — The Franken Model](#44-taxonomicyolo26--the-franken-model)
   - [Model Comparison](#45-model-comparison)
5. [Safety Pipeline](#5-safety-pipeline)
6. [MLOps & Deployment](#6-mlops--deployment)
7. [Project Structure](#7-project-structure)

---

## 1. Project Goal

The system is a **safety tool for foragers**. The core problem: a vision model alone is not trustworthy enough for life-or-death mushroom identification. A single high-confidence wrong prediction could be fatal.

The solution is a layered pipeline where no single component makes the final call:

- A **vision model** identifies the species and emits a confidence score
- An **ecological database** cross-references the prediction against known habitat, season, and region rules
- An **LLM audit layer** verifies whether the identification makes environmental sense
- A **deterministic risk engine** produces the final safety verdict using hard-coded Python rules the LLM cannot override

> The philosophy: *the LLM provides explanations, Python provides guarantees.*

---

## 2. Dataset

**Source:** `zlatan599/mushroom1` on Kaggle

| Property | Detail |
|---|---|
| Total species | 169 |
| Split | 80% train / 10% val / 10% test |
| Training CSV | ~689,000 image-label mappings |
| Folder structure | One folder per species (YOLO-compatible) |

Species range from choice edibles (*Boletus edulis*, *Cantharellus cibarius*) to the deadly Death Cap (*Amanita phalloides* — responsible for the majority of fatal mushroom poisonings globally) to psychoactive species (*Psilocybe* spp.) and lichens (*Cladonia*, *Lobaria*, *Parmelia*). The inclusion of lichens adds significant visual complexity since they look nothing like typical mushrooms but share taxonomic neighbours in the dataset.

The raw Kaggle download was restructured into `data/dataset_split/` using `scripts/setup/prepare_dataset.py`. A YOLO-compatible `data/dataset.yaml` was generated pointing to all 169 class names.

---

## 3. System Architecture

```
User uploads image + season + location
              │
              ▼
    ┌─────────────────────┐
    │   Gradio Brain UI    │  port 7860
    └──────────┬──────────┘
               │  HTTP POST /predict
               ▼
    ┌─────────────────────┐
    │  FastAPI Vision API  │  port 8000
    │  (YOLOv26 model)     │
    └──────────┬──────────┘
               │  (species_name, confidence)
               ▼
    ┌─────────────────────┐
    │  Ecological CSV DB   │
    │  toxicity / habitat  │
    │  season / region     │
    │  key_warnings        │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │   LLM Audit Layer    │  Gemini
    │  "Is this plausible  │
    │   given the env?"    │
    └──────────┬──────────┘
               │  verdict text
               ▼
    ┌─────────────────────┐
    │   Risk Engine        │
    │  deterministic       │
    │  Python if/else      │
    └──────────┬──────────┘
               │
               ▼
        Safety Report
   CRITICAL / HIGH / MODERATE / LOW
```

Each step is a separate Python module. They are deliberately decoupled so any one component can be swapped, upgraded, or tested independently.

---

## 4. Models

Four different architectures were trained on the same 169-class dataset. Each explores a different angle on the fine-grained species classification problem.

---

### 4.1 YOLOv8n-cls — Baseline

The smallest classifier from the YOLOv8 generation. Chosen as the starting point because it is well-documented, fast, and trivial to deploy via the Ultralytics library.

| Property | Value |
|---|---|
| Parameters | ~2.7 million |
| Model size | ~6.2 MB |
| Optimizer | AdamW (auto) |
| Loss | Cross-Entropy |
| Epochs trained | 50 |
| Top-1 accuracy | TBD |

**Architectural note:** YOLOv8's backbone was originally designed for object *detection* — finding and drawing bounding boxes around objects at multiple scales. When repurposed for classification it carries that detection-oriented heritage (C2f blocks, multi-scale feature pyramids) even though classification only needs a single global label per image. It works, but it is an architectural mismatch.

Training artifacts: `docs/yolo_runs/mushroom_classifier_v1/`

---

### 4.2 YOLOv26n-cls — Production Model

The primary model currently in production. YOLOv26 is a newer, more compact architecture than v8 that achieves higher accuracy in fewer parameters.

| Property | Value |
|---|---|
| Parameters | 1.74 million |
| Model size | 3.6 MB |
| Optimizer | AdamW (auto) |
| Loss | Cross-Entropy |
| Training hardware | NVIDIA RTX 3070 Ti (8 GB) |
| Training time | ~159 minutes (~2.65 hours) |
| Best epoch | 49 out of 50 |
| **Top-1 accuracy** | **88.1%** |
| **Top-5 accuracy** | **98.4%** |
| Final val loss | 0.41535 |
| Inference speed | 0.2 ms / image |

**What we found during training:**

Training loss and validation loss decreased together smoothly across all 50 epochs with no plateaus or spikes — a very clean convergence curve. At epoch 49, validation loss hit its minimum (0.41535). At epoch 50 it ticked up by +0.0003 (to 0.41565). This is a textbook early-overfitting signal: the model crossed the line from generalizing to memorizing right at the final epoch. Stopping at epoch 49's weights was effectively perfect timing.

**Why smaller but more accurate than v8n?**

YOLOv26 benefits from architectural improvements accumulated across several YOLO generations — better gradient flow, more efficient block designs — that allow it to extract more information from fewer parameters. Getting a smaller model with higher accuracy is a meaningful practical win: the production Docker image weighs less and inference is faster.

Weights: `docs/yolo_runs/yolo26_classifier_v1/weights/best.pt`
Training artifacts: `docs/yolo_runs/yolo26_classifier_v1/`

---

### 4.3 EfficientNet-B0 — Classification-Native CNN

A custom-headed EfficientNet-B0 fine-tuned from ImageNet weights. Built as a direct architectural alternative to the YOLO classifiers.

| Property | Value |
|---|---|
| Base model | EfficientNet-B0 (ImageNet pretrained) |
| Parameters | ~5.3 million |
| Model size | ~5 MB |
| Input size | 224 × 224 |
| Classifier head | `Dropout(0.3) → Linear(1280 → 169)` |
| Optimizer options | AdamW / SGD / RAdam |
| Loss options | CrossEntropy / CE+LabelSmoothing / Focal / Focal+LabelSmoothing |
| Expected Top-1 improvement over YOLOv26 | ~2–4% |

**Why EfficientNet instead of YOLO for classification?**

This is the key architectural question. YOLO was built for detection. Its backbone includes components designed for spatial reasoning at multiple scales (finding *where* things are). When you repurpose it for classification (just deciding *what* the image shows), those spatial components still exist and consume parameters, but they are doing work that isn't needed for the task.

EfficientNet was designed from the ground up specifically for classification using a **compound scaling law** — it scales model depth, width, and input resolution together in a mathematically derived ratio rather than independently. Every parameter is tuned for the classification objective. This is why a 5 MB EfficientNet is expected to outperform a 6.2 MB YOLOv8n on a pure classification task.

**Architecture changes from stock EfficientNet-B0:**

Stock EfficientNet-B0 has a classifier head of `[Dropout(0.2), Linear(1280 → 1000)]` for ImageNet's 1000 classes. We replace it with `[Dropout(0.3), Linear(1280 → 169)]`:
- The output dimension drops from 1000 to 169 mushroom species
- Dropout increases from 0.2 to 0.3 to combat overfitting on 169 fine-grained classes
- All backbone weights are kept from ImageNet pretraining — only the classifier head is random-initialized

The training script supports pluggable loss functions and optimizers for easy ablation experiments. Results saved to `docs/cnn_runs/`.

---

### 4.4 TaxonomicYOLO26 — The Franken Model

The most architecturally novel model in the project. Rather than predicting species alone, it simultaneously predicts **genus** and **species** using two separate output heads on a shared YOLOv26n-cls backbone.

```
YOLOv26n-cls backbone
(all layers except the final Classify head — stripped and reused)
        │
        │  global-average-pooled feature vector
        │
   ┌────┴────────────────────┐
   │                         │
genus_head               species_head
Dropout(0.4)             Dropout(0.4)
Linear(feat_dim → G)     Linear(feat_dim → S)
   │                         │
genus logits             species logits
```

| Property | Value |
|---|---|
| Backbone | YOLOv26n-cls pretrained (head removed) |
| Genus head | `Dropout(0.4) → Linear(feat_dim → num_genera)` |
| Species head | `Dropout(0.4) → Linear(feat_dim → num_species)` |
| Backbone LR | 1e-5 (gentle fine-tuning) |
| Head LR | 1e-3 (random-init, learns fast) |
| Scheduler | CosineAnnealingLR |
| Warmup period | Backbone frozen for first 5 epochs |
| Loss function | TaxonomicLoss (see below) |
| Label smoothing | 0.1 (both heads) |
| Early stopping | Patience = 10 epochs on species Top-1 |

**Why two heads?**

Genus prediction is strictly easier than species prediction — there are fewer genera than species, and the visual boundaries between genera are coarser and more distinct. Giving genus its own head means there is a second gradient signal flowing back through the shared backbone during training.

This acts as a **regularizer**: it pushes the backbone to learn features that generalize across the entire taxonomic tree rather than over-specializing on leaf-level species differences. The model learns "this looks like an *Amanita*" before it decides "and specifically, this is *Amanita phalloides*." This coarse-to-fine structure mirrors how expert mycologists actually identify mushrooms in the field.

**TaxonomicLoss:**

```
total_loss = 0.3 × CE(genus_logits, genus_labels)
           + 0.7 × CE(species_logits, species_labels)
```

Species gets 70% of the gradient weight since it is the primary objective. Genus gets 30% as an auxiliary regularizing signal. Label smoothing of 0.1 is applied to both terms to prevent the model from becoming overconfident on any single class. The 0.3/0.7 weights sum to 1.0, which keeps the total loss magnitude comparable to a single-head model and makes learning rate tuning easier.

**Training protocol:**

The backbone has pretrained knowledge from the YOLOv26 COCO training. The two heads are randomly initialized. If you unfreeze everything at once, the large random gradients from the heads will immediately damage the pretrained backbone weights — a phenomenon called catastrophic forgetting.

To prevent this:
1. **Epochs 1–5 (warmup):** Backbone is fully frozen. Only the genus and species heads train. They find stable starting representations without disturbing the backbone.
2. **Epoch 6 onward:** Backbone unfreezes at LR=1e-5. The differential learning rate (heads at 1e-3, backbone at 1e-5) means the backbone is nudged gently while the heads continue adapting quickly.

Training artifacts: `docs/franken_runs/taxonomic_yolo26/`

---

### 4.5 Model Comparison

| | YOLOv8n-cls | YOLOv26n-cls | EfficientNet-B0 | TaxonomicYOLO26 |
|---|---|---|---|---|
| **Parameters** | ~2.7M | 1.74M | ~5.3M | ~2M+ |
| **Model size** | ~6.2 MB | 3.6 MB | ~5 MB | ~4 MB |
| **Top-1 accuracy** | TBD | **88.1%** | TBD | TBD |
| **Top-5 accuracy** | TBD | **98.4%** | TBD | TBD |
| **Inference speed** | TBD | 0.2 ms/img | TBD | TBD |
| **Architecture origin** | Detection-adapted | Detection-adapted | Classification-native | Detection-adapted + dual head |
| **Outputs** | Species only | Species only | Species only | Genus + Species |
| **Pretrain source** | COCO | COCO | ImageNet | COCO (fine-tuned) |
| **Key strength** | Proven, fast | Smallest + best accuracy so far | Purpose-built for classification | Taxonomic regularization |
| **Script** | `scripts/training/yolo/train_yolo.py` | same | `scripts/training/cnn/train.py` | `scripts/training/franken/train.py` |

**The core architectural difference — YOLO vs EfficientNet:**

YOLO backbones carry detection-specific inductive biases (multi-scale feature pyramids, C2f blocks) that are optimized for finding *where* things are in an image. For classification — deciding *what* the image shows — these components are structural overhead. EfficientNet has none of that overhead; every architectural choice targets classification efficiency.

**The key innovation of TaxonomicYOLO26:**

No prior model in this project uses the biological taxonomy of mushrooms as a training signal. Mushrooms have a genus-species hierarchy (*Amanita phalloides*, *Amanita muscaria*, *Amanita velosa* share a genus). TaxonomicYOLO26 exploits this tree structure to create a harder learning problem that simultaneously produces a more generalizable backbone. Whether the dual-head approach measurably beats the single-head YOLOv26 in practice is the open experimental question.

---

## 5. Safety Pipeline

The pipeline runs after every vision model prediction. All logic lives in `services/brain_ui/pipeline/`.

### Step 1 — Vision Prediction
The Brain UI sends the uploaded image to the FastAPI Vision API (`POST /predict`). The API runs it through YOLOv26 and returns `(species_name, confidence)`.

### Step 2 — Ecological Context Lookup (`integration.py`)
The species name is looked up in `data/mushroom_context.csv`, returning:

| Field | Example |
|---|---|
| `toxicity_type` | "Deadly" |
| `habitat` | "Deciduous forests (often under oak)" |
| `season` | "Summer to autumn" |
| `region` | "Europe, North America, Australia" |
| `key_warnings` | "Contains fatal amatoxins. Responsible for the majority of fatal mushroom poisonings globally." |

If a species is not in the CSV, all fields default to "Unknown" and the system treats it as potentially dangerous.

### Step 3 — LLM Audit (`audit_layer.py`)
The species name, confidence, ecological context, user season, and user location are packaged into a structured prompt and sent to Gemini. The LLM must respond with exactly one of:

- `✅ PLAUSIBLE` — identification matches the environment
- `⚠️ SUSPICIOUS` — something doesn't add up (wrong season, wrong region, etc.)
- `🚨 DANGER` — species is toxic/deadly regardless of environmental match

Example: if the user is in Norway in winter but YOLO predicts a summer mushroom native to Brazil, the LLM catches this and flags it SUSPICIOUS.

### Step 4 — Risk Engine (`risk_engine.py`)
Four deterministic rules evaluate all evidence. These are plain Python `if/else` statements — the LLM cannot override them.

| Rule | Condition | Risk escalation |
|---|---|---|
| 1 | CSV toxicity contains "deadly", "fatal", "death", or "highly toxic" | → CRITICAL (always fires, no exceptions) |
| 2 | YOLO confidence < 70% | → HIGH |
| 3 | LLM verdict contains "suspicious", "danger", or "unlikely" | → HIGH |
| 4 | Species is toxic but not deadly | → MODERATE |

Risk levels escalate: LOW → MODERATE → HIGH → CRITICAL. Multiple rules can fire; the highest level wins.

| Final risk level | Recommendation |
|---|---|
| CRITICAL | "DO NOT CONSUME. This species is potentially DEADLY." |
| HIGH | "DO NOT CONSUME. Multiple risk factors detected." |
| MODERATE | "EXERCISE EXTREME CAUTION. Consult a local mycologist." |
| LOW | "Identification appears reliable. Always cross-reference with a local expert." |

### MLOps — Drift Detection
Any prediction with confidence < 70% triggers automatic drift logging: the image is saved to `data/drift_images/` with a timestamped filename containing the predicted species and confidence level. A Prometheus counter (`mushroom_drift_events_total`) is incremented per species so drift rate can be graphed in Grafana over time. These low-confidence images form the retraining dataset for future model versions.

---

## 6. MLOps & Deployment

### Local Stack (Docker Compose)

```bash
docker-compose -f deploy/docker-compose.yml up --build
```

| Service | Port | Purpose |
|---|---|---|
| Vision API (FastAPI) | 8000 | YOLO inference |
| Brain UI (Gradio) | 7860 | User-facing web app |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Metrics dashboards |

All four containers share a virtual Docker network (`mushroom-net`). The Brain UI reaches the Vision API by container name, not localhost.

**TFLite slim image:** The Vision API also ships a lightweight TFLite-only image (`services/vision_api/slim/`) that drops PyTorch entirely. The exported `best_float16.tflite` model produces identical predictions (Top-1: 88.1%, Top-5: 98.4%) at ~200 MB vs ~1.5 GB for the full PyTorch image. This is the default for Cloud Run.

### CI/CD Pipeline (GitHub Actions → Google Cloud Run)

Every `git push` to `master` triggers an automated pipeline:

```
git push origin master
        │
        ▼
GitHub Actions (.github/workflows/deploy.yml)
  1. Checkout repo
  2. Authenticate with GCP (via GCP_CREDENTIALS secret)
  3. Submit build to Cloud Build (reads cloudbuild.yaml)
        │
        ▼
Google Cloud Build
  Reads Dockerfile, builds image, pushes to Artifact Registry
        │
        ▼
Artifact Registry
  Stores vision-api:latest and brain-ui:latest
        │
        ▼
Cloud Run
  Pulls image, starts container, assigns public HTTPS URL
  Auto-scales to zero when idle, scales up under load
```

Cloud Run config (Vision API): 4 GiB RAM, 1 CPU, public HTTPS, scale-to-zero billing.

### Continuous Machine Learning (CML)

Pull requests trigger `.github/workflows/cml.yml`, which auto-generates a model evaluation report (confusion matrices, loss curves) and posts it as a PR comment. Model reviews happen inside the normal code review flow.

### Data Versioning (DVC + Hugging Face)

The 12 GB+ dataset and trained `.pt` weights are too large for Git. DVC tracks them as content-addressed pointers in the repo while the actual files live in a Hugging Face bucket. Any machine can run `dvc pull` to restore the full data ecosystem.

### Monitoring (Prometheus + Grafana)

The FastAPI Vision API exposes a `/metrics` endpoint via `prometheus-fastapi-instrumentator`. Prometheus scrapes it every 15 seconds. Grafana builds dashboards over the time-series data, tracking request latency, error rates, usage spikes, and the drift detection counter.

---

## 7. Project Structure

```
Mushroom/
├── data/
│   ├── dataset.yaml                   YOLO class config (169 species)
│   ├── dataset_info.md                Dataset source and split notes
│   ├── mushroom_context.csv           Ecological knowledge base
│   ├── mushroom_species.json          Master species name list
│   └── dataset_split/
│       ├── train/                     80% — one subfolder per species
│       ├── val/                       10%
│       └── test/                      10%
│
├── scripts/
│   ├── setup/
│   │   ├── prepare_dataset.py         Restructures Kaggle download into splits
│   │   ├── fix_dataset.py             Dataset cleaning utilities
│   │   └── upload_to_hf.py            Pushes weights to Hugging Face Hub
│   └── training/
│       ├── yolo/
│       │   ├── train_yolo.py          Trains YOLOv8n or YOLOv26n classifier
│       │   ├── export_tflite.py       Exports YOLO weights → TFLite
│       │   └── compare_pt_vs_tflite.py
│       ├── cnn/
│       │   ├── model.py               EfficientNet-B0 classifier head definition
│       │   ├── train.py               Training loop (pluggable loss/optimizer)
│       │   ├── dataset.py             DataLoader for mushroom split
│       │   ├── losses.py              CE / focal / label-smoothing builders
│       │   ├── evaluate.py            Post-training evaluation
│       │   ├── gradcam.py             Grad-CAM heatmap visualization
│       │   └── compare.py             Multi-run comparison utility
│       └── franken/
│           ├── model.py               TaxonomicYOLO26 dual-head architecture
│           ├── train.py               Training loop (warmup + early stopping)
│           ├── dataset.py             DataLoader returning genus + species labels
│           ├── losses.py              TaxonomicLoss (weighted genus + species CE)
│           ├── predict.py             Inference with both heads
│           └── visualize_activations.py
│
├── services/
│   ├── vision_api/
│   │   ├── main.py                    FastAPI server — YOLO inference endpoint
│   │   ├── Dockerfile                 Full PyTorch image (~1.5 GB)
│   │   ├── cloudbuild.yaml            Google Cloud Build config
│   │   ├── requirements.txt
│   │   └── slim/
│   │       ├── main.py                TFLite-only inference (no PyTorch)
│   │       ├── Dockerfile             Lightweight image (~200 MB)
│   │       └── requirements.txt
│   └── brain_ui/
│       ├── app.py                     Gradio UI — orchestrates the full pipeline
│       ├── Dockerfile
│       ├── cloudbuild.yaml
│       ├── requirements.txt
│       └── pipeline/
│           ├── predict.py             HTTP call to Vision API
│           ├── integration.py         CSV ecological context lookup
│           ├── audit_layer.py         LLM prompt construction + Gemini call
│           └── risk_engine.py         Deterministic safety rules
│
├── deploy/
│   └── docker-compose.yml             Local multi-container orchestration
│
├── .github/workflows/
│   ├── deploy.yml                     CI/CD: build + deploy on push to master
│   └── cml.yml                        ML report generation on pull requests
│
├── docs/
│   ├── project_overview.md            ← This file
│   ├── model_comparison.md            Training metrics log
│   ├── problems_log.md                Known issues and fixes
│   ├── cloud_deployment_pipeline.md   Detailed deployment walkthrough
│   ├── planning/plans.md              Original 6-phase project plan
│   ├── yolo_runs/
│   │   ├── mushroom_classifier_v1/    YOLOv8n results + weights
│   │   └── yolo26_classifier_v1/      YOLOv26 results + best.pt (production)
│   ├── cnn_runs/                      EfficientNet experiment results
│   └── franken_runs/                  TaxonomicYOLO26 experiment results
│
├── prometheus.yml                     Prometheus scrape config
├── dvc.yaml                           DVC pipeline definition
└── README.md                          Quick-start and ops guide
```
