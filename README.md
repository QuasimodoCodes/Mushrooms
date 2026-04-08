# Mushroom Guardian: Multimodal AI Classification & Safety System

A production-grade, microservice-based AI safety system that identifies mushroom species from photos and cross-references them with ecological context using an LLM audit layer. Built with multiple vision model architectures, end-to-end MLOps, CI/CD, and serverless cloud deployment.

> For a full technical deep-dive into every model and design decision, see [docs/project_overview.md](docs/project_overview.md).

---

## Quick Start

### 1. Configuration — the only file you need to touch

All runtime switches live in **`config.py`** at the project root. You never need to edit any other file to change the LLM, swap the vision model, or adjust safety thresholds.

```python
# ── Switch your vision model ──────────────────────────────────
YOLO_RUN_NAME = "yolo26_classifier_v1"   # folder name under docs/yolo_runs/

MODEL_FORMAT = "pt"      # "pt"     → full PyTorch (~1.5 GB Docker image)
                         # "tflite" → lightweight export (~200 MB, no PyTorch)

# ── Switch your LLM ──────────────────────────────────────────
ACTIVE_LLM_PROVIDER = "ollama"   # "ollama" → local/free  |  "gemini" → cloud

OLLAMA_MODEL = "llama3:latest"
GEMINI_MODEL = "gemini-3-flash-preview"

# ── Safety threshold ─────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70   # below this → risk escalates to HIGH
```

### 2. Pull dataset and weights (DVC)

```bash
dvc pull   # downloads the 12 GB+ dataset and trained .pt weights from cloud storage
```

### 3. Run locally with Docker

```bash
docker-compose -f deploy/docker-compose.yml up --build
```

| Service | URL | Purpose |
|---|---|---|
| Brain UI (Gradio) | http://localhost:7860 | Upload a photo, get a safety report |
| Vision API (FastAPI) | http://localhost:8000 | YOLO inference endpoint |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Dashboards (admin / admin) |

> To use the lightweight TFLite image instead of PyTorch, set `MODEL_FORMAT = "tflite"` in `config.py` and change the dockerfile path in `deploy/docker-compose.yml` to `services/vision_api/slim/Dockerfile`.

### 4. Run locally without Docker

```bash
python launch.py
```

`launch.py` handles everything in one command:
- Clears ports 8000 and 7860 if they are already occupied
- Auto-starts Ollama (if `ACTIVE_LLM_PROVIDER = "ollama"` in `config.py`)
- Starts the Vision API on port 8000
- Starts the Brain UI on port 7860

Open **http://localhost:7860** in your browser. Press `Ctrl+C` to stop both services.

> **API keys:** Create a `.env` file in the project root (never committed) before running:
> ```
> GEMINI_API_KEY=your_key_here
> ```
> `launch.py` loads this automatically so subprocesses inherit the keys.

---

## How the Pipeline Works

Every image goes through four stages before a safety verdict is produced:

```
Photo + season + location
        │
        ▼
1. YOLOv26 Vision API      →  (species_name, confidence)
        │
        ▼
2. Ecological CSV lookup   →  toxicity / habitat / season / region
        │
        ▼
3. LLM Audit (Gemini/Ollama) →  PLAUSIBLE / SUSPICIOUS / DANGER
        │
        ▼
4. Risk Engine (Python rules) →  CRITICAL / HIGH / MODERATE / LOW
```

**The philosophy:** the LLM provides explanations, Python provides guarantees. The risk engine uses hard-coded `if/else` rules that the LLM cannot override — if the CSV says a species is deadly, the verdict is CRITICAL regardless of anything else.

---

## Models

Four architectures were trained on the same 169-species dataset (`zlatan599/mushroom1` on Kaggle, 80/10/10 split, ~689k images).

| Model | Params | Size | Top-1 | Top-5 | Key difference |
|---|---|---|---|---|---|
| YOLOv8n-cls | ~2.7M | ~6.2 MB | TBD | TBD | Baseline — detection backbone repurposed for classification |
| **YOLOv26n-cls** | **1.74M** | **3.6 MB** | **88.1%** | **98.4%** | **Production model — smaller and more accurate than v8** |
| EfficientNet-B0 | ~5.3M | ~5 MB | TBD | TBD | Classification-native architecture (compound scaling law) |
| TaxonomicYOLO26 | ~2M+ | ~4 MB | TBD | TBD | Dual-head: predicts genus + species simultaneously |

**YOLOv26 training result:** Loss decreased smoothly for all 50 epochs. Epoch 49 hit the lowest validation loss (0.41535). Epoch 50 ticked up by +0.0003 — a perfect stop right at the overfitting boundary. Inference speed: 0.2 ms/image on an RTX 3070 Ti.

**EfficientNet-B0** replaces the stock 1000-class head with `Dropout(0.3) → Linear(1280 → 169)`. It was built as a direct comparison to YOLO since EfficientNet was designed from the ground up for classification, not detection.

**TaxonomicYOLO26** strips the YOLO26n-cls head and adds two parallel heads — one for genus, one for species. The genus head acts as a regularizer on the shared backbone, encouraging it to learn features that generalize across the taxonomic tree. Loss: `0.3 × genus_CE + 0.7 × species_CE` with label smoothing 0.1.

---

## Training

Each model has its own script. Hyperparameters are at the top of each file.

```bash
# YOLOv26 classifier
python scripts/training/yolo/train_yolo.py

# EfficientNet-B0  (supports --optimizer and --loss flags)
python scripts/training/cnn/train.py
python scripts/training/cnn/train.py --optimizer sgd --loss focal_smooth

# TaxonomicYOLO26 (dual-head franken model)
python scripts/training/franken/train.py
```

> Requires an NVIDIA GPU with CUDA. CPU training on 169 classes is impractically slow.

**To experiment with a new YOLO run:**
1. Open `scripts/training/yolo/train_yolo.py` and change `name` to a new folder name (e.g. `"yolo26_experiment_v2"`)
2. Train — results save to `docs/yolo_runs/<name>/`
3. Update `YOLO_RUN_NAME` in `config.py` to point at your new run
4. Restart the Vision API — it loads the new weights automatically

---

## Cloud Deployment

Push to `master` and everything deploys automatically:

```
git push origin master
        │
        ▼
GitHub Actions  →  authenticates with GCP
        │
        ▼
Cloud Build     →  builds Docker image from Dockerfile
        │
        ▼
Artifact Registry  →  stores vision-api:latest + brain-ui:latest
        │
        ▼
Cloud Run       →  serves live HTTPS endpoint, auto-scales to zero
```

**Vision API spec on Cloud Run:** 4 GiB RAM, 1 CPU, public HTTPS, scale-to-zero billing.

The slim TFLite image (`services/vision_api/slim/`) is used by default on Cloud Run — it produces identical predictions at ~200 MB vs ~1.5 GB for the full PyTorch image.

**Secrets** are stored in GitHub repo settings (never in code):
- `GCP_CREDENTIALS` — service account JSON for Cloud Build + Cloud Run
- `GEMINI_API_KEY` — passed into the Brain UI container at runtime

---

## MLOps

### Drift Detection
Any prediction with confidence below `DRIFT_CONFIDENCE_THRESHOLD` (set in `config.py`) is automatically saved to `data/drift_images/` with a timestamped filename. A Prometheus counter tracks drift events per species — visible in Grafana. These images become the retraining dataset for future model versions.

### CML — Automatic Model Reports
Opening a pull request triggers `.github/workflows/cml.yml`, which generates a model evaluation report (confusion matrices, loss curves) and posts it as a PR comment. Model review happens inside the normal code review flow.

### Monitoring
The FastAPI Vision API exposes `/metrics` via `prometheus-fastapi-instrumentator`. Prometheus scrapes it every 15 seconds. Grafana builds dashboards over the time-series data for latency, error rates, and drift counts.

### Data Versioning (DVC + Hugging Face)
The 12 GB+ image dataset and `.pt` weights are tracked by DVC and stored in a Hugging Face bucket. `dvc pull` restores everything on any machine. Git only stores the tiny `.dvc` pointer files.

---

## Project Structure

```
Mushroom/
├── config.py                          ← Runtime switches (LLM, model, thresholds)
├── launch.py                          ← One command to start everything locally
│
├── data/
│   ├── dataset.yaml                   YOLO class config (169 species)
│   ├── mushroom_context.csv           Ecological knowledge base
│   └── dataset_split/
│       ├── train/  val/  test/        80 / 10 / 10 split
│
├── scripts/
│   ├── setup/
│   │   ├── prepare_dataset.py         Restructures Kaggle download into splits
│   │   └── upload_to_hf.py            Pushes weights to Hugging Face
│   └── training/
│       ├── yolo/train_yolo.py         YOLOv8n / YOLOv26n classifier training
│       ├── cnn/                       EfficientNet-B0 training + GradCAM
│       └── franken/                   TaxonomicYOLO26 dual-head training
│
├── services/
│   ├── vision_api/
│   │   ├── main.py                    FastAPI — serves YOLO predictions
│   │   ├── Dockerfile                 Full PyTorch image (~1.5 GB)
│   │   └── slim/                      TFLite-only image (~200 MB)
│   └── brain_ui/
│       ├── app.py                     Gradio UI — orchestrates the pipeline
│       └── pipeline/
│           ├── predict.py             Calls Vision API
│           ├── integration.py         CSV ecological context lookup
│           ├── audit_layer.py         LLM prompt + Gemini/Ollama call
│           └── risk_engine.py         Deterministic safety rules
│
├── deploy/
│   └── docker-compose.yml             Local multi-container orchestration
│
├── .github/workflows/
│   ├── deploy.yml                     CI/CD: build + deploy on push to master
│   └── cml.yml                        ML report generation on pull requests
│
└── docs/
    ├── project_overview.md            Full technical write-up of every model
    ├── model_comparison.md            Training metrics log
    ├── cloud_deployment_pipeline.md   Deployment walkthrough
    ├── yolo_runs/                     YOLO training artifacts + weights
    ├── cnn_runs/                      EfficientNet experiment results
    └── franken_runs/                  TaxonomicYOLO26 experiment results
```

---

*Dataset: `zlatan599/mushroom1` (Kaggle) — 169 species, ~689k images. Primary production model: YOLOv26n-cls, 88.1% Top-1 accuracy, 0.2 ms inference.*
