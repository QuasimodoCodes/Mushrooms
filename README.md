# Mushroom Guardian — Multimodal AI Safety System

A production-grade AI system that identifies mushroom species from a photo and delivers a safety verdict using two independent AI layers: a fast vision model for identification and a multimodal LLM that visually verifies the result. Built with a full MLOps stack, CI/CD pipeline, and serverless cloud deployment.

> **Live demo:** [brain-ui-849718487429.us-central1.run.app](https://brain-ui-849718487429.us-central1.run.app)

> For the full technical write-up, see [docs/project_overview.md](docs/project_overview.md).

---

## Live Services

| Service | URL | Description |
|---|---|---|
| **Brain UI** | [brain-ui-849718487429.us-central1.run.app](https://brain-ui-849718487429.us-central1.run.app) | Main interface — upload a photo, get a safety report |
| **Vision API** | [vision-api-849718487429.us-central1.run.app](https://vision-api-849718487429.us-central1.run.app) | YOLO inference REST API |
| **API Docs** | [vision-api-849718487429.us-central1.run.app/docs](https://vision-api-849718487429.us-central1.run.app/docs) | Interactive Swagger UI for the Vision API |

---

## How It Works

Every submitted photo goes through four stages before a safety verdict is produced:

```
Photo + season + location
        │
        ▼
1. YOLOv26 Vision API        →  species name + confidence score
        │
        ▼
2. Ecological JSON lookup    →  toxicity / habitat / cap / gills / stem / lookalikes
        │
        ▼
3. LLM Visual Audit          →  AGREE / DISAGREE / DANGER
   (Gemma 4 / Gemini sees        "Does this photo actually look like X?"
    the actual photo)             + field checks the forager should perform
        │
        ▼
4. Risk Engine (Python rules) →  CRITICAL / HIGH / MODERATE / LOW
```

**The key insight:** YOLO identifies fast at 88.1% accuracy. The LLM sees the same photo and independently verifies — if they disagree, risk escalates automatically. Python hard-coded rules provide a final safety guarantee that no AI can override.

---

## Models

Four architectures trained on the same 169-species dataset (`zlatan599/mushroom1` on Kaggle, ~689k images, 80/10/10 split).

| Model | Params | Top-1 | Top-5 | File size | Notes |
|---|---|---|---|---|---|
| YOLOv8n-cls | 2.7M | 86.8% | 97.9% | 3.4 MB | Baseline |
| **YOLOv26n-cls** | **1.74M** | **88.1%** | **98.4%** | **3.6 MB** | **Production — best size/accuracy balance** |
| EfficientNet-B0 | 5.3M | 89.5% | 97.9% | 17.2 MB | Highest top-1, but 5× larger |
| TaxonomicYOLO26 | ~2M | 79.7%† | — | 25.0 MB | Dual-head: genus (81.6%) + species simultaneously |

† Lower species accuracy is expected — the shared backbone solves a harder dual task than single-head models.

Production uses YOLOv26n exported to **TFLite float16** — 0.2 ms inference, ~200 MB Docker image (vs ~1.5 GB for PyTorch).

---

## Quick Start

### Configuration — one file controls everything

All runtime switches live in **`config.py`**. No other file needs editing to swap the LLM, vision model, or safety thresholds.

```python
# ── Vision model ─────────────────────────────────────────────
YOLO_RUN_NAME = "yolo26_tflite"      # folder under docs/yolo_runs/
MODEL_FORMAT  = "tflite"             # "pt" → PyTorch  |  "tflite" → lightweight

# ── LLM provider ─────────────────────────────────────────────
ACTIVE_LLM_PROVIDER = "ollama"       # "ollama" → local  |  "gemini" → cloud API
OLLAMA_MODEL        = "gemma4:e2b"   # multimodal models receive the photo automatically
GEMINI_MODEL        = "gemma-4-26b-a4b-it"   # or "gemini-2.0-flash"

# ── Safety threshold ─────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70          # below this → risk escalates to HIGH
```

### Run locally with Docker

```bash
docker-compose -f deploy/docker-compose.yml up --build
```

| Service | URL |
|---|---|
| Brain UI | http://localhost:7860 |
| Vision API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

### Run locally without Docker

```bash
python launch.py
```

`launch.py` handles everything automatically:
- Frees ports 8000 and 7860 if occupied
- Auto-starts Ollama (when `ACTIVE_LLM_PROVIDER = "ollama"`)
- Starts Vision API on port 8000
- Starts Brain UI on port 7860

> **API keys:** Add a `.env` file in the project root before running:
> ```
> GEMINI_API_KEY=your_key_here
> HF_TOKEN=your_token_here
> ```

### Pull dataset and weights (DVC)

```bash
dvc pull   # downloads dataset and trained weights from Hugging Face
```

---

## Cloud Deployment

Push to `master` and everything deploys automatically via GitHub Actions:

```
git push origin master
        │
        ▼
GitHub Actions  →  authenticates with GCP
        │
        ▼
Cloud Build     →  builds Docker images
        │
        ▼
Artifact Registry  →  stores vision-api:latest + brain-ui:latest
        │
        ▼
Cloud Run       →  live HTTPS endpoints, auto-scales to zero
```

**Secrets** stored in GitHub repo settings (never in code):
- `GCP_CREDENTIALS` — service account JSON for Cloud Build + Cloud Run
- `GEMINI_API_KEY` — injected into the Brain UI container at runtime

---

## MLOps

### Drift Detection
Predictions below the 70% confidence threshold are automatically saved to `data/drift_images/` and tracked via a Prometheus counter (`mushroom_drift_events_total`). These images form the retraining dataset for future versions.

> **Note:** Drift image saving is functional locally and in Docker. Cloud Run containers have an ephemeral filesystem, so images are not persisted in production (GCS integration skipped due to cost).

### Automated Model Reports (CML)
Opening a pull request triggers `.github/workflows/cml.yml`, which runs model evaluation and posts confusion matrices and loss curves directly as a PR comment.

### Monitoring (Prometheus + Grafana)
The Vision API exposes `/metrics` via `prometheus-fastapi-instrumentator`. Prometheus scrapes it every 15 seconds. Grafana dashboards track latency, error rates, and drift counts. Available locally via Docker Compose.

### Data Versioning (DVC + Hugging Face)
The 12 GB+ dataset and `.pt` weights are tracked by DVC and stored on Hugging Face. `dvc pull` restores everything on any machine — Git only stores the `.dvc` pointer files.

---

## Training

```bash
# YOLOv26 classifier
python scripts/training/yolo/train_yolo.py

# EfficientNet-B0
python scripts/training/cnn/train.py

# TaxonomicYOLO26 (dual-head)
python scripts/training/franken/train.py
```

> Requires an NVIDIA GPU with CUDA.

To experiment with a new YOLO run:
1. Change `name` in `train_yolo.py` to a new folder (e.g. `"yolo26_v2"`)
2. Train — results save to `docs/yolo_runs/<name>/`
3. Update `YOLO_RUN_NAME` in `config.py`

---

## Project Structure

```
Mushroom/
├── config.py                          ← Single control file (LLM, model, thresholds)
├── launch.py                          ← One command to start everything locally
│
├── data/
│   ├── dataset.yaml                   YOLO class config (169 species)
│   └── mushroom_context.json          AI-enriched ecological database (169 species, 12 fields)
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
│   │   ├── main.py                    FastAPI — YOLO inference endpoint
│   │   ├── Dockerfile                 Full PyTorch image (~1.5 GB)
│   │   └── slim/                      TFLite image (~200 MB) — used in production
│   └── brain_ui/
│       ├── app.py                     Gradio UI — orchestrates the pipeline
│       └── pipeline/
│           ├── predict.py             Calls Vision API
│           ├── integration.py         JSON ecological context lookup (cached)
│           ├── audit_layer.py         Visual + text LLM audit (sees the actual photo)
│           ├── llm_provider.py        Multimodal LLM router (Ollama + Gemini)
│           └── risk_engine.py         Deterministic safety rules
│
├── deploy/
│   └── docker-compose.yml             Local multi-container orchestration
│
├── .github/workflows/
│   ├── deploy.yml                     CI/CD — build + deploy on push to master
│   └── cml.yml                        Automatic model reports on pull requests
│
└── docs/
    ├── project_overview.md            Full technical write-up
    ├── model_comparison.md            Training metrics log
    ├── cloud_deployment_pipeline.md   Deployment walkthrough
    └── yolo_runs/                     Training artifacts + weights
```

---

*Dataset: `zlatan599/mushroom1` (Kaggle) — 169 species, ~689k images. Primary model: YOLOv26n-cls, 88.1% Top-1, 0.2 ms inference.*
