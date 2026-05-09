# Mushroom Classifier

A production-grade AI system that identifies mushroom species from a photo and gives a verdict. It combines a fast on-device vision model (YOLOv26) with an LLM audit (Gemini) and a knowledge base of ecological rules to catch dangerous look-alikes.

---

## Live App

| Service | URL |
|:--------|:----|
| **Brain UI** (the app) | https://brain-ui-849718487429.us-central1.run.app |
| **Vision API** (backend) | https://vision-api-849718487429.us-central1.run.app |

The Brain UI is what you open in a browser. The Vision API is the backend it calls — you do not need to interact with it directly.

---

## Using the App

1. Open the Brain UI link above.
2. Upload a photo of a mushroom (drag-and-drop or click to browse).
3. Select the current **season** from the dropdown.
4. Type your **location** (e.g. `Norway`, `Pacific Northwest`, `Denmark`).
5. Click **Submit**.

The app streams its progress in real time:

- Sends the image to the Vision API → YOLO identifies the species and confidence score
- Looks up the species in the knowledge base (toxicity, habitat, season, region, warnings)
- Asks Gemini to audit whether the visual prediction makes sense given your location and season
- Runs hard rules (confidence < 70% → automatic unsafe flag regardless of LLM verdict)
- Returns a final **SAFE / UNSAFE / UNCERTAIN** verdict with an explanation

**What the output tells you:**
- The predicted species name and confidence score
- Ecological context (toxicity type, typical habitat, season, region)
- The LLM's reasoning about whether the prediction fits your environment
- The final risk decision and any warnings

> The app is an aid, not a substitute for expert identification. Never eat a wild mushroom based on an AI verdict alone.

---

## How to Run Locally

### Option A — Docker (recommended, runs everything)

```bash
# Start the Vision API, Brain UI, Prometheus, and Grafana together
docker-compose -f deploy/docker-compose.yml up --build -d
```

| Service | Local URL |
|:--------|:----------|
| Brain UI | http://localhost:7860 |
| Vision API | http://localhost:8000 |
| Vision API metrics | http://localhost:8000/metrics |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (login: admin / admin) |

### Option B — Python terminals (no Docker)

```bash
# Terminal 1: Vision API
cd services/vision_api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Brain UI
cd services/brain_ui
python app.py
```

Then open http://localhost:7860.

### Prerequisites

- Python 3.12
- Docker + Docker Compose (for Option A)
- A `.env` file in the project root with:

```
GEMINI_API_KEY=your_key_here
```

- Model weights and dataset pulled via DVC:

```bash
dvc pull
```

---

## Deployment

Deployment is fully automated. Push to `master` and GitHub Actions handles the rest:

```bash
git push origin master
```

This triggers `.github/workflows/deploy.yml`, which:
1. Authenticates with Google Cloud using the `GCP_CREDENTIALS` secret
2. Runs Cloud Build to build both Docker images
3. Deploys them to Cloud Run in `us-central1`

The Brain UI container receives the Vision API URL and Gemini key as environment variables — no config changes needed.

To check the status of a running service:

```bash
gcloud run services describe vision-api --region us-central1
gcloud run services describe brain-ui --region us-central1
```

---

## Training a New Model

```bash
# 1. Create a branch
git checkout -b my-experiment

# 2. Edit hyperparameters in scripts/training/train_yolo.py
#    Change `name` to avoid overwriting the baseline run

# 3. Pull the dataset (12 GB+)
dvc pull

# 4. Train (requires CUDA GPU)
python scripts/training/train_yolo.py
# Output saved to docs/yolo_runs/<name>/

# 5. Push and open a PR
git push origin my-experiment
```

Opening a PR automatically triggers the CML workflow, which posts confusion matrices and training curves as a comment on the PR.

---

## What Each Service Does

| Service | Path | What it does |
|:--------|:-----|:-------------|
| **Vision API** | `services/vision_api/` | FastAPI server. Accepts a mushroom image, runs YOLOv26, returns species name + confidence. Default image uses TFLite (~200 MB). Full PyTorch image also available. |
| **Brain UI** | `services/brain_ui/` | Gradio web UI. Orchestrates the full pipeline: calls Vision API, fetches ecological context, runs LLM audit, applies risk rules, streams results to the user. |
| **Pipeline modules** | `services/brain_ui/pipeline/` | `predict.py` — calls Vision API. `integration.py` — knowledge base lookup. `audit_layer.py` — LLM audit. `risk_engine.py` — final decision logic. `llm_provider.py` — Gemini / Ollama abstraction. |

---

## Directory Guide

```
Mushroom/
├── services/
│   ├── vision_api/           ← FastAPI prediction server
│   │   ├── main.py           ← PyTorch/ultralytics inference
│   │   ├── Dockerfile        ← Full PyTorch image (~1.5 GB)
│   │   ├── cloudbuild.yaml   ← Google Cloud Build config
│   │   └── slim/             ← TFLite-only image (~200 MB, default for cloud)
│   └── brain_ui/
│       ├── app.py            ← Gradio UI + pipeline orchestration
│       ├── Dockerfile
│       ├── cloudbuild.yaml
│       └── pipeline/         ← predict, integration, audit, risk, llm modules
│
├── data/
│   ├── mushroom_context.csv  ← Knowledge base: toxicity, habitat, season, region
│   ├── mushroom_species.json ← Species list
│   ├── dataset.yaml          ← YOLO class config
│   └── drift_images/         ← Auto-saved low-confidence images (gitignored)
│
├── scripts/
│   ├── training/             ← YOLO training scripts
│   └── setup/                ← Dataset scraping and Hugging Face upload scripts
│
├── docs/
│   ├── yolo_runs/            ← Training outputs: weights, metrics, loss curves
│   │   └── yolo26_classifier_v1/weights/best.pt  ← Production weights
│   ├── cloud_deployment_pipeline.md  ← Full deployment walkthrough
│   ├── model_comparison.md
│   └── problems_log.md
│
├── ablation/                 ← Ablation study: vision model × LLM combinations
│   └── README.md             ← Study methodology and findings
│
├── benchmarks/               ← Speed and accuracy benchmarks
│
├── deploy/
│   ├── docker-compose.yml    ← Local multi-container setup
│   └── prometheus.yml        ← Prometheus scrape config
│
├── .github/workflows/
│   ├── deploy.yml            ← CI/CD: build + deploy on push to master
│   └── cml.yml               ← Auto-generates model report on PRs
│
├── config.py                 ← Central config (paths, thresholds, LLM provider)
├── dvc.yaml                  ← DVC pipeline for dataset and weights versioning
└── prometheus.yml            ← Prometheus config for local monitoring
```

---

## MLOps Features

| Feature | Where | What it does |
|:--------|:------|:-------------|
| **DVC** | `dvc.yaml`, `.dvc/` | Tracks the 12 GB+ dataset and model weights in Hugging Face. Run `dvc pull` to restore. |
| **CI/CD** | `.github/workflows/deploy.yml` | Push to master → automatic Cloud Run deployment. |
| **CML** | `.github/workflows/cml.yml` | Opens a PR → training graphs posted as a PR comment automatically. |
| **Drift detection** | `services/brain_ui/app.py` | Confidence < 70% → image saved to `data/drift_images/` for future retraining. |
| **Prometheus metrics** | `services/vision_api/main.py` | `/metrics` endpoint exposes request counts, latency, error rates. |
| **Grafana** | `deploy/docker-compose.yml` | Dashboard over Prometheus data. Available at localhost:3000 when running locally. |
| **Health check** | `services/vision_api/main.py` | `GET /health` returns `{"status": "healthy"}` — used by Cloud Run load balancer. |
