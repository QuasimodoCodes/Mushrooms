# Mushroom Guardian: Multimodal AI Classification & Safety System

A production-grade, microservice-based AI safety system that identifies mushroom species visually (YOLOv8) and cross-references them with ecological context using an LLM audit layer. Built with end-to-end MLOps, CI/CD, and serverless cloud deployment.

---

## 🛠️ How to Operate the System 

Whether you are retraining the vision model or spinning up the microservices locally, here is your quick-start guide:

### 1. Syncing the Massive Dataset (DVC)
Since the 12GB+ dataset and heavy Pytorch weights are stored in the cloud (Hugging Face / Google Cloud or S3) to keep this repository small, use DVC to fetch them:
```bash
# Pull all raw data and weights into the local workspace
dvc pull
```

### 2. Training the YOLO Model Locally
If you want to train the model from scratch on your own GPU:
```bash
# Ensure your virtual environment is active
.\.venv\Scripts\Activate.ps1

# Run the training script directly
python scripts/training/train_yolo.py

# Expected Output: A new run folder inside docs/yolo_runs/ containing fresh .pt weights and metrics
```

### 3. Running Microservices Locally (Docker)
You can boot up the entire architecture on your local laptop using Docker Compose. This starts the FastAPI Vision layer, Gradio UI layer, and the MLOps monitoring stack simultaneously, bridging them over an internal Docker network.
```bash
# Build and launch all containers using the orchestration file in the deploy folder
docker-compose -f deploy/docker-compose.yml up --build -d

# What to see:
# -> The Gradio UI will be available at http://localhost:7860
# -> The Vision API will be listening on http://localhost:8000
# -> The API Metrics will be scraping at http://localhost:8000/metrics
# -> Prometheus will be available at http://localhost:9090
# -> Grafana Dashboards will be available at http://localhost:3000 (admin/admin)
```

### 4. Running Microservices Locally (Python/Terminal)
If you don't want to use Docker and prefer two raw Python terminal tabs:
```bash
# Terminal 1: Boot the Vision API
cd services/vision_api/
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Boot the Web UI
cd services/brain_ui/
python app.py
```

### 5. Deploying to the Cloud & Reviewing Models
Because we set up CI/CD using GitHub Actions, deployment and model evaluation are completely hands-off! 

**To review model metrics (CML):**
```bash
# Push your code to any new branch to open a Pull Request
git checkout -b new-model-update
git add .
git commit -m "Trained a new YOLO model"
git push origin new-model-update
```
> *Behind the scenes: CML will automatically run and post the new model's confusion matrices and training graphs directly to your GitHub PR so your team can review the accuracy.*

**To deploy the live application:**
```bash
# Merge your PR or push directly to the 'master' branch
git checkout master
git merge new-model-update
git push origin master
```
> *Behind the scenes: GitHub Actions will detect the push to master, trigger Google Cloud Build to compile your Docker images, and roll out the new containers serverlessly to Google Cloud Run.*

---

## Project Structure

```text
Mushroom/
├── .github/workflows/        ← CI/CD Automations (GitHub Actions & CML)
├── data/
│   ├── dataset.yaml          ← YOLO class mapping config
│   ├── mushroom_context.csv  ← Ecological rules (Knowledge Base)
│   └── drift_images/         ← Auto-saved low-confidence field data
├── deploy/                   ← Infrastructure as Code 
│   ├── docker-compose.yml    ← Local multi-container orchestration
│   └── prometheus.yml        ← Prometheus metrics scraping config
├── docs/
│   ├── planning/             ← Brainstorms and schema planners
│   └── yolo_runs/            ← YOLO metrics, loss graphs, PR curves
├── scripts/                  
│   ├── setup/                ← Scripts for scraping data & managing HF uploads
│   └── training/             ← YOLO model training logic
├── services/                 ← Containerized Microservices
│   ├── brain_ui/             ← Gradio UI, LLM Audit, Risk Engine (app.py)
│   │   ├── Dockerfile
│   │   └── pipeline/         ← Core evaluation logic scripts
│   └── vision_api/           ← FastAPI YOLO Server (main.py)
│       └── Dockerfile
├── README.md                 ← You are here
└── dvc.yaml                  ← Data Version Control pipelines

---

## 🏗️ System Architecture & Workflow

This project has evolved from a local Python script into a robust, cloud-ready microservice architecture. Here is how the whole pipeline works end-to-end:

### 1. The Vision API (FastAPI + YOLOv8)

Instead of loading massive PyTorch models directly into the user interface, we decoupled the vision logic into its own containerized microservice: the **Vision API**.

- A field user uploads an image of a mushroom via the web UI.
- The UI sends a fast HTTP POST request to the Vision API (`services/vision_api/`).
- **YOLOv8 Nano** processes the image, extracting the `Top 1 Predicted Class` (e.g., _Amanita muscaria_) and its `Confidence Score` (e.g., _0.91_ or 91%).

### 2. Context Fetching (CSV Knowledge Base)

Visual identification alone is incredibly dangerous in the wild. A mushroom might look edible, but if it is growing on the wrong type of wood or in the wrong season, it is likely a toxic look-alike.

- Instead of trying to teach YOLO abstract ecological rules, we maintain a structured **Knowledge Base (`data/mushroom_context.csv`)**.
- We use `pandas` to take YOLO's top prediction and fetch its toxicity status, primary growing season, geographic region, and key warnings from the CSV.

### 3. The LLM Audit Layer (Llama3 / Gemini)

This is the "Reasoning" phase of the pipeline.

- We merge the YOLO visual prediction, the structured CSV ecological rules, and the field user's provided metadata (GPS location & current season) into a formatted text prompt.
- An **LLM (Large Language Model)** acts as a safety auditor. We ask it: _"Does this visual prediction logically make sense given the user's current environment?"_
- If a user is situated in Norway in the dead of Winter, but the YOLO model predicts a Summer mushroom native to Brazil, the LLM intelligently catches the hallucination and flags the prediction as unsafe!

### 4. Risk-Aware Decision Engine

While AI systems probabilistically hallucinate, hard-coded software rules do not. We mapped specific safety gates as a final fallback:

- **Rule 1:** If the **YOLO Confidence** is below 70%, the system aborts and warns the user of an unsafe visual lock.
- **Rule 2:** If the **LLM Audit Layer** detects an ecological mismatch, the system vetoes the prediction, regardless of how confident the Vision model was.

---

## ☁️ Cloud Deployment & Microservices

To make the application globally accessible and horizontally scalable, the architecture is entirely containerized and deployed to the cloud:

1. **Dockerization:** Both the Vision API and Brain UI have their own unique `Dockerfile`s. This decoupling allows us to isolate heavy hardware/CUDA dependencies strictly to the Vision API, while keeping the UI container extremely lightweight.
2. **Google Cloud Run:** Using **Google Cloud Build**, the Docker containers are compiled into images and hosted serverlessly on Cloud Run, scaling instantly with user traffic.
3. **Environment Injection:** The UI reaches the remote Vision API securely via public cloud URLs injected into the container's environment variables (`VISION_API_URL`).

---

## 🚀 MLOps & Production Engineering

Machine Learning doesn't stop when you save a `.pt` weights file. We implemented strict MLOps principles across the repository:

### Data Version Control (DVC) + Hugging Face

Git was fundamentally built for code text, not 12 Gigabyte datasets of images or heavy 50MB PyTorch binaries.

- We utilize **DVC (Data Version Control)** to independently track the system's massive image datasets and computed `best.pt` model weights.
- DVC natively uploads these heavy assets to a **Hugging Face bucket (Cloud Storage)**, leaving only tiny `.dvc` text-based pointers/hashes inside this Git repository.
- **Why?** This keeps standard `git clone` operations lightning fast, entirely avoids GitHub's harsh 100MB file limits, and massively accelerates development portability. If we ever rent a blank Cloud GPU to retrain the model, that new machine can simply run `dvc pull` to instantly restore the entire data ecosystem directly from Hugging Face!

### Continuous Integration / Deployment (CI/CD)

- The repository is rigged with **GitHub Actions** (`.github/workflows/deploy.yml`). Pushing a validated code update to the `main` branch automatically triggers cloud runners to spin up newly patched Docker images and roll them out live to the Google Cloud Run production environment.

### Observability, Logging & Health Checks

- **Centralized Logging:** Legacy monolithic `print()` logic was upgraded to Python's robust `logging` module so we can view streaming cloud outputs.
- **API Health Endpoints:** The Vision API instances feature a root `/health` heartbeat endpoint that allows Google Cloud's load balancers to easily verify that a container is still actively resolving requests.

### Model Drift Detection

- Machine learning models naturally degrade in production environments when exposed to new conditions (e.g. dirty camera lenses, crushed mushroom caps).
- The pipeline handles this using active **Drift Detection**: Any time a user submits an image and the YOLO model yields an uncertain confidence score **< 0.70**, the architecture natively intercepts the transmission and seamlessly saves the input image to an isolated `data/drift_images/` staging pool. These failure cases manually construct our next dataset for future model fine-tuning!

### Continuous Machine Learning (CML)
- We use **CML (Continuous Machine Learning)** to automatically generate and post model evaluation reports.
- Whenever code is pushed to a Pull Request on the `master` branch, a GitHub Action triggered by `iterative/setup-cml` automatically creates a Markdown report (`report.md`) embedding the visual accuracy charts (like confusion matrices and loss curves) generated by our YOLO training runs and tracked by DVC.
- It then executes `cml comment create report.md` to post these visual analytics directly into the PR timeline, bringing ML model evaluation natively into standard code review.

### Model Monitoring (Prometheus & Grafana)
- Production insights are gathered at the microservice level. We instrumented the FastAPI application to natively expose real-time metrics on a `/metrics` endpoint using `prometheus-fastapi-instrumentator`.
- This automatically tracks standard HTTP metrics like `http_requests_total`, `http_request_duration_seconds`, and error counts without any custom application code.
- Our local `docker-compose` architecture spins up **Prometheus** (configured via `prometheus.yml` to actively scrape the Vision API every 15 seconds) and **Grafana** (to construct flexible, visual dashboards over the Prometheus timeseries data). This allows us to track API latency, usage spikes, and system health in a centralized visualization layer.