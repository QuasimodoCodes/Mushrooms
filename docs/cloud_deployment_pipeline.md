# From Local Code to the Cloud: A Complete Deployment Pipeline

> This document walks through every layer between your working code and a live application on Google Cloud — using the Mushroom Vision API project as a running example.

---

## Table of Contents

1. [Local Code — The App Itself](#1-local-code--the-app-itself)
2. [Dockerfile — Wrapping the App](#2-dockerfile--wrapping-the-app)
3. [Docker Image — The Built Artifact](#3-docker-image--the-built-artifact)
4. [Docker Container — Running the Image](#4-docker-container--running-the-image)
5. [GitHub Repository — Storing Everything](#5-github-repository--storing-everything)
6. [GitHub Actions — The Automation Engine](#6-github-actions--the-automation-engine)
7. [YAML Files — The Language of Pipelines](#7-yaml-files--the-language-of-pipelines)
8. [Google Cloud — Where It All Runs](#8-google-cloud--where-it-all-runs)
9. [The Full Pipeline — End to End](#9-the-full-pipeline--end-to-end)

---

## 1. Local Code — The App Itself

> 📁 **Project files:**
>
> - `services/vision_api/main.py` — FastAPI prediction server
> - `services/vision_api/requirements.txt` — Python dependencies for the API
> - `services/brain_ui/app.py` — Gradio front-end UI
> - `services/brain_ui/requirements.txt` — Python dependencies for the UI
> - `docs/yolo_runs/yolo26_classifier_v1/weights/best.pt` — Trained YOLO model weights
> - `data/mushroom_context.csv` — Species context data used by the Brain UI

### What it is

This is the software you wrote. In our case it's a **FastAPI** application that accepts an uploaded mushroom photo, runs it through a trained YOLO classification model, and returns the species name and confidence score.

```python
# services/vision_api/main.py  (simplified)
app = FastAPI(title="Mushroom Vision API")
model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)
    return {"class": class_name, "confidence": confidence}
```

Alongside the code lives a **requirements file** listing every Python package the app needs:

```
fastapi==0.115.6
uvicorn==0.34.0
ultralytics==8.3.83
torch==2.6.0
Pillow>=10.2.0
prometheus-fastapi-instrumentator==7.0.0
```

### Why it exists

This is the thing that actually does the work. Everything else in this document exists to **package, ship, and run** this code reliably.

### How it connects to the next step

Your code runs perfectly on your laptop because you installed the right Python version, the right packages, and you have the model weights in the right folder. The problem is: **no one else's machine looks like yours**. The Dockerfile solves that.

```
┌──────────────────────────────────┐
│         YOUR LAPTOP              │
│                                  │
│  main.py  +  requirements.txt    │
│  model weights (best.pt)         │
│  Python 3.12, pip, libraries...  │
│                                  │
│  $ uvicorn main:app --port 8000  │
│         ✓ Works here             │
│         ✗ Works nowhere else     │
└──────────────────────────────────┘
                 │
                 │  "How do I make this portable?"
                 ▼
          [ Dockerfile ]
```

---

## 2. Dockerfile — Wrapping the App

> 📁 **Project files:**
>
> - `services/vision_api/Dockerfile` — Builds the Vision API image
> - `services/brain_ui/Dockerfile` — Builds the Brain UI image

### What it is

A **Dockerfile** is a plain-text recipe that describes, step by step, how to build a self-contained environment for your application. Think of it as an _instruction sheet inside a flat-pack furniture box_ — anyone with the sheet and the parts can assemble the exact same thing.

Here is the actual Dockerfile from the Mushroom Vision API:

```dockerfile
# Start from a lightweight Python 3.12 environment
FROM python:3.12-slim

# Create a working folder inside the container
WORKDIR /app

# Install system-level libraries YOLO needs
RUN apt-get update && apt-get install -y \
    gcc libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY services/vision_api/requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy the trained model weights
COPY docs/yolo_runs/yolo26_classifier_v1/weights/best.pt \
     /app/docs/yolo_runs/yolo26_classifier_v1/weights/best.pt

# Copy the application code
COPY services/vision_api/main.py /app/services/vision_api/main.py

# Tell Docker which port the app listens on
EXPOSE 8000

# The command that starts the server
CMD ["uvicorn", "services.vision_api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
```

### Why it exists

Without a Dockerfile, deploying means saying _"install Python 3.12, then pip install these 11 packages, then download the model to exactly this path, then run this command..."_ — and hoping nothing goes wrong. The Dockerfile **encodes all of that** into a single, repeatable file.

| Problem                  | Dockerfile solution                   |
| ------------------------ | ------------------------------------- |
| "It works on my machine" | The Dockerfile **is** the machine     |
| Forgetting a dependency  | Every dependency is listed explicitly |
| Wrong Python version     | `FROM python:3.12-slim` pins it       |
| Model file not found     | `COPY` places it at the exact path    |

### How it connects

The Dockerfile takes your local code and produces a **Docker image** — the next layer.

```
┌──────────────────┐         ┌──────────────────┐
│    Local Code     │         │   Docker Image    │
│                   │  build  │                   │
│  main.py          ├────────►│  Frozen snapshot  │
│  requirements.txt │         │  of your app +    │
│  best.pt          │         │  OS + libraries   │
└──────────────────┘         └──────────────────┘
         ▲                            │
         │                            │
    [ Dockerfile ]               Ready to run
    (the recipe)                 anywhere
```

---

## 3. Docker Image — The Built Artifact

> 📁 **Project files:** None — the image is a binary artifact _built from_ the Dockerfiles above. Not stored in the repo.

### What it is

A **Docker image** is the output of running `docker build`. It's a frozen, read-only snapshot containing your app code, the Python runtime, all installed packages, the model weights, and even the Linux OS libraries — everything packed into one portable file.

The analogy: if the Dockerfile is a recipe, the Docker image is the **sealed, frozen meal** produced from it. You can ship it anywhere, and when someone heats it up (runs it), they get the exact same dish.

```
$ docker build -t mushroom-vision-api -f services/vision_api/Dockerfile .
```

After this command finishes, the image `mushroom-vision-api` exists on your machine, typically a few GB in size (because of PyTorch and YOLO).

### Why it exists

Images solve the **distribution** problem. You don't ship source code to a server and pray it compiles — you build the image once, and the exact same binary blob runs identically on:

- Your laptop
- Your teammate's Mac
- A CI server in GitHub's data center
- A Google Cloud machine in Iowa

### How it connects

An image is inert — it's just a file sitting on disk. To actually serve traffic, you **run** it, which creates a container.

```
                 docker build
[ Dockerfile ] ──────────────► [ Docker Image ]
                                (mushroom-vision-api:latest)
                                      │
                                      │  docker run
                                      ▼
                               [ Docker Container ]
                               (a live, running process)
```

---

## 4. Docker Container — Running the Image

> 📁 **Project files:**
>
> - `deploy/docker-compose.yml` — Defines and orchestrates all local containers
> - `prometheus.yml` — Prometheus scrape config (mounted into the Prometheus container)

### What it is

A **Docker container** is a running instance of an image. When you execute `docker run`, Docker takes the frozen image, creates an isolated process from it, and starts the command defined by `CMD` in the Dockerfile. The container is a lightweight, sandboxed mini-computer running inside your real computer.

```
$ docker run -p 8000:8000 mushroom-vision-api
```

Now `http://localhost:8000/predict` is live and accepting mushroom photos — running inside the container, not directly on your OS.

### Why it exists

Containers provide **isolation**. Your app runs in its own little world:

- It has its own filesystem (won't conflict with other apps)
- It has its own network port mapping (`-p 8000:8000`)
- It can be stopped and destroyed without leaving traces
- You can run multiple containers side by side

In the Mushroom project, **Docker Compose** orchestrates multiple containers at once — the Vision API, the Brain UI, Prometheus, and Grafana — all on a shared virtual network:

```
┌─── docker-compose up ───────────────────────────────────┐
│                                                         │
│  ┌─────────────┐    ┌─────────────┐   ┌────────────┐    │
│  │ Vision API  │◄───│  Brain UI   │   │ Prometheus │    │
│  │  :8000      │    │  :7860      │   │  :9090     │    │
│  └─────────────┘    └─────────────┘   └────────────┘    │
│        ▲                                     │          │
│        │            ┌─────────────┐          │          │
│        └────────────│   Grafana   │◄─────────┘          │
│                     │   :3000     │                     │
│                     └─────────────┘                     │
│                                                         │
│              [ mushroom-net (virtual network) ]         │
└─────────────────────────────────────────────────────────┘
```

### How it connects

Running containers locally proves everything works. But you can't keep your laptop open 24/7 to serve users. The code, Dockerfile, and configs need to live somewhere permanent and trigger automated builds. That's where GitHub comes in.

```
[ Docker Container ] ── "This works locally, now automate it."
         │
         ▼
[ GitHub Repository ]
```

---

## 5. GitHub Repository — Storing Everything

> 📁 **Project files:** The entire `Mushroom/` directory _is_ the repository.

### What it is

A **GitHub repository** is a remote, versioned vault for your entire project. It stores your source code, Dockerfiles, YAML configs, model training scripts, documentation — everything needed to reproduce the project from scratch.

For the Mushroom project, the repository structure looks like:

```
Mushroom/
├── services/
│   ├── vision_api/
│   │   ├── main.py              ← The actual app
│   │   ├── Dockerfile           ← How to build it
│   │   ├── cloudbuild.yaml      ← How Google Cloud builds it
│   │   └── requirements.txt     ← Python dependencies
│   └── brain_ui/
│       ├── app.py
│       └── Dockerfile
├── deploy/
│   └── docker-compose.yml       ← Local multi-container setup
├── .github/
│   └── workflows/
│       ├── deploy.yml           ← CI/CD pipeline to Cloud Run
│       └── cml.yml              ← Auto-generated model reports
└── docs/
    └── yolo_runs/               ← Trained model artifacts
```

### Why it exists

GitHub solves three problems at once:

1. **Version control** — Every change is recorded. You can roll back to any prior state.
2. **Collaboration** — Multiple people can work on the code safely.
3. **Automation trigger** — GitHub can detect events (like a push to `master`) and kick off automated workflows. This is the bridge to CI/CD.

### How it connects

The repository doesn't just store code — it **watches for changes**. When you push to the `master` branch, GitHub Actions workflows wake up and start running.

```
                      git push origin master
[ Your Laptop ] ──────────────────────────────► [ GitHub Repo ]
                                                      │
                                                      │ "New code on master!"
                                                      │ (event trigger)
                                                      ▼
                                              [ GitHub Actions ]
                                              (reads .github/workflows/*.yml)
```

---

## 6. GitHub Actions — The Automation Engine

> 📁 **Project files:**
>
> - `.github/workflows/deploy.yml` — CI/CD pipeline: builds images and deploys to Cloud Run on push to `master`
> - `.github/workflows/cml.yml` — ML reporting: auto-generates a model evaluation report on pull requests

### What it is

**GitHub Actions** is a built-in CI/CD (Continuous Integration / Continuous Deployment) system. It reads YAML instruction files from your `.github/workflows/` directory and executes them on fresh virtual machines whenever a specified event occurs.

Think of it as a **robot assistant** that lives inside GitHub. You leave it a to-do list (the YAML file), and every time you push code, it spins up a clean computer, follows the instructions, and reports back.

Here is the actual deployment workflow from the Mushroom project:

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy to Cloud Run

on:
  push:
    branches:
      - master # Only runs when code lands on master

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
      # Step 1: Download the repo code onto the CI machine
      - name: Checkout repository
        uses: actions/checkout@v3

      # Step 2: Authenticate with Google Cloud
      - name: Auth via Google Credentials
        uses: google-github-actions/auth@v1
        with:
          credentials_json: "${{ secrets.GCP_CREDENTIALS }}"

      # Step 3: Build the Vision API Docker image (on Google Cloud Build)
      - name: Build Vision API Image
        run: gcloud builds submit --config services/vision_api/cloudbuild.yaml .

      # Step 4: Deploy the image to Cloud Run
      - name: Deploy Vision API to Cloud Run
        run: |
          gcloud run deploy vision-api \
            --image us-central1-docker.pkg.dev/.../vision-api:latest \
            --region us-central1 \
            --platform managed \
            --allow-unauthenticated \
            --port 8000 \
            --memory 4Gi \
            --cpu 1
```

### Why it exists

Without CI/CD, deploying means manually SSHing into a server, pulling code, building images, and restarting services — every single time. That's slow, error-prone, and doesn't scale.

GitHub Actions automates the entire sequence:

| Manual step           | Automated by GitHub Actions     |
| --------------------- | ------------------------------- |
| Pull latest code      | `actions/checkout@v3`           |
| Log into Google Cloud | `google-github-actions/auth@v1` |
| Build Docker image    | `gcloud builds submit`          |
| Deploy to Cloud Run   | `gcloud run deploy`             |

### How it connects

GitHub Actions is the **bridge** between your repository and Google Cloud. It reads YAML files for instructions, authenticates with cloud credentials stored as **secrets**, and executes cloud commands on your behalf.

```
┌─────────────────────────────────────────────────────┐
│                  GitHub Actions                     │
│                                                     │
│  1. Spin up fresh Ubuntu VM                         │
│  2. Clone the repo                                  │
│  3. Authenticate with GCP (using stored secret)     │
│  4. Run: gcloud builds submit (builds Docker image) │
│  5. Run: gcloud run deploy   (deploys the image)    │
│  6. Tear down the VM                                │
│                                                     │
└─────────────────────────────────────────────────────┘
         ▲                              │
         │                              │
  [ YAML workflow file ]         [ Google Cloud ]
  (the instruction set)          (receives + runs
                                  the image)
```

---

## 7. YAML Files — The Language of Pipelines

> 📁 **Project files (all YAML):**
>
> - `deploy/docker-compose.yml` — Local multi-container orchestration
> - `services/vision_api/cloudbuild.yaml` — Cloud Build instructions for the Vision API
> - `services/brain_ui/cloudbuild.yaml` — Cloud Build instructions for the Brain UI
> - `.github/workflows/deploy.yml` — GitHub Actions deployment workflow
> - `.github/workflows/cml.yml` — GitHub Actions ML reporting workflow
> - `data/dataset.yaml` — Dataset configuration for YOLO training
> - `dvc.yaml` — DVC pipeline definition for data versioning
> - `prometheus.yml` — Prometheus monitoring scrape targets

### What it is

**YAML** (YAML Ain't Markup Language) is a human-readable data format used to write configuration files. It's the standard language for Docker Compose, GitHub Actions, Google Cloud Build, Kubernetes, and nearly every modern DevOps tool.

YAML uses **indentation** (spaces, not tabs) to represent structure — no curly braces, no angle brackets, just clean key-value pairs.

### Why it's powerful

YAML files are **declarative** — you describe _what you want_, not _how to do it_. The tool reading the YAML figures out the execution details.

Three YAML files drive the entire Mushroom deployment:

#### A. `docker-compose.yml` — Local orchestration

Tells Docker: "run these four containers on a shared network."

```yaml
services:
  visionapi:
    build:
      context: ..
      dockerfile: services/vision_api/Dockerfile
    ports:
      - "8000:8000"
    networks:
      - mushroom-net

  brainui:
    build:
      context: ..
      dockerfile: services/brain_ui/Dockerfile
    ports:
      - "7860:7860"
    depends_on:
      - visionapi # Start Vision API first
    networks:
      - mushroom-net
```

#### B. `cloudbuild.yaml` — Google Cloud Build instructions

Tells Google Cloud Build: "build this Dockerfile, tag the image, push it to Artifact Registry."

```yaml
steps:
  - name: "gcr.io/cloud-builders/docker"
    args:
      - "build"
      - "-t"
      - "us-central1-docker.pkg.dev/.../vision-api:latest"
      - "-f"
      - "services/vision_api/Dockerfile"
      - "."

images:
  - "us-central1-docker.pkg.dev/.../vision-api:latest"
```

#### C. `deploy.yml` — GitHub Actions workflow

Tells GitHub: "when master updates, authenticate with GCP and run these shell commands."

```yaml
on:
  push:
    branches: [master]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: google-github-actions/auth@v1
        with:
          credentials_json: "${{ secrets.GCP_CREDENTIALS }}"
      - run: gcloud builds submit --config services/vision_api/cloudbuild.yaml .
      - run: gcloud run deploy vision-api --image ... --region us-central1
```

### YAML anatomy cheat sheet

```yaml
key: value # Simple string
port: 8000 # Number
enabled: true # Boolean

list_of_items: # A list (array)
  - item_one
  - item_two

nested_object: # Nested structure
  child_key: child_value
  another_key: 42

multiline: | # Multi-line string (preserves newlines)
  gcloud run deploy \
    --image my-image \
    --region us-central1
```

### How it connects

Every tool in the pipeline reads a different YAML file, but they all follow the same format. YAML is the **common language** that lets your local setup, CI system, and cloud provider all understand the same instructions.

```
┌──────────────────────────────────────────────────────┐
│                   YAML Files                          │
│                                                       │
│  docker-compose.yml ──► Docker (local orchestration) │
│  cloudbuild.yaml    ──► Cloud Build (image building) │
│  deploy.yml         ──► GitHub Actions (CI/CD)       │
│  cml.yml            ──► GitHub Actions (ML reports)  │
│                                                       │
│  Same format. Different readers. One pipeline.       │
└──────────────────────────────────────────────────────┘
```

---

## 8. Google Cloud — Where It All Runs

> 📁 **Project files:**
>
> - `services/vision_api/cloudbuild.yaml` — Tells Cloud Build how to build the Vision API image
> - `services/brain_ui/cloudbuild.yaml` — Tells Cloud Build how to build the Brain UI image
> - `.github/workflows/deploy.yml` — Contains the `gcloud run deploy` commands that target Cloud Run

### What it is

**Google Cloud Platform (GCP)** is a set of cloud computing services that provides the machines, networking, and infrastructure to run your app 24/7 without owning a server.

The Mushroom project uses three GCP services:

| Service               | Role                                                                           |
| --------------------- | ------------------------------------------------------------------------------ |
| **Cloud Build**       | Builds Docker images on Google's servers (so you don't need a powerful laptop) |
| **Artifact Registry** | Stores built Docker images (like a private Docker Hub)                         |
| **Cloud Run**         | Runs containers on-demand, scaling from zero to many instances automatically   |

### How each service fits

```
┌────────────────────────── Google Cloud Platform ──────────────────────────┐
│                                                                           │
│   ┌──────────────┐       ┌────────────────────┐      ┌───────────────┐  │
│   │ Cloud Build  │       │ Artifact Registry   │      │   Cloud Run   │  │
│   │              │ push  │                     │ pull │               │  │
│   │ Reads your   ├──────►│ Stores the built    ├─────►│ Runs the      │  │
│   │ Dockerfile   │       │ Docker image        │      │ container     │  │
│   │ + source code│       │ (vision-api:latest) │      │ and serves    │  │
│   │              │       │                     │      │ HTTPS traffic │  │
│   └──────────────┘       └────────────────────┘      └───────┬───────┘  │
│                                                               │          │
│                                                               ▼          │
│                                                     https://vision-api   │
│                                                     ...run.app/predict   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Why Cloud Run specifically

Cloud Run is a **serverless** container platform. You give it a Docker image and it handles everything else:

- **Automatic HTTPS** — No SSL certificate management
- **Scale to zero** — When nobody's using the app, you pay nothing
- **Scale up automatically** — If 1,000 users hit it at once, Cloud Run spins up more instances
- **No server management** — No patching, no SSH, no uptime monitoring

The deploy command tells Cloud Run exactly what to run:

```bash
gcloud run deploy vision-api \
  --image us-central1-docker.pkg.dev/.../vision-api:latest \
  --region us-central1 \        # Physical location of the server
  --platform managed \          # Let Google manage the infrastructure
  --allow-unauthenticated \     # Public access (no login required)
  --port 8000 \                 # Which port the container listens on
  --memory 4Gi \                # RAM allocation (YOLO needs room)
  --cpu 1                       # CPU allocation
```

### How it connects

Cloud Run is the **final destination**. Once deployed, your app has a public URL and is serving real users. But it all started with a `git push`.

```
[ git push ] ──► [ GitHub Actions ] ──► [ Cloud Build ] ──► [ Cloud Run ]
                                                                   │
                                                                   ▼
                                                          🌐 Live Application
                                                https://...run.app/predict
```

---

## 9. The Full Pipeline — End to End

> 📁 **Every project file in the pipeline, mapped to its role:**
>
> | File                                   | Role in pipeline                        |
> | -------------------------------------- | --------------------------------------- |
> | `services/vision_api/main.py`          | The application code (FastAPI server)   |
> | `services/brain_ui/app.py`             | The front-end code (Gradio UI)          |
> | `services/vision_api/requirements.txt` | Python deps for Vision API              |
> | `services/brain_ui/requirements.txt`   | Python deps for Brain UI                |
> | `services/vision_api/Dockerfile`       | Packages Vision API into an image       |
> | `services/brain_ui/Dockerfile`         | Packages Brain UI into an image         |
> | `deploy/docker-compose.yml`            | Runs all containers locally             |
> | `services/vision_api/cloudbuild.yaml`  | Cloud Build recipe for Vision API       |
> | `services/brain_ui/cloudbuild.yaml`    | Cloud Build recipe for Brain UI         |
> | `.github/workflows/deploy.yml`         | CI/CD: build + deploy on push to master |
> | `.github/workflows/cml.yml`            | CI: model report on pull requests       |
> | `prometheus.yml`                       | Monitoring config (local + container)   |
> | `docs/yolo_runs/.../weights/best.pt`   | Trained model weights                   |

Here is the complete journey from your editor to a live application, as it works in this project:

```
 YOU (Developer)
  │
  │  Write / edit code
  ▼
┌─────────────────────────────────┐
│  LOCAL CODE                     │
│  main.py, app.py, best.pt      │
│  requirements.txt               │
│  Dockerfile                     │
└────────────────┬────────────────┘
                 │
                 │  git add, git commit, git push origin master
                 ▼
┌─────────────────────────────────┐
│  GITHUB REPOSITORY              │
│  Stores code, configs, history  │
│  Detects push to master branch  │
└────────────────┬────────────────┘
                 │
                 │  Event trigger (on: push)
                 ▼
┌─────────────────────────────────┐
│  GITHUB ACTIONS                 │
│  Reads .github/workflows/       │
│  deploy.yml                     │
│                                  │
│  1. Checks out code             │
│  2. Authenticates with GCP      │
│  3. Submits build to Cloud Build│
│  4. Deploys to Cloud Run        │
└────────────────┬────────────────┘
                 │
                 │  gcloud builds submit
                 ▼
┌─────────────────────────────────┐
│  GOOGLE CLOUD BUILD             │
│  Reads cloudbuild.yaml          │
│  Reads Dockerfile               │
│  Builds Docker image on GCP     │
│  Pushes image to Artifact Reg.  │
└────────────────┬────────────────┘
                 │
                 │  Image stored in registry
                 ▼
┌─────────────────────────────────┐
│  ARTIFACT REGISTRY              │
│  us-central1-docker.pkg.dev/    │
│  .../vision-api:latest          │
│  .../brain-ui:latest            │
└────────────────┬────────────────┘
                 │
                 │  gcloud run deploy --image ...
                 ▼
┌─────────────────────────────────┐
│  CLOUD RUN                      │
│  Pulls image from registry      │
│  Starts container               │
│  Assigns HTTPS URL              │
│  Auto-scales based on traffic   │
└────────────────┬────────────────┘
                 │
                 ▼
        🌐 LIVE APPLICATION
   https://vision-api-....run.app
   https://brain-ui-....run.app
```

### The whole thing in one sentence

> You push code to GitHub, GitHub Actions authenticates with Google Cloud, Cloud Build turns your Dockerfile into an image, and Cloud Run serves that image to the world — automatically, every time.

### What triggers what

| Event                    | System            | Action                                     |
| ------------------------ | ----------------- | ------------------------------------------ |
| `git push` to `master`   | GitHub            | Triggers Actions workflow                  |
| `deploy.yml` runs        | GitHub Actions    | Authenticates + calls `gcloud`             |
| `gcloud builds submit`   | Cloud Build       | Reads `cloudbuild.yaml`, builds image      |
| Image pushed to registry | Artifact Registry | Stores the versioned image                 |
| `gcloud run deploy`      | Cloud Run         | Pulls image, starts container, assigns URL |
| HTTP request hits URL    | Cloud Run         | Routes to container, auto-scales           |

### Secrets and credentials

The pipeline needs to authenticate with Google Cloud without exposing passwords in code. This is handled by **GitHub Secrets** — encrypted variables stored in the repository settings:

```
GitHub Repo Settings → Secrets → Actions
  ├── GCP_CREDENTIALS     ← Service account JSON key
  └── GEMINI_API_KEY      ← API key for the Brain UI's Gemini integration
```

These are referenced in YAML as `${{ secrets.GCP_CREDENTIALS }}` and are **never** visible in logs or code.

---

## Quick Reference: Commands You Actually Run

| What                     | Command                                                          |
| ------------------------ | ---------------------------------------------------------------- |
| Build image locally      | `docker build -t vision-api -f services/vision_api/Dockerfile .` |
| Run container locally    | `docker run -p 8000:8000 vision-api`                             |
| Run all services locally | `cd deploy && docker-compose up --build`                         |
| Deploy everything        | `git push origin master` (Actions handles the rest)              |
| Check Cloud Run status   | `gcloud run services describe vision-api --region us-central1`   |
| View container logs      | `gcloud run services logs read vision-api --region us-central1`  |

---

_Generated from the Mushroom Vision API project — a YOLO-based mushroom classification system deployed on Google Cloud Run._
