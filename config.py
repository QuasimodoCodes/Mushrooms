"""
config.py — Runtime switches for the Mushroom Safety System.

This is the only file you ever need to touch to change how the app behaves.
Training hyperparameters live inside each training script — they're not here
because you only set those once before a run, not day-to-day.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
#  VISION MODEL
#  Change YOLO_RUN_NAME to swap which trained model the API serves.
#  The name must match a folder inside docs/yolo_runs/.
# ─────────────────────────────────────────────────────────────────────────────

YOLO_RUN_NAME = "yolo26_tflite"
# Other options already trained:  "mushroom_classifier_v1"  (YOLOv8n baseline)

# "pt"     → full PyTorch weights  (~1.5 GB Docker image, requires torch)
# "tflite" → exported TFLite model (~200 MB Docker image, no PyTorch needed)
MODEL_FORMAT = os.environ.get("MODEL_FORMAT", "pt").lower()

# Resolved automatically from the two settings above — do not edit these.
_weights = os.path.join(PROJECT_ROOT, "docs", "yolo_runs", YOLO_RUN_NAME, "weights")
MODEL_PATH = (os.path.join(_weights, "best_float16.tflite")
              if MODEL_FORMAT == "tflite"
              else os.path.join(_weights, "best.pt"))

# ─────────────────────────────────────────────────────────────────────────────
#  LLM PROVIDER
#  Set ACTIVE_LLM_PROVIDER to switch between local and cloud inference.
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_LLM_PROVIDER = "gemini"   # "ollama"  →  local via Ollama (any model below)
                                  # "gemini"  →  Google cloud (needs API key)

# Ollama — change OLLAMA_MODEL to swap models.
# Multimodal models (e.g. gemma4:e4b) will automatically receive the image too.
# Text-only models (e.g. llama3) will just get the text prompt.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")  # options: "llama3:latest", "gemma4:e2b", "gemma3:12b"
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

# Gemini — API key read from the GEMINI_API_KEY environment variable 
# Models "gemini-3-flash-preview" "gemma-4-26b-a4b-it"
GEMINI_MODEL   = "gemini-3-flash-preview"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
#  SAFETY THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

# Model confidence below this → risk escalates to HIGH.  Range: 0.0 – 1.0
CONFIDENCE_THRESHOLD     = 0.70
DRIFT_CONFIDENCE_THRESHOLD = 0.70   # Images below this are saved for retraining

# CSV toxicity containing any of these words → instant CRITICAL, no exceptions.
DEADLY_KEYWORDS = ["deadly", "fatal", "death", "highly toxic"]

# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL — derived paths used by the pipeline (no need to edit)
# ─────────────────────────────────────────────────────────────────────────────

VISION_API_URL   = os.environ.get("VISION_API_URL", "http://127.0.0.1:8000/predict")
CONTEXT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "mushroom_context.json")
DRIFT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "drift_images")
PROMETHEUS_PORT  = 8001
