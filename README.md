# Mushroom Safety Classification System

A multimodal AI safety system that identifies mushroom species visually and cross-references them with ecological context using an LLM audit layer.

## Clone & Run (For Friends / New Users)

### Prerequisites

- **Python 3.10+** installed
- **Ollama** installed ([download here](https://ollama.com)) with the Llama3 model:
  ```bash
  ollama pull llama3
  ```

### Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd my_project_plan

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

### Run

```bash
# Option A: Command line
python main.py path/to/your/mushroom_photo.jpg

# Option B: Web UI (drag & drop)
python app.py
# Then open http://localhost:7860 in your browser
```

> **Note:** Demo mode (`python main.py` with no image) requires the full dataset which is not included in the repo due to its size (104k images). Pass your own image instead.

## How It Works

```
Image → YOLO (Vision) → CSV (Knowledge) → LLM (Reasoning) → Risk Engine (Safety) → Report
```

1. **YOLOv8 Nano** identifies the mushroom species from a photo
2. **Pandas** looks up the species in `mushroom_context.csv` for toxicity, habitat, and season data
3. **Ollama (Llama3)** audits whether the identification makes sense given the user's location and season
4. **Risk Engine** applies hard-coded safety rules that _cannot_ be overridden by the LLM

## Project Structure

```text
Mushroom/
├── app.py                      ← Web UI entry point
├── main.py                     ← CLI entry point
├── plan.json                   ← Project planning data
├── plans.md                    ← Project plans documentation
├── README.md                   ← Project documentation
├── requirements.txt            ← Python dependencies
├── data/
│   ├── dataset_split/          ← 169 species (80/10/10 train/val/test)
│   ├── dataset.yaml            ← YOLO configuration
│   ├── mushroom_context.csv    ← Ecological rules for all 169 species
│   ├── mushroom_species.json   ← Master species list
│   ├── test.csv                ← Test dataset annotations
│   ├── train.csv               ← Train dataset annotations
│   └── val.csv                 ← Validation dataset annotations
├── docs/
│   ├── features_backlog.md     ← Planned features
│   ├── problems_log.md         ← Known issues and logs
│   └── yolo_runs/              ← Training charts, loss graphs, model weights
└── scripts/
    ├── pipeline/               ← Active system modules
    │   ├── audit_layer.py      → LLM reasoning
    │   ├── integration.py      → CSV lookup
    │   ├── llm_provider.py     → Ollama/Gemini provider swap
    │   ├── predict.py          → Vision prediction
    │   └── risk_engine.py      → Safety decision rules
    ├── setup/                  ← One-time data preparation scripts
    │   ├── fix_dataset.py      → Dataset formatting fixes
    │   └── prepare_dataset.py  → Initial data preparation
    └── training/               ← YOLO model training
        ├── train_yolo.py       → YOLO training script
        └── yolov8n-cls.pt      → Pre-trained YOLO model
```

## Development Phases

| Phase | Description                                    | Status      |
| ----- | ---------------------------------------------- | ----------- |
| 1     | Dataset Preparation (169 species, 104k images) | ✅ Complete |
| 2     | YOLOv8 Classification Training (CUDA GPU)      | ✅ Complete |
| 3     | Multimodal Context Integration (CSV + Pandas)  | ✅ Complete |
| 4     | LLM Audit Layer (Ollama/Llama3)                | ✅ Complete |
| 5     | Risk-Aware Decision Logic (4 safety rules)     | ✅ Complete |
| 6     | End-to-End Pipeline Integration                | ✅ Complete |

## Training Configuration

- **Model**: YOLOv8n-cls (Nano, 1.65M parameters)
- **GPU**: NVIDIA GeForce RTX 3070 Ti (CUDA 12.1)
- **Epochs**: 50 (with early stopping, patience=10)
- **Local Optima Defense**: Cosine LR scheduling
- **Image Size**: 224×224

## Switching LLM Provider

Edit `scripts/pipeline/llm_provider.py`:

```python
ACTIVE_PROVIDER = "gemini"     # Change from "ollama" to "gemini"
GEMINI_API_KEY = "your-key"    # Set your Google AI API key
```
