# Mushroom Safety Classification System

A multimodal AI safety system that identifies mushroom species visually and cross-references them with ecological context using an LLM audit layer.

## Quick Start

```bash
# 1. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py                              # Demo mode (uses a test image)
python main.py path/to/mushroom_photo.jpg   # Analyze your own image
```

## How It Works

```
Image → YOLO (Vision) → CSV (Knowledge) → LLM (Reasoning) → Risk Engine (Safety) → Report
```

1. **YOLOv8 Nano** identifies the mushroom species from a photo
2. **Pandas** looks up the species in `mushroom_context.csv` for toxicity, habitat, and season data
3. **Ollama (Llama3)** audits whether the identification makes sense given the user's location and season
4. **Risk Engine** applies hard-coded safety rules that _cannot_ be overridden by the LLM

## Project Structure

```
my_project_plan/
├── main.py                     ← Single entry point
├── data/
│   ├── dataset_split/          ← 169 species (80/10/10 train/val/test)
│   ├── mushroom_context.csv    ← Ecological rules for all 169 species
│   ├── dataset.yaml            ← YOLO configuration
│   └── mushroom_species.json   ← Master species list
├── scripts/
│   ├── setup/                  ← One-time data preparation scripts
│   ├── training/               ← YOLO model training (train_yolo.py)
│   └── pipeline/               ← Active system modules
│       ├── predict.py          → Vision prediction
│       ├── integration.py      → CSV lookup
│       ├── audit_layer.py      → LLM reasoning
│       ├── llm_provider.py     → Ollama/Gemini provider swap
│       └── risk_engine.py      → Safety decision rules
└── docs/
    └── yolo_runs/              ← Training charts, loss graphs, model weights
```

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset Preparation (169 species, 104k images) | ✅ Complete |
| 2 | YOLOv8 Classification Training (CUDA GPU) | ✅ Complete |
| 3 | Multimodal Context Integration (CSV + Pandas) | ✅ Complete |
| 4 | LLM Audit Layer (Ollama/Llama3) | ✅ Complete |
| 5 | Risk-Aware Decision Logic (4 safety rules) | ✅ Complete |
| 6 | End-to-End Pipeline Integration | ✅ Complete |

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
