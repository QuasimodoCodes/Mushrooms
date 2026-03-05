# Mushroom Safety Classification System

This repository tracks the development of a multimodal AI safety system designed to identify mushroom species visually and cross-reference them with regional and seasonal ecological context using an LLM audit layer.

## Project Structure

*   `data/`: Contains the split image dataset, CSV metadata, and the `dataset.yaml` configuration.
*   `scripts/`: Contains Python scripts for executing various phases of the pipeline.
*   `docs/`: Contains logs, features backlogs, and YOLO training run outputs.
*   `plan.json`: The living task list driving the development of this project.

## Development Phases

### ✅ Phase 1: Dataset Preparation and Preprocessing
We downloaded the `zlatan599/mushroom1` dataset containing 169 distinct species. The dataset was preprocessed into an 80/10/10 split specifically formatted for YOLO classification.

**Scripts Run:**
*   `prepare_dataset.py` / `fix_dataset.py`: Handled downloading, class extraction, and train/val/test splitting.

### 🟡 Phase 2: Classification Model Training
We are setting up YOLOv8 Nano (`yolov8n-cls.pt`) to act as our primary vision model for species identification. We are utilizing PyTorch with CUDA support to leverage GPU acceleration.

**Scripts to Run:**
*   `scripts/train_yolo.py`: Trains the classification model. Note that it requires an Nvidia GPU for reasonable training times.

### 🔴 Phase 3: Multimodal Context Integration (Pending)
### 🔴 Phase 4: LLM Audit Layer Development (Pending)
### 🔴 Phase 5: Risk-Aware Decision Logic Implementation (Pending)
### 🔴 Phase 6: End-to-End System Integration (Pending)

## Setup Instructions
1. Create a Python Virtual Environment.
2. Activate the environment.
3. Install dependencies: `pip install -r requirements.txt`
