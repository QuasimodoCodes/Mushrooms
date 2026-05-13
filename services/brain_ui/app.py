"""
Mushroom Classifier - Gradio Web UI
============================================
A simple drag-and-drop interface for the mushroom pipeline.
Run this script and open the URL in your browser.

Usage:
  python app.py
"""

import gradio as gr
import sys
import os
import logging
import shutil
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
import subprocess
from datetime import datetime

# Load central config
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from config import DRIFT_CONFIDENCE_THRESHOLD, PROMETHEUS_PORT, CONTEXT_CSV_PATH, ACTIVE_LLM_PROVIDER


# Configure centralized cloud-compatible logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the pipeline directory to Python's path
pipeline_dir = os.path.join(os.path.dirname(__file__), "pipeline")
sys.path.insert(0, pipeline_dir)

import prometheus_client
from prometheus_client import Counter

# Define a Prometheus Counter for our MLOps drift metric
DRIFT_EVENTS = Counter('mushroom_drift_events_total', 'Total number of classification events resulting in low confidence (drift)', ['species'])

# Start the Prometheus metrics server
prometheus_client.start_http_server(PROMETHEUS_PORT)

from predict import predict_image
from integration import get_mushroom_context
from audit_layer import audit_prediction
from risk_engine import assess_risk


def log_drift_image(image_path, confidence, predicted_species):
    """
    MLOps Drift Detection:
    If the model's confidence is very low, we save the image to a 'drift_images' folder.
    This allows data scientists to manually review and retrain the model later.
    """
    if confidence >= DRIFT_CONFIDENCE_THRESHOLD:
        return # Skip, confidence is high enough
        
    logger.warning(f"[DRIFT DETECTED] Low confidence ({confidence:.2f}) for species '{predicted_species}'. Saving image for manual review.")
    
    # Try to resolve path whether we are locally testing or in Docker
    base_dir = os.path.dirname(__file__)
    drift_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "data", "drift_images")
    if not os.path.exists(drift_dir):
        # Fallback to local data dir if in Docker
        drift_dir = os.path.join(base_dir, "data", "drift_images")
        
    os.makedirs(drift_dir, exist_ok=True)
    
    # Create a unique filename with timestamp and prediction
    timestamp = datetime.now().strftime("%Y%md_%H%M%S")
    safe_species = predicted_species.replace(" ", "_").replace("/", "-")
    file_name = f"drift_{timestamp}_{safe_species}_conf{int(confidence*100)}.jpg"
    
    dest_path = os.path.join(drift_dir, file_name)
    try:
        shutil.copy2(image_path, dest_path)
        # Increment Prometheus Drift Metric explicitly
        DRIFT_EVENTS.labels(species=safe_species).inc()
    except Exception as e:
        logger.error(f"Failed to log drift image: {e}")

def classify_mushroom(image, season, location, progress=None):
    """
    Main function called by Gradio when the user submits an image.
    """
    if image is None:
        yield "Please upload an image first."
        return

    if progress: progress(0.0, desc="🍄 Initializing...")
    yield "### 🍄 Initializing classification process..."

    if progress: progress(0.1, desc="🍄 Locating Context Database...")
    base_dir = os.path.dirname(__file__)
    # Prefer JSON (enriched), fall back to CSV
    for filename in ("mushroom_context.json", "mushroom_context.csv"):
        if os.path.exists(os.path.join(base_dir, "data", filename)):
            csv_path = os.path.join(base_dir, "data", filename)
            break
        elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(base_dir)), "data", filename)):
            csv_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "data", filename)
            break

    yield "### 🍄 Uploading image to Vision API for identification..."
    # Step 1: YOLO Vision (via Vision API)
    if progress: progress(0.2, desc="🍄 Uploading image to Vision API...")
    predicted_species, confidence = predict_image(image)
    
    if predicted_species is None:
        yield "❌ Error: Could not connect to the Vision API. Please ensure it is running in another terminal."
        return
        
    # Format the species name nicely if it has underscores
    formatted_species = predicted_species.replace("_", " ").title() if predicted_species else "Unknown"

    yield f"### 🍄 Identified as **{formatted_species}**. Checking data drift..."
    # Step 1.5: Trigger Drift Detection Logic (MLOps)
    # Background save if confidence is too low
    if progress: progress(0.4, desc="🍄 Checking classification confidence & data drift...")
    log_drift_image(image, confidence, predicted_species)

    yield f"### 🍄 Fetching ecological context for **{formatted_species}**..."
    if progress: progress(0.6, desc="🍄 Fetching ecological context from Knowledge Base...")
    context = get_mushroom_context(formatted_species, csv_path)
    if "error" in context:
        context = {
            "toxicity_type": "Unknown", "habitat": "Unknown",
            "season": "Unknown", "region": "Unknown",
            "key_warnings": "Species not found in database. Treat as potentially dangerous."
        }
    
    # Step 3: LLM Audit
    yield f"### 🍄 Requesting LLM Audit for **{formatted_species}**..."
    if progress: progress(0.7, desc="🍄 Requesting LLM Audit (Llama3/Gemini)...")
    llm_verdict = audit_prediction(formatted_species, confidence, context, season, location, image_path=image)
    
    # Step 4: Risk Decision
    yield "### 🍄 Calculating final risk level..."
    if progress: progress(0.9, desc="🍄 Calculating final risk level...")
    decision = assess_risk(formatted_species, confidence, context, llm_verdict)
    
    yield "### 🍄 Generating Report..."
    if progress: progress(1.0, desc="🍄 Generating Report...")
    # Build the output report
    risk_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MODERATE": "⚠️", "LOW": "✅"}
    
    report = f"""## {risk_emoji.get(decision['risk_level'], '❓')} Risk Level: {decision['risk_level']}

### Identification
| Field | Value |
|-------|-------|
| **Species** | {formatted_species} |
| **Confidence** | {confidence * 100:.1f}% |
| **Toxicity** | {context.get('toxicity_type', 'Unknown')} |
| **Habitat** | {context.get('habitat', 'Unknown')} |
| **Season** | {context.get('season', 'Unknown')} |
| **Region** | {context.get('region', 'Unknown')} |

### Recommendation
{decision['recommendation']}
"""
    
    if decision['risk_factors']:
        report += "\n### Risk Factors\n"
        for factor in decision['risk_factors']:
            report += f"- {factor}\n"
    
    report += f"\n### LLM Audit\n{llm_verdict}\n"
    
    if context.get('key_warnings'):
        report += f"\n### Key Warnings\n⚠️ {context['key_warnings']}\n"
    
    yield report


CSS = """
.panel { border-radius: 8px; padding: 12px; background: #111827; }
.scroll-panel .prose { max-height: 72vh; overflow-y: auto; padding-right: 6px; }
.status-bar { font-size: 0.95rem; font-weight: 600; padding: 6px 10px; border-radius: 6px;
              background: #374151; color: #f9fafb; }
"""

with gr.Blocks(title="🍄 Mushroom Classification System", css=CSS) as demo:
    gr.Markdown(
        "# 🍄 Mushroom Classification System\n"
        "Upload a photo of a mushroom to identify it and receive an assessment. "
        "The system uses YOLOv26 for visual identification, an ecological database for context, "
        "and an LLM to verify the results."
    )

    # ── Top row — inputs ──────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="🍄 Drop a mushroom photo here", height=220)
        with gr.Column(scale=1):
            season_input = gr.Dropdown(
                choices=["Spring", "Summer", "Autumn", "Winter"],
                value="Autumn", label="🗓️ Season"
            )
            location_input = gr.Textbox(value="United Kingdom", label="📍 Location")
            with gr.Row():
                clear_btn  = gr.ClearButton(value="Clear")
                submit_btn = gr.Button("Submit", variant="primary")

    gr.Markdown("---")

    # ── Bottom row — results ──────────────────────────────────────────
    with gr.Row():
        # YOLO result
        with gr.Column(scale=1):
            yolo_status = gr.Textbox(value="", label="⏳ Vision Model", interactive=False,
                                     elem_classes=["status-bar"], visible=False)
            yolo_box = gr.Markdown(value="", label="🔍 YOLO Identification",
                                   elem_classes=["scroll-panel"], visible=False)

        # LLM audit + report
        with gr.Column(scale=1):
            llm_status = gr.Textbox(value="", label="⏳ LLM Audit", interactive=False,
                                    elem_classes=["status-bar"], visible=False)
            llm_box = gr.Markdown(value="", label="🧠 Report",
                                  elem_classes=["scroll-panel"], visible=False)
            flag_btn = gr.Button("Flag", visible=False)

    def _run(image, season, location):
        _H = lambda v, txt="": gr.update(visible=v, value=txt)

        if image is None:
            yield _H(False), _H(False), _H(False), _H(False), _H(False)
            return

        # ── Locate context DB ─────────────────────────────────────────
        base_dir = os.path.dirname(__file__)
        csv_path = None
        for fname in ("mushroom_context.json", "mushroom_context.csv"):
            for base in (base_dir, os.path.dirname(os.path.dirname(base_dir))):
                p = os.path.join(base, "data", fname)
                if os.path.exists(p):
                    csv_path = p
                    break
            if csv_path:
                break

        # ── Step 1: YOLO (TFLite ~0.2ms — no spinner needed) ─────────
        predicted_species, confidence = predict_image(image)

        if predicted_species is None:
            yield _H(False), _H(True, "❌ Vision API unavailable."), _H(False), _H(False), _H(False)
            return

        formatted = predicted_species.replace("_", " ").title()
        log_drift_image(image, confidence, predicted_species)
        context = get_mushroom_context(formatted, csv_path) if csv_path else {}
        if "error" in context:
            context = {"toxicity_type": "Unknown", "habitat": "Unknown",
                       "season": "Unknown", "region": "Unknown",
                       "key_warnings": "Species not found in database."}

        conf_pct = confidence * 100
        yolo_md = f"""## 🔍 Vision Model Result

| Field | Value |
|---|---|
| **Species** | {formatted} |
| **Confidence** | {conf_pct:.1f}% |
| **Toxicity** | {context.get('toxicity_type', 'Unknown')} |
| **Habitat** | {context.get('habitat', 'Unknown')} |
| **Season** | {context.get('season', 'Unknown')} |
| **Region** | {context.get('region', 'Unknown')} |"""

        # Show YOLO result + LLM spinner together
        yield _H(False), _H(True, yolo_md), _H(True, "🧠 Requesting LLM visual audit — this takes 10–30s..."), _H(False), _H(False)

        # ── Step 2: LLM audit ─────────────────────────────────────────
        llm_verdict = audit_prediction(formatted, confidence, context, season, location, image_path=image)
        decision    = assess_risk(formatted, confidence, context, llm_verdict)

        risk_emoji  = {"CRITICAL": "🚨", "HIGH": "⚠️", "MODERATE": "⚠️", "LOW": "✅"}
        risk_colour = {"CRITICAL": "red", "HIGH": "orange", "MODERATE": "orange", "LOW": "green"}
        rl = decision['risk_level']

        report  = f"## {risk_emoji.get(rl,'❓')} Risk Level: {rl}\n\n"
        report += f"**Recommendation:** {decision['recommendation']}\n\n"

        if decision['risk_factors']:
            report += "### Risk Factors\n"
            for rf in decision['risk_factors']:
                report += f"- {rf}\n"

        report += f"\n### LLM Audit\n{llm_verdict}\n"
        if context.get('key_warnings'):
            report += f"\n---\n⚠️ **Key Warning:** {context['key_warnings']}\n"

        yield _H(False), _H(True, yolo_md), _H(False), _H(True, report), _H(True)

    submit_btn.click(
        fn=_run,
        inputs=[image_input, season_input, location_input],
        outputs=[yolo_status, yolo_box, llm_status, llm_box, flag_btn]
    )
    clear_btn.add([image_input, yolo_box, llm_box])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
