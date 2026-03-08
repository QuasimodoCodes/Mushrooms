"""
Mushroom Safety Classifier - Gradio Web UI
============================================
A simple drag-and-drop interface for the mushroom safety pipeline.
Run this script and open the URL in your browser.

Usage:
  python app.py
"""

import gradio as gr
import sys
import os
import logging
import shutil
from datetime import datetime

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

# Start the Prometheus metrics server on port 8001
prometheus_client.start_http_server(8001)

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
    if confidence >= 0.70:
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

def classify_mushroom(image, season, location, progress=gr.Progress()):
    """
    Main function called by Gradio when the user submits an image.
    """
    if image is None:
        yield "Please upload an image first."
        return

    progress(0.0, desc="🍄 Initializing...")
    yield "### 🍄 Initializing classification process..."

    progress(0.1, desc="🍄 Locating Context Database...")
    # Look for the CSV locally in data/ or in the Docker path
    base_dir = os.path.dirname(__file__)
    if os.path.exists(os.path.join(base_dir, "data", "mushroom_context.csv")):
        csv_path = os.path.join(base_dir, "data", "mushroom_context.csv")
    else:
        # Running locally in repo structure
        csv_path = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "data", "mushroom_context.csv")

    yield "### 🍄 Uploading image to Vision API for identification..."
    # Step 1: YOLO Vision (via Vision API)
    progress(0.2, desc="🍄 Uploading image to Vision API...")
    predicted_species, confidence = predict_image(image)
    
    if predicted_species is None:
        yield "❌ Error: Could not connect to the Vision API. Please ensure it is running in another terminal."
        return
        
    # Format the species name nicely if it has underscores
    formatted_species = predicted_species.replace("_", " ").title() if predicted_species else "Unknown"

    yield f"### 🍄 Identified as **{formatted_species}**. Checking data drift..."
    # Step 1.5: Trigger Drift Detection Logic (MLOps)
    # Background save if confidence is too low
    progress(0.4, desc="🍄 Checking classification confidence & data drift...")
    log_drift_image(image, confidence, predicted_species)

    yield f"### 🍄 Fetching ecological context for **{formatted_species}**..."
    progress(0.6, desc="🍄 Fetching ecological context from Knowledge Base...")
    context = get_mushroom_context(formatted_species, csv_path)
    if "error" in context:
        context = {
            "toxicity_type": "Unknown", "habitat": "Unknown",
            "season": "Unknown", "region": "Unknown",
            "key_warnings": "Species not found in database. Treat as potentially dangerous."
        }
    
    # Step 3: LLM Audit
    yield f"### 🍄 Requesting Safety Audit from LLM for **{formatted_species}**..."
    progress(0.7, desc="🍄 Requesting Safety Audit from LLM (Llama3/Gemini)...")
    llm_verdict = audit_prediction(formatted_species, confidence, context, season, location)
    
    # Step 4: Risk Decision
    yield "### 🍄 Calculating final risk level..."
    progress(0.9, desc="🍄 Calculating final risk level...")
    decision = assess_risk(formatted_species, confidence, context, llm_verdict)
    
    yield "### 🍄 Generating Safety Report..."
    progress(1.0, desc="🍄 Generating Safety Report...")
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


# Build the Gradio interface
demo = gr.Interface(
    fn=classify_mushroom,
    inputs=[
        gr.Image(type="filepath", label="🍄 Drop a mushroom photo here"),
        gr.Dropdown(
            choices=["Spring", "Summer", "Autumn", "Winter"],
            value="Autumn",
            label="🗓️ Current Season"
        ),
        gr.Textbox(
            value="United Kingdom",
            label="📍 Your Location"
        )
    ],
    outputs=gr.Markdown(label="📋 Safety Report"),
    title="🍄 Mushroom Safety Classification System",
    description="Upload a photo of a mushroom to identify it and receive a safety assessment. The system uses YOLOv26 for visual identification, an ecological database for context, and an LLM to verify the results."
)


if __name__ == "__main__":
    # server_name="0.0.0.0" is required inside Docker so the app is
    # reachable from outside the container (i.e., your browser on the host).
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
