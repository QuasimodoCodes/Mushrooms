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

# Add the pipeline directory to Python's path
pipeline_dir = os.path.join(os.path.dirname(__file__), "scripts", "pipeline")
sys.path.insert(0, pipeline_dir)

from predict import predict_image
from integration import get_mushroom_context
from audit_layer import audit_prediction
from risk_engine import assess_risk


def classify_mushroom(image, season, location):
    """
    Main function called by Gradio when the user submits an image.
    """
    if image is None:
        return "Please upload an image first."
    
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "data", "mushroom_context.csv")
    model_path = os.path.join(base_dir, "docs", "yolo_runs", "mushroom_classifier_v1", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        return "❌ Error: Model weights not found. Please train the model first (scripts/training/train_yolo.py)."
    
    # Step 1: YOLO Vision
    predicted_species, confidence = predict_image(image, model_path)
    formatted_species = predicted_species.replace("_", " ")
    
    # Step 2: CSV Knowledge Lookup
    context = get_mushroom_context(formatted_species, csv_path)
    if "error" in context:
        context = {
            "toxicity_type": "Unknown", "habitat": "Unknown",
            "season": "Unknown", "region": "Unknown",
            "key_warnings": "Species not found in database. Treat as potentially dangerous."
        }
    
    # Step 3: LLM Audit
    llm_verdict = audit_prediction(formatted_species, confidence, context, season, location)
    
    # Step 4: Risk Decision
    decision = assess_risk(formatted_species, confidence, context, llm_verdict)
    
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
    
    return report


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
    description="Upload a photo of a mushroom to identify it and receive a safety assessment. The system uses YOLOv8 for visual identification, an ecological database for context, and an LLM to verify the results."
)


if __name__ == "__main__":
    demo.launch()
