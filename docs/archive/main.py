"""
Mushroom Safety Classification System - Main Pipeline
======================================================
This is the single entry point that chains every module together:

  Image → YOLO (Vision) → CSV (Knowledge) → LLM (Reasoning) → Risk Engine (Safety) → Final Report

Usage:
  python main.py                          # Uses a demo image from the test set
  python main.py path/to/mushroom.jpg     # Analyze a specific image
"""

import sys
import os

# Add the pipeline directory to Python's path
pipeline_dir = os.path.join(os.path.dirname(__file__), "scripts", "pipeline")
sys.path.insert(0, pipeline_dir)

from predict import predict_image
from integration import get_mushroom_context
from audit_layer import audit_prediction
from risk_engine import assess_risk


def run_pipeline(image_path, user_season="Unknown", user_location="Unknown"):
    """
    Runs the full mushroom safety pipeline on a single image.
    
    Args:
        image_path (str): Path to the mushroom image
        user_season (str): The current season where the user found the mushroom
        user_location (str): The user's geographic location
    
    Returns:
        dict: Complete safety report
    """
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "data", "mushroom_context.csv")
    model_path = os.path.join(base_dir, "docs", "yolo_runs", "mushroom_classifier_v1", "weights", "best.pt")
    
    # ─── STEP 1: YOLO Vision ───
    print("\n" + "="*50)
    print("  🍄 MUSHROOM SAFETY CLASSIFICATION SYSTEM 🍄")
    print("="*50)
    print(f"\n📸 Image: {os.path.basename(image_path)}")
    print(f"📍 Location: {user_location} | 🗓️ Season: {user_season}")
    
    print("\n[1/4] 👁️  Running YOLO Vision Model...")
    predicted_species, confidence = predict_image(image_path, model_path)
    formatted_species = predicted_species.replace("_", " ")
    
    # ─── STEP 2: CSV Knowledge Lookup ───
    print(f"[2/4] 📊 Looking up '{formatted_species}' in ecological database...")
    context = get_mushroom_context(formatted_species, csv_path)
    
    if "error" in context:
        print(f"  ❌ {context['error']}")
        print("  Falling back to safety-first approach...")
        context = {"toxicity_type": "Unknown", "habitat": "Unknown", "season": "Unknown", "region": "Unknown", "key_warnings": "Species not found in database. Treat as potentially dangerous."}
    
    # ─── STEP 3: LLM Audit ───
    print(f"[3/4] 🔍 Asking LLM to verify identification...")
    llm_verdict = audit_prediction(formatted_species, confidence, context, user_season, user_location)
    
    # ─── STEP 4: Risk Decision ───
    print(f"[4/4] ⚖️  Running risk assessment...")
    decision = assess_risk(formatted_species, confidence, context, llm_verdict)
    
    # ─── FINAL REPORT ───
    print("\n" + "="*50)
    print(f"  RISK LEVEL: {decision['risk_level']}")
    print("="*50)
    print(f"\n🏷️  Species:     {formatted_species}")
    print(f"📊 Confidence:  {confidence * 100:.1f}%")
    print(f"☠️  Toxicity:    {context.get('toxicity_type', 'Unknown')}")
    print(f"🌿 Habitat:     {context.get('habitat', 'Unknown')}")
    print(f"🗓️  Season:      {context.get('season', 'Unknown')}")
    print(f"🌍 Region:      {context.get('region', 'Unknown')}")
    
    if decision['risk_factors']:
        print(f"\n{'─'*50}")
        print("RISK FACTORS:")
        for factor in decision['risk_factors']:
            print(f"  {factor}")
    
    print(f"\n{'─'*50}")
    print(f"LLM AUDIT:\n  {llm_verdict}")
    
    print(f"\n{'─'*50}")
    print(f"RECOMMENDATION:\n  {decision['recommendation']}")
    
    if context.get('key_warnings'):
        print(f"\n{'─'*50}")
        print(f"⚠️  KEY WARNINGS:\n  {context['key_warnings']}")
    
    print("\n" + "="*50 + "\n")
    
    return decision


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        # Default: grab a test image
        test_dir = os.path.join(os.path.dirname(__file__), "data", "dataset_split", "test")
        
        # Try to use a Death Cap image for a dramatic demo
        target = "Amanita_phalloides"
        if os.path.exists(os.path.join(test_dir, target)):
            species_dir = os.path.join(test_dir, target)
        else:
            species_dir = os.path.join(test_dir, os.listdir(test_dir)[0])
        
        img_path = os.path.join(species_dir, os.listdir(species_dir)[0])
        print(f"No image provided. Using demo: {os.path.basename(img_path)}")
    
    # Run with mock environment data
    run_pipeline(img_path, user_season="Autumn", user_location="United Kingdom")
