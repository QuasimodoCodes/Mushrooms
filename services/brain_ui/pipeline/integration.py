import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import predict_image

# Cache the loaded database in memory so we only read the file once per session
_DB_CACHE = {}

def get_mushroom_context(species_name, csv_path):
    """
    Looks up a species in the mushroom context database (JSON or CSV).
    Accepts either a .json or .csv path — prefers JSON if a .json equivalent exists.
    """
    global _DB_CACHE

    # Prefer JSON version if it exists alongside the CSV
    json_path = os.path.splitext(csv_path)[0] + ".json"
    if os.path.exists(json_path):
        db_path = json_path
    else:
        db_path = csv_path

    try:
        if db_path not in _DB_CACHE:
            if db_path.endswith(".json"):
                with open(db_path, encoding="utf-8") as f:
                    records = json.load(f)
                _DB_CACHE[db_path] = {r["species_name"].lower(): r for r in records}
            else:
                import pandas as pd
                df = pd.read_csv(db_path)
                _DB_CACHE[db_path] = {
                    row["species_name"].lower(): row.to_dict()
                    for _, row in df.iterrows()
                }

        db = _DB_CACHE[db_path]
        match = db.get(species_name.lower())
        if match:
            return match
        else:
            return {"error": f"Species '{species_name}' not found in context database."}

    except Exception as e:
        return {"error": f"Failed to read context database: {str(e)}"}

def main():
    print("=========================================")
    print("   Mushroom Safety: End-to-End Pipeline  ")
    print("=========================================")
    
    # 1. Setup paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    csv_path = os.path.join(base_dir, "data", "mushroom_context.csv")
    model_path = os.path.join(base_dir, "docs", "yolo_runs", "yolo26_classifier_v1", "weights", "best.pt")
    
    # Grab a test image (using the same logic from our predict script)
    test_dir = os.path.join(base_dir, "data", "dataset_split", "test")
    # Let's deliberately test it on a highly toxic mushroom if it exists in the test set, otherwise grab the first one
    target_species = "Amanita_phalloides" # Death Cap
    if os.path.exists(os.path.join(test_dir, target_species)):
        species_dir = os.path.join(test_dir, target_species)
    else:
        # Fallback to the first available folder
        species_dir = os.path.join(test_dir, os.listdir(test_dir)[0])
        
    test_image = os.path.join(species_dir, os.listdir(species_dir)[0])
    
    print("\n[STEP 1] Running YOLO Vision Model...")
    # 2. Run the visual prediction
    predicted_species, confidence = predict_image(test_image, model_path)
    
    # Format the species name to match the CSV (YOLO uses underscores, CSV uses spaces)
    formatted_species = predicted_species.replace("_", " ")
    
    print(f"\n[STEP 2] Querying Ecological Database for: {formatted_species}...")
    # 3. Look up the ecological context
    context = get_mushroom_context(formatted_species, csv_path)
    
    print("\n==========================")
    print("    🍄 FINAL REPORT 🍄     ")
    print("==========================")
    print(f"Species:    {formatted_species}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("--------------------------")
    
    if "error" in context:
        print(f"Database Error: {context['error']}")
    else:
        print(f"Toxicity:   {context['toxicity_type']}")
        print(f"Habitat:    {context['habitat']}")
        print(f"Season:     {context['season']}")
        print(f"Region:     {context['region']}")
        print(f"\n⚠️ WARNING: {context['key_warnings']}")
    print("==========================\n")

if __name__ == "__main__":
    main()
