import pandas as pd
import os
import sys

# Add the scripts directory to Python's path so we can import our other scripts directly
sys.path.insert(0, os.path.dirname(__file__))
from predict import predict_image

def get_mushroom_context(species_name, csv_path):
    """
    Searches the context CSV for a specific mushroom species and returns its ecological rules.
    """
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        # Search for the specific species (case-insensitive)
        match = df[df['species_name'].str.lower() == species_name.lower()]
        
        if not match.empty:
            # We found it! Extract the row data as a dictionary
            return match.iloc[0].to_dict()
        else:
            return {"error": f"Species '{species_name}' not found in context database."}
            
    except Exception as e:
        return {"error": f"Failed to read context database: {str(e)}"}

def main():
    print("=========================================")
    print("   Mushroom Safety: End-to-End Pipeline  ")
    print("=========================================")
    
    # 1. Setup paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "mushroom_context.csv")
    model_path = os.path.join(base_dir, "docs", "yolo_runs", "mushroom_classifier_v1", "weights", "best.pt")
    
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
