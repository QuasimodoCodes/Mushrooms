import sys
import os
from ultralytics import YOLO

def predict_image(image_path, model_path):
    print(f"Loading custom YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running inference on image: {image_path}")
    # YOLO returns a list of Results objects
    results = model(image_path)
    
    # Extract the top prediction from the first result
    for r in results:
        top1_index = r.probs.top1              # The index of the highest probability class
        top1_conf = r.probs.top1conf.item()    # The actual probability/confidence score
        top1_name = r.names[top1_index]        # The string name of that class
        
        print("\n==========================")
        print("     🏆 PREDICTION 🏆      ")
        print("==========================")
        print(f"Species:    {top1_name}")
        print(f"Confidence: {top1_conf * 100:.2f}%")
        print("==========================\n")
        
        return top1_name, top1_conf

if __name__ == "__main__":
    # If the user didn't pass an image, grab a random one from the test set for demonstration
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        base_test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset_split", "test"))
        
        # Grab the first species folder in the test set
        first_species = os.listdir(base_test_dir)[0]
        species_dir = os.path.join(base_test_dir, first_species)
        
        # Grab the first image in that folder
        first_img = os.listdir(species_dir)[0]
        test_img_path = os.path.join(species_dir, first_img)
        
        print(f"\nNo image provided! Defaulting to a demo image from the test set:")
        print(f"True Label Should Be: {first_species}\n")
    else:
        test_img_path = sys.argv[1]
        
    # Path to the BEST weights our model just calculated
    model_weight_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "yolo_runs", "mushroom_classifier_v1", "weights", "best.pt"))
    
    if not os.path.exists(model_weight_path):
        print(f"Error: Model not found at {model_weight_path}")
    else:
        predict_image(test_img_path, model_weight_path)
