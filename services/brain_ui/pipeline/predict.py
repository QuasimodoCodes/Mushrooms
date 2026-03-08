import sys
import os
import requests

# URL of our new Vision API
# We default to localhost for local testing, but Docker Compose will override this!
VISION_API_URL = os.environ.get("VISION_API_URL", "http://127.0.0.1:8000/predict")

def predict_image(image_path):
    print(f"Sending image to Vision API at: {VISION_API_URL}")
    print(f"Running inference on image: {image_path}")
    
    # We open the image file in binary read mode ('rb')
    with open(image_path, "rb") as image_file:
        # We package the file into a dictionary that the 'requests' library understands
        files = {"file": (os.path.basename(image_path), image_file, "image/jpeg")}
        
        try:
            # We send a POST request with our image to the API
            response = requests.post(VISION_API_URL, files=files, timeout=30)
            
            # If the API returned an error code (like 404 or 500), this will raise an exception
            response.raise_for_status()
            
            # The API returns JSON like {"class": "Amanita", "confidence": 0.9}
            # We parse that JSON into a normal Python dictionary
            result_data = response.json()
            
            top1_name = result_data.get("class")
            top1_conf = result_data.get("confidence")
            
            print("\n==========================")
            print("     🏆 PREDICTION 🏆      ")
            print("==========================")
            print(f"Species:    {top1_name}")
            print(f"Confidence: {top1_conf * 100:.2f}%")
            print("==========================\n")
            
            return top1_name, top1_conf
            
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] Failed to connect to the Vision API: {e}")
            print("Ensure the API is running (uvicorn services.vision_api.main:app --reload)")
            return None, 0.0

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
        
    predict_image(test_img_path)
