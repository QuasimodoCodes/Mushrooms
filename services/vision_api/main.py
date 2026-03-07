import os
import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from ultralytics import YOLO

# 1. Initialize the FastAPI application
# This 'app' object acts as our web server, listening for incoming requests.
app = FastAPI(title="Mushroom Vision API")

# 2. Define the path to our trained YOLO classification model.
# We traverse up 3 directories from main.py (vision_api -> services -> Mushroom)
# to find the 'docs/yolo_runs/...' folder where our weights are stored.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "docs", "yolo_runs", "mushroom_classifier_v1", "weights", "best.pt")

# 3. Load the YOLO model into memory.
# We do this OUTSIDE the endpoint function so it only loads once when the server starts,
# ensuring each prediction request is fast.
model = YOLO(MODEL_PATH)

# ==========================================
# HEALTH ENDPOINT FOR CLOUD RUN MONITORING
# ==========================================
@app.get("/health")
async def health_check():
    """
    Health check endpoint used by Cloud Run and other orchestrators
    to verify the container is alive and capable of responding.
    """
    return {"status": "healthy", "model_loaded": model is not None}

# 4. Define our prediction endpoint.
# The @app.post decorator means this function only responds to HTTP POST requests 
# (which are used when sending data, like uploading a file).
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, processes it through YOLO, 
    and returns the top classification and confidence score.
    """
    
    # Read the raw binary data of the uploaded image file
    image_bytes = await file.read()
    
    # Convert the binary data into a Python Pillow (PIL) Image object
    # This prepares the image so our YOLO model can natively process it.
    image = Image.open(io.BytesIO(image_bytes))
    
    # Pass the image through our artificial neural network (YOLO)
    # This triggers the actual inference/prediction.
    results = model(image)
    
    # YOLO returns a list of results (since you could pass multiple images at once).
    # We take the first result since we only sent one image.
    prediction = results[0]
    
    # Extract the index number of the most likely class (Top 1)
    top_class_index = prediction.probs.top1
    
    # Look up the human-readable string name for that index (e.g., 'Amanita_muscaria')
    class_name = prediction.names[top_class_index]
    
    # Extract the confidence score as a standard Python float
    confidence = float(prediction.probs.top1conf)
    
    # Return the data as a clean JSON dictionary.
    # The Brain UI will easily parse this dictionary natively.
    return {"class": class_name, "confidence": confidence}
