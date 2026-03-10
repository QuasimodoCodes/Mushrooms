import os
import io
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from prometheus_fastapi_instrumentator import Instrumentator

# 1. Initialize the FastAPI application
app = FastAPI(title="Mushroom Vision API (TFLite)")

# Initialize and expose Prometheus metrics
Instrumentator().instrument(app).expose(app)

# 2. Paths — inside the container everything lives under /app
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WEIGHTS_DIR = os.path.join(BASE_DIR, "docs", "yolo_runs", "yolo26_classifier_v1", "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_float16.tflite")
CLASS_NAMES_PATH = os.path.join(WEIGHTS_DIR, "class_names.json")

# 3. Load class names (index -> species name mapping)
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = {int(k): v for k, v in json.load(f).items()}

# 4. Load TFLite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
_input_details = interpreter.get_input_details()[0]
_output_details = interpreter.get_output_details()[0]
_input_index = _input_details["index"]
_output_index = _output_details["index"]

print(f"Loaded TFLite model: {os.path.basename(MODEL_PATH)}")
print(f"  Classes: {len(CLASS_NAMES)}, Input shape: {_input_details['shape']}")


def preprocess(image: Image.Image) -> np.ndarray:
    """Center-crop to square, resize to 224x224, normalize to [0, 1]."""
    image = image.convert("RGB")
    w, h = image.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize((224, 224), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    return arr.reshape(1, 224, 224, 3)


def predict_tflite(input_tensor: np.ndarray) -> tuple[str, float]:
    """Run inference and return (class_name, confidence)."""
    interpreter.set_tensor(_input_index, input_tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(_output_index)[0]
    top_index = int(np.argmax(probs))
    confidence = float(probs[top_index])
    class_name = CLASS_NAMES.get(top_index, f"unknown_{top_index}")
    return class_name, confidence


@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run monitoring."""
    return {"status": "healthy", "model_loaded": interpreter is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, processes it through TFLite,
    and returns the top classification and confidence score.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    input_tensor = preprocess(image)
    class_name, confidence = predict_tflite(input_tensor)

    return {"class": class_name, "confidence": confidence}
