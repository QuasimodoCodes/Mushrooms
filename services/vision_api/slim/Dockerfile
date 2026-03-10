# TFLite Slim Deployment
# ~200MB Docker image vs ~1.5GB PyTorch version
# No torch, no torchvision, no ultralytics — just TFLite runtime

FROM python:3.12-slim

WORKDIR /app

# Minimal system deps (no gcc needed)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install slim Python dependencies (no PyTorch!)
COPY services/vision_api/slim/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what we need: TFLite model + class names
COPY docs/yolo_runs/yolo26_classifier_v1/weights/best_float16.tflite /app/docs/yolo_runs/yolo26_classifier_v1/weights/best_float16.tflite
COPY docs/yolo_runs/yolo26_classifier_v1/weights/class_names.json /app/docs/yolo_runs/yolo26_classifier_v1/weights/class_names.json

# Copy the slim API script
COPY services/vision_api/slim/main.py /app/services/vision_api/slim/main.py

EXPOSE 8000

CMD ["uvicorn", "services.vision_api.slim.main:app", "--host", "0.0.0.0", "--port", "8000"]
