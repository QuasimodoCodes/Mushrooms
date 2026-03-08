# 1. Base Image: This is our 'operating system' inside the container.
# We use a lightweight version of Python 3.12 (slim) to keep the container small.
FROM python:3.12-slim

# 2. Set the working directory inside the container. 
# All subsequent commands will run from this folder.
WORKDIR /app

# 3. Install necessary system dependencies.
# YOLO requires some basic C++ building tools (gcc) and image processing libraries
# that don't come pre-installed on the minimalist 'slim' python image.
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file into the container.
COPY services/vision_api/requirements.txt .

# 5. Install Python dependencies inside the container.
# We use --no-cache-dir to avoid storing the downloaded zip files, keeping the image small.
# Note: We explicitly install the CPU-only version of PyTorch to save space,
# as YOLO inferences run just fine on CPUs for standard web requests.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 6. Copy the model weights into the container.
# We place this in the exact folder structure that our main.py expects.
COPY docs/yolo_runs/yolo26_classifier_v1/weights/best.pt /app/docs/yolo_runs/yolo26_classifier_v1/weights/best.pt

# 7. Copy the actual FastAPI script into the container.
COPY services/vision_api/main.py /app/services/vision_api/main.py

# 8. Expose port 8000 so the outside world (like our Brain UI) can talk to this container.
EXPOSE 8000

# 9. Define the command that runs when the container starts.
# This is the exact same command you ran locally in your terminal!
CMD ["uvicorn", "services.vision_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
