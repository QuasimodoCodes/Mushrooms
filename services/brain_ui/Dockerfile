# 1. Base Image: This is our 'operating system' inside the container.
# We use a lightweight version of Python 3.12 (slim) to keep the container small.
FROM python:3.12-slim

# 2. Set the working directory inside the container. 
# All subsequent commands will run from this folder.
WORKDIR /app

# 3. Copy the requirements file into the container.
COPY services/brain_ui/requirements.txt .

# 4. Install Python dependencies inside the container.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code and necessary data.
# Note how we are treating app.py (the Gradio interface) essentially as the "front door"
# for the Brain UI service. The business logic lives in `scripts/pipeline/`.
COPY app.py /app/app.py
COPY scripts /app/scripts
COPY data/mushroom_context.csv /app/data/mushroom_context.csv

# 6. Expose port 7860 so the outside world can view the Gradio UI.
EXPOSE 7860

# 7. Define the command that runs when the container starts.
CMD ["python", "app.py"]
