"""
launch.py — Start the full Mushroom System with one command.

Usage:
    python launch.py

Starts:
    - Vision API  on http://localhost:8000
    - Brain UI    on http://localhost:7860
"""

import subprocess
import sys
import time
import os
import signal

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

def kill_port(port):
    """Kill any process listening on the given port (Windows)."""
    try:
        result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
        pids = set()
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts:
                    pids.add(parts[-1])
        for pid in pids:
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
            print(f"  Killed PID {pid} on port {port}")
    except Exception as e:
        print(f"  Warning: could not kill port {port}: {e}")

print("Clearing ports 8000 and 7860...")
kill_port(8000)
kill_port(7860)
time.sleep(1)

# Load .env into the environment so subprocesses inherit the keys
_env_path = os.path.join(ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# ── Start Ollama if configured ────────────────────────────────────────────────
sys.path.insert(0, ROOT)
from config import ACTIVE_LLM_PROVIDER

if ACTIVE_LLM_PROVIDER == "ollama":
    import shutil
    ollama_exe = shutil.which("ollama") or r"C:\Users\spide\AppData\Local\Programs\Ollama\ollama.exe"
    if ollama_exe and os.path.exists(ollama_exe):
        print(f"Starting Ollama ({ollama_exe}) ...")
        subprocess.Popen([ollama_exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Waiting for Ollama to be ready...")
        time.sleep(5)
    else:
        print("WARNING: Ollama not found. LLM audit will fail. Install from https://ollama.com")

print("Starting Vision API on http://localhost:8000 ...")
vision = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "services.vision_api.main:app",
     "--host", "0.0.0.0", "--port", "8000"],
    cwd=ROOT,
)

# Give the Vision API a moment to load the model before the UI tries to connect
print("Waiting for Vision API to load model...")
time.sleep(5)

print("Starting Brain UI on http://localhost:7860 ...")
ui = subprocess.Popen(
    [PYTHON, "services/brain_ui/app.py"],
    cwd=ROOT,
)

print("\nBoth services running. Open http://localhost:7860 in your browser.")
print("Press Ctrl+C to stop everything.\n")

try:
    vision.wait()
    ui.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    vision.terminate()
    ui.terminate()
