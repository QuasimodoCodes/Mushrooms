"""
LLM Provider Abstraction Layer
===============================
This module provides a clean interface to swap between LLM backends.
All configuration lives in config.py at the project root — edit there.

Providers:
  "ollama"  — local Ollama server (any text model e.g. llama3)
  "gemma"   — local Ollama server using Gemma 4 with multimodal image support
  "gemini"  — Google Gemini cloud API
"""

import base64
import os
import sys
import requests

# Resolve project root and import central config
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)
from config import (ACTIVE_LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL,
                    GEMINI_MODEL, GEMINI_API_KEY)

ACTIVE_PROVIDER = ACTIVE_LLM_PROVIDER

# Multimodal-capable Ollama models — these receive the image when available.
# Add any new vision-capable models here.
_MULTIMODAL_OLLAMA_MODELS = ("gemma4", "gemma3", "llava", "bakllava", "moondream")


def query_llm(prompt, image_path=None):
    """
    Sends a prompt to whichever LLM backend is currently active.

    Args:
        prompt (str): The text prompt to send.
        image_path (str|None): Optional path to an image file. Passed automatically
                               to multimodal models (e.g. gemma4, llava).
    Returns:
        str: The LLM's text response.
    """
    if ACTIVE_PROVIDER == "ollama":
        return _query_ollama(prompt, image_path)
    elif ACTIVE_PROVIDER == "gemini":
        return _query_gemini(prompt, image_path)
    else:
        return f"Error: Unknown provider '{ACTIVE_PROVIDER}'"


def _query_ollama(prompt, image_path=None):
    """
    Sends a prompt to the local Ollama server.
    - Multimodal models (gemma4, llava etc.) use /api/chat with image attached.
    - Text-only models use /api/generate.
    """
    try:
        is_multimodal = any(name in OLLAMA_MODEL for name in _MULTIMODAL_OLLAMA_MODELS)
        has_image = is_multimodal and image_path and os.path.exists(str(image_path))
        if is_multimodal and image_path and not os.path.exists(str(image_path)):
            import logging
            logging.getLogger(__name__).warning(f"Image path does not exist, falling back to text-only: {image_path}")

        if has_image:
            # /api/chat supports images; /api/generate does not on newer models
            # Handle both file paths and PIL Image objects
            if hasattr(image_path, 'save'):  # PIL Image object
                import io
                buf = io.BytesIO()
                image_path.save(buf, format="JPEG")
                img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            else:
                with open(str(image_path), "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
            chat_url = OLLAMA_URL.replace("/api/generate", "/api/chat")
            payload = {
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
            }
            response = requests.post(chat_url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "No response from Ollama.")
        else:
            payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            response = requests.post(OLLAMA_URL, json=payload, timeout=180)
            response.raise_for_status()
            return response.json().get("response", "No response from Ollama.")

    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it is running (run 'ollama serve' in a terminal)."
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"


def _query_gemini(prompt, image_path=None):
    """Sends a prompt to Google Gemini API with optional image input."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not set."
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

        if image_path and os.path.exists(image_path):
            import PIL.Image
            img = PIL.Image.open(image_path)
            response = model.generate_content([prompt, img])
        else:
            response = model.generate_content(prompt)

        return response.text
    except ImportError:
        return "Error: 'google-generativeai' package not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"
