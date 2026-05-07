"""
LLM Provider Abstraction Layer
===============================
This module provides a clean interface to swap between LLM backends 
(Ollama for local, Gemini for cloud) without changing any other code.

Currently active: Ollama (local)
To switch to Gemini: Change ACTIVE_PROVIDER to "gemini" and set your API key.
"""



import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CONFIGURATION - Change these to switch LLM providers
# ============================================================
ACTIVE_PROVIDER = "gemini"  # Options: "ollama" or "gemini"
OLLAMA_MODEL = "llama3:latest"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
_MULTIMODAL_OLLAMA_MODELS = ("gemma4", "gemma", "llava", "bakllava")

# Gemini config — key is loaded from the GEMINI_API_KEY environment variable (set in .env)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


def query_llm(prompt, image_path=None):
    """
    Sends a prompt to whichever LLM backend is currently active.
    Returns the LLM's text response as a string.
    """
    if ACTIVE_PROVIDER == "ollama":
        return _query_ollama(prompt)
    elif ACTIVE_PROVIDER == "gemini":
        return _query_gemini(prompt)
    else:
        return f"Error: Unknown provider '{ACTIVE_PROVIDER}'"


def _query_ollama(prompt, image_path=None):
    """Sends a prompt to the local Ollama server. Passes image if model is multimodal."""
    import base64

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 200}
    }

    # Attach image for multimodal models (e.g. gemma4, llava)
    if image_path and any(m in OLLAMA_MODEL for m in _MULTIMODAL_OLLAMA_MODELS):
        try:
            with open(image_path, "rb") as f:
                payload["images"] = [base64.b64encode(f.read()).decode("utf-8")]
        except Exception:
            pass  # fall back to text-only if image can't be read

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "No response from Ollama.")
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it is running (run 'ollama serve' in a terminal)."
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"


def _query_gemini(prompt, image_path=None):
    """Sends a prompt (and optional image) to Google Gemini API."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not set in llm_provider.py"

    try:
        import warnings
        warnings.filterwarnings("ignore")
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gen_config = genai.types.GenerationConfig(max_output_tokens=250)
        model = genai.GenerativeModel(GEMINI_MODEL)

        if image_path:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            contents = [prompt, img]
        else:
            contents = prompt

        import time
        for attempt in range(3):
            try:
                response = model.generate_content(contents,
                                                  generation_config=gen_config)
                return response.text
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "exhausted" in str(e).lower():
                    wait = 30 * (attempt + 1)
                    print(f"\n[Gemini rate limit] waiting {wait}s before retry {attempt+1}/3...")
                    time.sleep(wait)
                else:
                    return f"Error querying Gemini: {str(e)}"
        return "Error querying Gemini: rate limit after 3 retries"

    except ImportError:
        return "Error: 'google-generativeai' package not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"
