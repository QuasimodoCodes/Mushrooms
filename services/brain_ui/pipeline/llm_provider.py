"""
LLM Provider Abstraction Layer
===============================
This module provides a clean interface to swap between LLM backends.
All configuration lives in config.py at the project root — edit there.
"""

import os
import sys
import requests

# Resolve project root and import central config
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _ROOT)
from config import ACTIVE_LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL, GEMINI_MODEL, GEMINI_API_KEY

ACTIVE_PROVIDER = ACTIVE_LLM_PROVIDER


def query_llm(prompt):
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


def _query_ollama(prompt):
    """Sends a prompt to the local Ollama server."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False  # Get the full response at once
            },
            timeout=120  # LLMs can be slow locally
        )
        response.raise_for_status()
        return response.json().get("response", "No response from Ollama.")
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it is running (run 'ollama serve' in a terminal)."
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"


def _query_gemini(prompt):
    """Sends a prompt to Google Gemini API. (For future use)"""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not set in llm_provider.py"
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        return "Error: 'google-generativeai' package not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"
