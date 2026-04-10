"""
LLM Audit Layer
================
This script takes YOLO's prediction + the CSV ecological context + the user's
location/season and asks an LLM: "Does this identification make sense?"

This is the 'reasoning layer' that catches mistakes the vision model makes.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))
from llm_provider import query_llm, ACTIVE_PROVIDER, _MULTIMODAL_OLLAMA_MODELS, OLLAMA_MODEL

logger = logging.getLogger(__name__)


def build_audit_prompt(species_name, confidence, context, user_season, user_location):
    """Ecological plausibility prompt — used by text-only providers (ollama)."""
    return f"""You are a mycology safety auditor AI. Your job is to verify whether a visual mushroom identification is plausible given the environmental context.

Here is the evidence:

**Visual Identification (from YOLO model):**
- Predicted Species: {species_name}
- Confidence Score: {confidence * 100:.1f}%

**Ecological Database Entry for {species_name}:**
- Toxicity: {context.get('toxicity_type', 'Unknown')}
- Known Habitat: {context.get('habitat', 'Unknown')}
- Typical Season: {context.get('season', 'Unknown')}
- Known Region: {context.get('region', 'Unknown')}
- Key Warnings: {context.get('key_warnings', 'None')}

**User's Current Environment:**
- Season: {user_season}
- Location: {user_location}

**Your Task:**
1. State whether the identification is PLAUSIBLE or SUSPICIOUS based on the environmental match.
2. If the confidence score is below 50%, flag that the model is uncertain.
3. If the species is toxic or deadly, issue a clear safety warning regardless of plausibility.
4. Keep your response concise (3-5 sentences max).

Begin your response with either "✅ PLAUSIBLE" or "⚠️ SUSPICIOUS" or "🚨 DANGER"
"""


def build_visual_audit_prompt(species_name, confidence, context, user_season, user_location):
    """
    Visual audit prompt — used by multimodal providers (gemma, gemini).
    The image is passed separately alongside this prompt.
    Includes full CSV ecological context and asks for specific field identification guidance.
    """
    toxicity    = context.get('toxicity_type', 'Unknown')
    habitat     = context.get('habitat', 'Unknown')
    season      = context.get('season', 'Unknown')
    region      = context.get('region', 'Unknown')
    warnings    = context.get('key_warnings', 'None listed.')
    edibility   = context.get('edibility', '')
    cap         = context.get('cap_description', '')
    gills       = context.get('gill_description', '')
    stem        = context.get('stem_description', '')
    smell       = context.get('smell', '')
    lookalikes  = context.get('lookalikes', '')

    # Build the feature section dynamically — only include fields that have data
    features = []
    if cap:       features.append(f"- Cap: {cap}")
    if gills:     features.append(f"- Gills: {gills}")
    if stem:      features.append(f"- Stem: {stem}")
    if smell:     features.append(f"- Smell: {smell}")
    if lookalikes: features.append(f"- Dangerous lookalikes: {lookalikes}")
    features_block = "\n".join(features) if features else "- See key warnings below."

    return f"""You are a field mycology safety auditor. You have been given a photo of a mushroom taken in the field.

A computer vision model identified it as **{species_name}** with {confidence * 100:.1f}% confidence.

**Ecological database entry for {species_name}:**
- Toxicity: {toxicity}
- Edibility: {edibility if edibility else 'See toxicity'}
- Typical habitat: {habitat}
- Season: {season}
- Region: {region}
- Key warnings: {warnings}

**Visual identification features to check in the photo:**
{features_block}

**User's current environment:** {user_season}, {user_location}

**Your Task:**
1. Look at the photo carefully and compare it against the known features above.
2. Comment on what you can see — cap shape/colour, gill colour, stem, any bruising or staining.
3. Mention relevant field checks the user should do in person (e.g. "check if the gills are free from the stem", "look for a ring on the stem", "smell for an ink/phenol odour", "check for nearby oak trees").
4. State AGREE if the photo is consistent with {species_name}, or DISAGREE if something looks wrong.
5. If the species is toxic or deadly, issue a clear safety warning.

Begin your response with "✅ AGREE", "⚠️ DISAGREE", or "🚨 DANGER", then give your field assessment.
"""


def audit_prediction(species_name, confidence, context,
                     user_season="Unknown", user_location="Unknown",
                     image_path=None):
    """
    Runs the full audit: builds the prompt, queries the LLM, returns the verdict.
    Multimodal providers (gemma, gemini) receive the image for visual verification.
    """
    is_multimodal_ollama = (ACTIVE_PROVIDER == "ollama" and
                            any(name in OLLAMA_MODEL for name in _MULTIMODAL_OLLAMA_MODELS))
    logger.info(f"[AUDIT] image_path={image_path!r}  exists={os.path.exists(str(image_path)) if image_path else False}")
    if (is_multimodal_ollama or ACTIVE_PROVIDER == "gemini") and image_path:
        prompt = build_visual_audit_prompt(species_name, confidence, context, user_season, user_location)
        logger.info(f"[AUDIT] Sending image + prompt to {OLLAMA_MODEL if ACTIVE_PROVIDER == 'ollama' else ACTIVE_PROVIDER} for visual verification...")
    else:
        prompt = build_audit_prompt(species_name, confidence, context, user_season, user_location)
        logger.info("[AUDIT] Sending data to LLM for verification...")

    response = query_llm(prompt, image_path=image_path)
    logger.info(f"[AUDIT] LLM Response: {response}")
    return _strip_thinking(response)


def _strip_thinking(text: str) -> str:
    """Remove chain-of-thought preamble — keep only from the final verdict onwards."""
    import re
    # Handle explicit <think>...</think> blocks (some models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Find the last occurrence of a verdict marker and return from there
    for marker in ("🚨 DANGER", "⚠️ DISAGREE", "✅ AGREE", "DANGER", "DISAGREE", "AGREE"):
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx:].strip()
    return text

# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    # Mock data to test the audit layer independently
    mock_context = {
        "toxicity_type": "Deadly",
        "habitat": "Deciduous forests (often under oak)",
        "season": "Summer to autumn",
        "region": "Europe, North America, Australia",
        "key_warnings": '"Death Cap". Contains fatal amatoxins. Responsible for the majority of fatal mushroom poisonings globally.'
    }

    print("=========================================")
    print("   LLM Audit Layer - Standalone Test     ")
    print("=========================================")

    # Test Case: A deadly mushroom found in the wrong season
    result = audit_prediction(
        species_name="Amanita phalloides",
        confidence=0.72,
        context=mock_context,
        user_season="Winter",
        user_location="Norway"
    )

    print("\n==========================")
    print("   🔍 LLM AUDIT VERDICT   ")
    print("==========================")
    print(result)
    print("==========================\n")

