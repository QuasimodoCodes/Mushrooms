"""
LLM Audit Layer
================
This script takes YOLO's prediction + the CSV ecological context + the user's 
location/season and asks an LLM: "Does this identification make sense?"

This is the 'reasoning layer' that catches mistakes the vision model makes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from llm_provider import query_llm

def build_audit_prompt(species_name, confidence, context, user_season, user_location):
    """
    Constructs a carefully formatted prompt that gives the LLM all the evidence
    it needs to verify or reject the YOLO prediction.
    """
    prompt = f"""You are a mycology safety auditor AI. Your job is to verify whether a visual mushroom identification is plausible given the environmental context.

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

Begin your response with either "✅ PLAUSIBLE" or "⚠️ SUSPICIOUS" or "🚨 DANGER"."""

    return prompt


def audit_prediction(species_name, confidence, context, user_season="Unknown", user_location="Unknown"):
    """
    Runs the full audit: builds the prompt, queries the LLM, returns the verdict.
    """
    prompt = build_audit_prompt(species_name, confidence, context, user_season, user_location)
    
    print("\n[AUDIT] Sending data to LLM for verification...")
    response = query_llm(prompt)
    
    return response


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
