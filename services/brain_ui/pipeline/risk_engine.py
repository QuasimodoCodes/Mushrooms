"""
Risk-Aware Decision Logic
==========================
This module contains the hard-coded safety rules that determine the final
verdict for the user. These rules CANNOT be overridden by the LLM — they
are deterministic Python if/else statements that act as the ultimate safety net.

The philosophy: the LLM provides *explanations*, but Python provides *guarantees*.
"""

# ============================================================
# SAFETY THRESHOLDS (Tunable)
# ============================================================
CONFIDENCE_THRESHOLD = 0.70       # Below this, YOLO's guess is considered unreliable
DEADLY_KEYWORDS = ["deadly", "fatal", "death", "highly toxic"]  # Auto-reject if CSV toxicity contains these


def assess_risk(species_name, confidence, context, llm_verdict):
    """
    Takes all the evidence and produces a final safety decision.
    
    Args:
        species_name (str): The species YOLO predicted
        confidence (float): YOLO's confidence score (0.0 to 1.0)
        context (dict): The CSV row for this species (toxicity, habitat, etc.)
        llm_verdict (str): The raw text response from the LLM audit
    
    Returns:
        # dict: A structured decision with 'risk_level', 'recommendation', and 'reasoning'
    """
    
    risk_factors = []
    risk_level = "LOW"  # Start optimistic, escalate based on evidence
    
    toxicity = context.get("toxicity_type", "Unknown").lower()
    
    # ─── RULE 1: Is the species inherently deadly? ───
    # This rule ALWAYS fires regardless of confidence or LLM opinion.
    # If the CSV says it's deadly, the system says DO NOT EAT. Period.
    is_deadly = any(keyword in toxicity for keyword in DEADLY_KEYWORDS)
    if is_deadly:
        risk_level = "CRITICAL"
        risk_factors.append(f"🚨 CRITICAL: '{species_name}' is classified as '{context.get('toxicity_type', 'Unknown')}' in our database.")
    
    # ─── RULE 2: Is YOLO confident enough? ───
    # A model that is less than 70% sure is essentially guessing.
    if confidence < CONFIDENCE_THRESHOLD:
        risk_level = max_risk(risk_level, "HIGH")
        risk_factors.append(f"⚠️ LOW CONFIDENCE: YOLO is only {confidence*100:.1f}% sure. This is below the {CONFIDENCE_THRESHOLD*100:.0f}% safety threshold.")
    
    # ─── RULE 3: Did the LLM flag a mismatch? ───
    # If the LLM said "SUSPICIOUS" or "DANGER", escalate.
    llm_lower = llm_verdict.lower()
    if "suspicious" in llm_lower or "danger" in llm_lower or "unlikely" in llm_lower:
        risk_level = max_risk(risk_level, "HIGH")
        risk_factors.append("⚠️ CONTEXT MISMATCH: The LLM audit found the environmental conditions don't match this species' known habitat/season.")
    
    # ─── RULE 4: Is it toxic but not deadly? ───
    if "toxic" in toxicity and not is_deadly:
        risk_level = max_risk(risk_level, "MODERATE")
        risk_factors.append(f"⚠️ TOXIC: '{species_name}' has some level of toxicity: '{context.get('toxicity_type', 'Unknown')}'.")
    
    # ─── GENERATE RECOMMENDATION ───
    if risk_level == "CRITICAL":
        recommendation = "🚫 DO NOT CONSUME. This species is potentially DEADLY. Seek expert verification immediately."
    elif risk_level == "HIGH":
        recommendation = "🚫 DO NOT CONSUME. Multiple risk factors detected. The identification is unreliable."
    elif risk_level == "MODERATE":
        recommendation = "⚠️ EXERCISE EXTREME CAUTION. Consult a local mycologist before consuming."
    else:
        recommendation = "✅ Identification appears reliable. Always cross-reference with a local expert before consuming any wild mushroom."
    
    return {
        "risk_level": risk_level,
        "recommendation": recommendation,
        "risk_factors": risk_factors,
        "llm_explanation": llm_verdict
    }


def max_risk(current, new):
    """Returns the higher of two risk levels."""
    levels = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    return current if levels.get(current, 0) >= levels.get(new, 0) else new


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=========================================")
    print("   Risk Decision Engine - Test Cases     ")
    print("=========================================")
    
    # Test Case 1: Deadly mushroom, high confidence, LLM says plausible
    print("\n--- TEST 1: Deadly + High Confidence + Plausible ---")
    result = assess_risk(
        "Amanita phalloides", 0.92,
        {"toxicity_type": "Deadly", "habitat": "Oak forests", "season": "Autumn", "region": "Europe"},
        "✅ PLAUSIBLE: The identification matches the environment."
    )
    print(f"Risk Level: {result['risk_level']}")
    print(f"Decision:   {result['recommendation']}")
    for f in result['risk_factors']: print(f"  {f}")
    
    # Test Case 2: Edible mushroom, low confidence
    print("\n--- TEST 2: Edible + Low Confidence ---")
    result = assess_risk(
        "Cantharellus cibarius", 0.35,
        {"toxicity_type": "Edible (Choice)", "habitat": "Forests", "season": "Summer", "region": "Northern Hemisphere"},
        "✅ PLAUSIBLE: Environment matches."
    )
    print(f"Risk Level: {result['risk_level']}")
    print(f"Decision:   {result['recommendation']}")
    for f in result['risk_factors']: print(f"  {f}")
    
    # Test Case 3: Edible, high confidence, everything checks out
    print("\n--- TEST 3: Edible + High Confidence + All Clear ---")
    result = assess_risk(
        "Boletus edulis", 0.95,
        {"toxicity_type": "Edible (Choice)", "habitat": "Conifer forests", "season": "Autumn", "region": "Northern Hemisphere"},
        "✅ PLAUSIBLE: Perfect match."
    )
    print(f"Risk Level: {result['risk_level']}")
    print(f"Decision:   {result['recommendation']}")
    for f in result['risk_factors']: print(f"  {f}")
