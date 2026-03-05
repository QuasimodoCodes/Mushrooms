

### Phase 1: Dataset Preparation and Preprocessing

* **The Action:** Download a **species-specific** image dataset (like `zlatan599/mushroom1`) instead of a simple binary edible/poisonous one.
* **The Goal:** Prepare images that are categorized by their exact scientific names (e.g., *Amanita_phalloides*) so the system can precisely identify them later.
* **The Output:** A clean folder structure of images and a master list of all the mushroom class names your model will learn.

### Phase 2: Classification Model Training for Toxicity

* **The Action:** Train your vision model (YOLO or CNN) using the species-specific dataset.
* **The Goal:** Teach the model to act as a "Face ID" for mushrooms. It learns visual features to output a predicted species name and a **confidence score**.
* **The Safety Catch:** If a mushroom is damaged, missing parts, or occluded by mud, the model will naturally output a *low* confidence score, which flags the system that visual identification is unreliable.

### Phase 3: Multimodal Context Integration (The Background Check)

* **The Action:** Create your "Context Table (CSV)" by feeding your list of species names to an LLM and asking it to generate the ecological rules (toxicity, season, habitat) for each.
* **The Goal:** Build the fusion layer. When the vision model predicts a species name, your Python code immediately looks up that name in the CSV to pull its toxicity type and environmental rules.
* **The Output:** A context-validated prediction that pairs the visual guess with hard ecological facts.

### Phase 4: LLM Audit Layer Development

* **The Action:** Pass the vision prediction, the CSV rules, and the user's real-world environment (their current GPS location and season) into the LLM Audit Layer.
* **The Goal:** Have the LLM perform logical verification and knowledge reasoning to ensure the visual prediction makes sense in the real world (e.g., ensuring a winter mushroom isn't predicted in the summer).
* **The Output:** A human-understandable explanation verifying or questioning the result.

### Phase 5: Risk-Aware Decision Logic Implementation

* **The Action:** Build a safety framework that ingests the vision model's confidence score, the context validation, and the LLM reasoning.
* **The Goal:** Provide a final safety assessment. If the confidence score is low (because the mushroom was damaged) or the context doesn't match, the system overrides the label and focuses purely on safety.
* **The Output:** A clear risk level, safety warning, and actionable recommendation (e.g., "Confidence is too low due to visual damage; do not ingest").

### Phase 6: End-to-End System Integration and Deployment

* **The Action:** Connect all these distinct modules (Vision -> CSV Lookup -> LLM Audit -> Decision Logic) into one continuous pipeline.
* **The Goal:** Deploy the final, working safety-aware application.

