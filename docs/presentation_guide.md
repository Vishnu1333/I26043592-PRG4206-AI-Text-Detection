# 7-Minute Presentation Guide & Script
## PRG4206 Individual Assignment: A Comparative and Explainable Machine Learning System for AI-Generated Text Detection

This guide structures the 7-minute academic oral presentation with slide timings, speaking notes, and visual recommendations.

---

### Slide 1: Title & Overview (0:00 – 0:45)
- **Title**: A Comparative and Explainable Machine Learning System for AI-Generated Text Detection
- **Presenter**: Lead ML Engineer / Student Researcher
- **Course**: PRG4206 Machine Learning
- **Core Motivation**:
  - The proliferation of LLMs creates an unprecedented challenge: distinguishing machine-generated fluency from authentic human thought.
  - Academic integrity, digital forensics, and media authenticity require rigorous, open, and explainable detection tools.
- **Key Talking Point**:
  > *"Rather than viewing AI detection as a silver bullet, this project investigates how classical linear models, tree ensembles, and transformers perform under benchmark conditions, while critically unpacking their limitations and explainability."*

---

### Slide 2: Research Questions & Dataset Benchmark (0:45 – 1:30)
- **Primary Benchmark**: HC3 (Human ChatGPT Comparison Corpus, Guo et al., 2023).
- **Verified Dataset Statistics**:
  - 79,331 verified clean samples across 5 domains (`reddit_eli5`, `finance`, `open_qa`, `medicine`, `wiki_csai`).
  - Class distribution: 53,086 Human (66.9%) vs. 26,245 AI (33.1%).
  - Key finding: AI texts exhibit tight length consistency (median 174 words, std 60.8) vs. high human variance (median 84 words, std 164.9).
- **Research Questions**:
  - RQ1: Classification accuracy across models.
  - RQ2: Classical vs. Transformer architecture comparison.
  - RQ3: Stylometric and linguistic feature significance.
  - RQ4: Cross-domain generalizability.

---

### Slide 3: Methodology & Architecture (1:30 – 2:30)
- **Leakage Prevention**:
  - Strict 70% Train (55,531), 15% Val (11,900), 15% Test (11,900) stratified split.
  - All feature transformers (TF-IDF, Scaler) fit exclusively on training data.
- **Four Model Lineup**:
  1. *Baseline*: TF-IDF (50k features, sublinear TF) + Logistic Regression (GridSearch CV).
  2. *Model 2*: TF-IDF + Linear SVM (Platt Calibrated).
  3. *Model 3*: XGBoost on 21 engineered stylometric features (character, sentence, lexical diversity, punctuation ratios).
  4. *Model 4*: DistilBERT transformer sequence classifier.

---

### Slide 4: Live System Demonstration (2:30 – 4:00)
- **Action**: Switch to the Streamlit web application (`streamlit run app/app.py`).
- **Demonstration Steps**:
  1. **Mode A — Single Text Detection**:
     - Select a quick preset (e.g. *Authentic Human Casual* vs. *AI-Generated Explanatory*).
     - Show detector output: class verdict badge, calibrated confidence meter, surface metrics (words, chars, sentences).
     - Review local word-level attributions (green chips for human tokens, red chips for AI transition tokens).
  2. **Mode B — Full Document Analysis**:
     - Upload an academic file (`.txt`, `.pdf`, or `.docx`).
     - Highlight intelligent chunking (preserving paragraph boundaries within configurable 50–250 words).
     - Display document-level estimate, AI chunk ratio %, and chronological chunk-by-chunk probability timeline.
     - Demonstrate passage inspection (identifying most AI-like vs most Human-like passages) and CSV export.
  3. **Performance & XAI Tabs**:
     - Review verified test metrics (SVM 98.61% Acc, 0.9790 F1), McNemar significance card, and SHAP stylometric feature plots.
  4. **Ethical Disclaimer**:
     - Emphasize the prominent ethical disclaimer: probabilistic model estimate, never a sole arbiter for academic integrity penalties.

---

### Slide 5: Quantitative Results & Model Comparison (4:00 – 5:15)
- **Comparison Table**: Review test set Accuracy, Precision, Recall, $F_1$, and ROC-AUC.
- **Key Empirical Observations**:
  - Linear models with TF-IDF achieve exceptionally strong discriminative performance due to distinct vocabulary choices and formulaic transition phrases in ChatGPT responses.
  - XGBoost on stylometry demonstrates that even without vocabulary signals, syntactic regularity alone provides significant discriminative signal.
  - Transformer models capture contextual nuance but require substantially higher compute.
- **Statistical Significance (Verified McNemar Test)**:
  - Linear SVM vs. Logistic Regression: $\chi^2 = 2.13, p = 0.1443$ (no statistically significant difference; models are statistically equivalent).
  - Linear SVM vs. XGBoost: $\chi^2 = 78.56, p < 0.0001$ (statistically significant advantage for TF-IDF).
  - Logistic Regression vs. XGBoost: $\chi^2 = 69.66, p < 0.0001$ (statistically significant advantage for TF-IDF).

---

### Slide 6: Explainability & Error Analysis (5:15 – 6:15)
- **XAI Insights**:
  - What makes text look like AI? Words like *"overall"*, *"furthermore"*, *"in summary"*, structured enumerations, and balanced punctuation ratios.
  - What indicates human writing? Contractions, emotional markers, spelling inconsistencies, and conversational punctuation.
- **Error Taxonomy**:
  - *False Positives* (Human flagged as AI): Occur primarily on formal, highly structured academic writing.
  - *False Negatives* (AI missed as Human): Occur on very short texts (< 20 words) or casual prompts.
- **Domain Breakdown**: Performance variation across medical, financial, and Reddit domains.

---

### Slide 7: Ethical Implications, Limitations & Conclusion (6:15 – 7:00)
- **Crucial Limitations**:
  - Susceptible to paraphrase attacks and minor prompt engineering.
  - Dataset bias: trained primarily on ChatGPT; may underperform on unseen models (Claude, Llama).
- **Ethical Takeaway**:
  - Automated detectors must **never** be used as standalone evidence for disciplinary or academic penalty. Human review and procedural safeguards are imperative.
- **Final Conclusion**:
  - Classical and modern ML models can effectively detect AI text within a known distribution, but transparent explainability and rigorous error bounds are essential for responsible deployment.
