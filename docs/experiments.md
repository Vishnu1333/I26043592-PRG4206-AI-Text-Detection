# Experimental Design and Results Log
## PRG4206 | AI-Generated Text Detection System

### 1. Research Questions Addressed

- **RQ1**: How accurately can machine learning models distinguish human-written text from AI-generated text using HC3?
- **RQ2**: How do classical machine learning models compare with transformer-based models for AI-generated text detection?
- **RQ3**: Which textual and linguistic characteristics contribute most to AI-generated text detection?
- **RQ4**: How does model performance vary across different HC3 domains or categories?
- **RQ5**: What types of text are most difficult for the models to classify correctly?
- **RQ6 (Robustness)**: How robust is the detector under variations in text length and domain shift?

---

### 2. Experimental Matrix

| Experiment ID | Model Name | Feature Representation | Hyperparameter Search Space | Tuning Method | Objective / Purpose |
|---|---|---|---|---|---|
| `EXP-01` | Logistic Regression (Baseline) | Word + Bigram TF-IDF (max 50,000 features, sublinear TF) | $C \in \{0.01, 0.1, 1, 10, 100\}$, `class_weight` $\in \{\text{None}, \text{balanced}\}$, `solver` $\in \{\text{liblinear}, \text{saga}\}$ | 5-Fold Stratified CV Grid Search | Classical NLP linear baseline with probability calibration |
| `EXP-02` | Linear SVM | Word + Bigram TF-IDF (max 50,000 features, sublinear TF) | $C \in \{0.01, 0.1, 1, 10, 100\}$, `class_weight` $\in \{\text{None}, \text{balanced}\}$ | 5-Fold Stratified CV Grid Search + Platt Calibration | Maximum-margin linear classification |
| `EXP-03` | XGBoost (Stylometric) | 21 Engineered Stylometric & Linguistic Features (StandardScaled) | `n_estimators` $\in [100, 500]$, `max_depth` $\in [3, 7]$, `lr` $\in [0.01, 0.3]$, `subsample` $\in [0.7, 1.0]$, `colsample` $\in [0.7, 1.0]$ | 5-Fold Stratified RandomizedSearchCV (30 iterations) | Content-free writing-style classification |
| `EXP-04` | DistilBERT | Full-sequence token embeddings (`distilbert-base-uncased`, max length 256) | `lr` $= 2 \times 10^{-5}$, `batch_size` $= 16$, `epochs` $= 3$, `weight_decay` $= 0.01$ | AdamW with linear warmup schedule & early stopping | Deep contextual transformer sequence classifier |

---

### 3. Quantitative Evaluation Metrics Definitions

All metrics are evaluated strictly on the untouched test partition ($N = 11,900$):

1. **Accuracy**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision (Positive Predictive Value for AI)**:
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity for AI detection)**:
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **$F_1$-Score**:
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Specificity (True Negative Rate for Human text)**:
   $$\text{Specificity} = \frac{TN}{TN + FP}$$
6. **Balanced Accuracy**:
   $$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$
7. **Matthews Correlation Coefficient (MCC)**:
   $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
8. **ROC-AUC & PR-AUC**:
   Area under the Receiver Operating Characteristic and Precision-Recall Curves over continuous confidence threshold sweeps.

---

### 4. Explainable AI (XAI) Protocol

- **Global Explanations**:
  - Linear Model coefficients: Top positive (indicating AI) and negative (indicating Human) lexical n-grams.
  - Tree-based SHAP (`shap.TreeExplainer`): Mean absolute SHAP values across 21 stylometric features on XGBoost.
- **Local Instance Explanations**:
  - LIME (`LimeTextExplainer`): Word perturbation importance for individual test samples, highlighting word-level contributions toward either class.
  - Feature-level attributions mapped directly in the interactive Streamlit user interface.

---

### 5. Error Analysis and Domain-Wise Evaluation Protocol

- Classification into error taxonomy:
  - **False Positives (FP)**: Human-authored text incorrectly flagged as AI-generated (high ethical and academic penalty).
  - **False Negatives (FN)**: AI-generated text incorrectly accepted as human-written.
- Stratified performance breakdown across the five HC3 domains:
  - `reddit_eli5`
  - `finance`
  - `open_qa`
  - `medicine`
  - `wiki_csai`
- Length vs. Error analysis: Investigating error concentration across short ($\le 50$ words) vs. standard length texts.
