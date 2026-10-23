# Methodology Documentation
## PRG4206 | AI-Generated Text Detection System

---

## 1. Overview

This project implements a comparative machine learning system for binary
classification of text as either **human-written** (label 0) or
**AI-generated** (label 1), using the HC3 benchmark dataset.

The methodology follows standard ML research best practices:
- Stratified train/val/test splitting with a fixed random seed
- Hyperparameter tuning only on training + validation data
- Final evaluation on a held-out test set used exactly once

---

## 2. Dataset

**HC3 — Human ChatGPT Comparison Corpus**

- Source: `Hello-SimpleAI/HC3` (HuggingFace Datasets)
- License: CC BY-SA 4.0
- Reference: Guo et al. (2023), arXiv:2301.07597

HC3 contains human-written and ChatGPT-generated responses to questions
across multiple domains (medicine, finance, open_qa, reddit, wiki_csai, etc.).

### Data Preparation
1. Dataset downloaded via HuggingFace `datasets` library.
2. Nested `human_answers` / `chatgpt_answers` lists flattened into one
   row per answer.
3. Empty and duplicate texts removed.
4. Text-length statistics computed.

---

## 3. Data Splitting

**Strategy: 70/15/15 train/val/test split**

- Stratified by class label (0 = human, 1 = AI)
- Fixed random seed: 42
- Test set sealed until final evaluation

TF-IDF and feature scalers are fitted **only on training data**.
Hyperparameter tuning uses training + validation data only.
The test set is evaluated **exactly once** per model.

---

## 4. Models

### 4.1 Logistic Regression + TF-IDF (Baseline)

- TF-IDF (word unigrams + bigrams, max 50,000 features, sublinear TF)
- LogisticRegression (`liblinear` or `saga` solver)
- Tuned via GridSearchCV (5-fold stratified CV)
- Provides calibrated probability outputs

### 4.2 Linear SVM + TF-IDF

- Same TF-IDF representation as LR
- LinearSVC wrapped in CalibratedClassifierCV for probability estimates
- Tuned via GridSearchCV

### 4.3 XGBoost + Stylometric Features

- 21 stylometric features (see `features.py`)
- StandardScaler (fit on train only)
- RandomizedSearchCV tuning (30 iterations)
- Captures writing-style differences independent of vocabulary

### 4.4 DistilBERT Sequence Classifier

- Pre-trained `distilbert-base-uncased`
- Fine-tuned for 3 epochs
- AdamW optimiser with linear warmup scheduler
- Supports CPU and Apple Silicon MPS

---

## 5. Evaluation

All models evaluated on the same held-out test set with:
- Accuracy, Precision, Recall, F1 (binary, macro, weighted)
- ROC-AUC, PR-AUC
- Matthews Correlation Coefficient
- Balanced Accuracy, Specificity

Statistical comparison via McNemar's test.

---

## 6. Explainability

- **SHAP LinearExplainer** for LR (TF-IDF feature contributions)
- **SHAP TreeExplainer** for XGBoost (stylometric feature importances)
- **LIME text explainer** for word-level local explanations
- Token-level analysis for DistilBERT (planned)

---

## 7. Error Analysis

For each model:
1. Predictions labelled as TP / TN / FP / FN
2. FP (human → AI) and FN (AI → human) analysed by:
   - Text length distribution
   - Domain
   - Confidence distribution
3. Domain-wise metrics computed separately

---

## 8. Limitations

| Limitation | Description |
|---|---|
| Domain shift | HC3 covers 6–7 domains; generalisation to others is unknown |
| Short texts | Harder to classify reliably; pattern may not be detectable |
| Single LLM | HC3 only contains ChatGPT; other LLMs may not be detected |
| Paraphrasing | Human-edited AI text is likely to fool all models |
| Calibration | Probabilities may not reflect true AI authorship likelihood |
| Adversarial | Deliberate obfuscation can bypass the detector |
