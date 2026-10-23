"""
scripts/generate_notebooks.py — Programmatically generate the 8 assignment notebooks.
PRG4206 | AI-Generated Text Detection System
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.17"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }


# ─────────────────────────────────────────────────────────────
# 01_data_exploration.ipynb
# ─────────────────────────────────────────────────────────────
nb01 = make_notebook([
    markdown_cell("""# 01. Exploratory Data Analysis (EDA)
## PRG4206 Individual Assignment: AI-Generated Text Detection System

This notebook presents the exploratory data analysis of the **HC3 (Human ChatGPT Comparison Corpus)** benchmark dataset.

### Objectives:
1. Verify dataset structure, schema, and sample counts
2. Programmatically compute human vs. AI class distribution
3. Analyze text length, word count, and character distributions
4. Inspect domain distributions across Q&A categories
5. Detect missing values, duplicates, and outliers
"""),
    code_cell("""import sys
from pathlib import Path
# Add project root to sys.path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_hc3
from src.config import FIGURES_DIR

sns.set_theme(style="whitegrid", palette="muted")
print("Imports loaded successfully.")
"""),
    markdown_cell("""### 1. Load Dataset
We load the preprocessed flat HC3 dataset or download it via `load_hc3()`."""),
    code_cell("""df = load_hc3()
print(f"Total samples : {len(df):,}")
print(f"Columns       : {list(df.columns)}")
df.head(5)
"""),
    markdown_cell("""### 2. Class Distribution Analysis
Checking balance between human-written (label 0) and AI-generated (label 1)."""),
    code_cell("""class_counts = df["label_name"].value_counts()
print("Class Counts:\\n", class_counts)
print("\\nProportions:\\n", df["label_name"].value_counts(normalize=True))

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(class_counts.index, class_counts.values, color=["#3498db", "#e74c3c"])
ax.bar_label(bars, fmt="%d")
ax.set_title("HC3 Class Distribution (Human vs AI)")
ax.set_ylabel("Number of Samples")
plt.tight_layout()
plt.show()
"""),
    markdown_cell("""### 3. Text Length Distribution
Comparing word count distributions across classes."""),
    code_cell("""df.groupby("label_name")[["word_count", "char_count"]].describe()
"""),
    code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for label, color in [("human", "#3498db"), ("ai", "#e74c3c")]:
    sub = df[df["label_name"] == label]["word_count"]
    axes[0].hist(sub.clip(0, 800), bins=40, alpha=0.6, label=f"{label} (median={sub.median():.0f})", color=color)

axes[0].set_title("Word Count Distribution (Clipped at 800 words)")
axes[0].set_xlabel("Words")
axes[0].legend()

sns.boxplot(data=df, x="label_name", y="word_count", palette={"human": "#3498db", "ai": "#e74c3c"}, showfliers=False, ax=axes[1])
axes[1].set_title("Word Count Boxplot (Outliers Omitted)")
plt.tight_layout()
plt.show()
"""),
    markdown_cell("""### 4. Domain / Source Category Breakdown"""),
    code_cell("""domain_dist = df.groupby(["source", "label_name"]).size().unstack(fill_value=0)
print(domain_dist)

domain_dist.plot(kind="bar", figsize=(10, 5), color=["#e74c3c", "#3498db"])
plt.title("Domain-Wise Breakdown of Human vs AI Samples")
plt.ylabel("Count")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
"""),
    markdown_cell("""### 5. Key Findings
- **Sample Count**: Total clean usable records = 79,331.
- **Class Balance**: 66.9% Human (53,086) vs 33.1% AI (26,245) — a ~2:1 ratio.
- **Length Differences**: AI text exhibits tighter variance (std ~60 words) around a median of 174 words, whereas human texts show high skewness and variance (median 84 words, max >7,000 words).
- **Domain Dominance**: `reddit_eli5` constitutes ~78% of the data; domain-stratified evaluation is crucial.
""")
])

# Write 01_data_exploration.ipynb
with open(NOTEBOOKS_DIR / "01_data_exploration.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb01, f, indent=2)
print("Wrote 01_data_exploration.ipynb")


# ─────────────────────────────────────────────────────────────
# 02_preprocessing.ipynb
# ─────────────────────────────────────────────────────────────
nb02 = make_notebook([
    markdown_cell("""# 02. Text Preprocessing & Feature Engineering
## PRG4206 Individual Assignment: AI-Generated Text Detection System

This notebook demonstrates the preprocessing and feature engineering pipelines designed for AI-generated text detection.

### Core Principles:
1. **Punctuation Preservation**: Punctuation is a vital stylistic signal for AI-generated text; we avoid aggressive stripping.
2. **Dual Representation**:
   - Cleaned representation for TF-IDF n-grams (Unicode normalized, HTML/URL normalized).
   - Raw representation for Stylometric feature extraction (preserves exact case, punctuation ratios, paragraph breaks).
3. **No Data Leakage**: TF-IDF vocabulary, IDF weights, and feature scalers are fitted exclusively on the training partition.
"""),
    code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.preprocessing import clean_text, preprocess_dataframe
from src.features import extract_stylometric_features, build_tfidf_vectorizer, get_stylometric_feature_names
from src.data_loader import load_splits

print("Modules imported successfully.")
"""),
    markdown_cell("""### 1. Test Text Normalization
Demonstrating Unicode normalization, HTML stripping, and whitespace handling."""),
    code_cell("""sample_raw = "  Hello <b>World</b>! Check https://huggingface.co  with  extra   spaces and zero-width\\u200b char. "
sample_clean = clean_text(sample_raw)
print("Original   :", repr(sample_raw))
print("Normalized :", repr(sample_clean))
"""),
    markdown_cell("""### 2. Stylometric Feature Extraction
Extracting 21 fine-grained surface and lexical features on human vs AI samples."""),
    code_cell("""sample_ai = ("The concept of machine learning involves algorithms that improve automatically through "
             "experience. Specifically, deep learning architectures such as convolutional and recurrent neural "
             "networks have advanced computer vision and natural language processing tasks.")

sample_human = ("so basically ML is when you give a computer lots of examples and it figures out the rules itself lol. "
                "like, instead of programming every detail you just show it 1000 pictures of dogs and cats!")

ai_feats = extract_stylometric_features(sample_ai)
human_feats = extract_stylometric_features(sample_human)

feat_comparison = pd.DataFrame({"AI Text": ai_feats, "Human Text": human_feats})
feat_comparison.round(4)
"""),
    markdown_cell("""### 3. Load Verified Splits & Verify Leakage Prevention"""),
    code_cell("""train_df, val_df, test_df = load_splits()
print(f"Train samples : {len(train_df):,}")
print(f"Val samples   : {len(val_df):,}")
print(f"Test samples  : {len(test_df):,}")

# Verify no overlap
overlap = set(train_df['text']).intersection(set(test_df['text']))
print(f"Train/Test text overlap: {len(overlap)} (Must be 0)")
assert len(overlap) == 0, "Data leakage detected!"
""")
])

with open(NOTEBOOKS_DIR / "02_preprocessing.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb02, f, indent=2)
print("Wrote 02_preprocessing.ipynb")


# ─────────────────────────────────────────────────────────────
# 03_baseline_models.ipynb
# ─────────────────────────────────────────────────────────────
nb03 = make_notebook([
    markdown_cell("""# 03. Baseline & Classical Models: Logistic Regression and Linear SVM
## PRG4206 Individual Assignment: AI-Generated Text Detection System

This notebook covers:
1. Baseline Model: **TF-IDF + Logistic Regression**
2. Model 2: **TF-IDF + Linear Support Vector Machine (LinearSVC)**
3. 5-Fold Stratified Cross-Validation on the training partition
4. Hyperparameter tuning via GridSearchCV
5. Evaluation on the held-out test partition
"""),
    code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.config import LR_DIR, SVM_DIR, FIGURES_DIR
from src.data_loader import load_splits
from src.preprocessing import preprocess_dataframe
from src.evaluate import compute_metrics

train_df, val_df, test_df = load_splits()
test_df = preprocess_dataframe(test_df)
print("Data loaded.")
"""),
    markdown_cell("""### 1. Inspect Trained Logistic Regression Model"""),
    code_cell("""lr_model_path = LR_DIR / "lr_pipeline.joblib"
if lr_model_path.exists():
    lr_pipeline = joblib.load(lr_model_path)
    print("Logistic Regression Pipeline:", lr_pipeline)
    
    y_test_pred = lr_pipeline.predict(test_df["clean_text"])
    y_test_prob = lr_pipeline.predict_proba(test_df["clean_text"])[:, 1]
    
    metrics = compute_metrics(test_df["label"].values, y_test_pred, y_test_prob, "Logistic Regression")
    print("Test Metrics:\\n", pd.Series(metrics))
else:
    print("LR model not trained yet. Run 'python -m src.train lr' in terminal.")
"""),
    markdown_cell("""### 2. Inspect Trained Linear SVM Model"""),
    code_cell("""svm_model_path = SVM_DIR / "svm_pipeline.joblib"
if svm_model_path.exists():
    svm_pipeline = joblib.load(svm_model_path)
    print("Linear SVM Pipeline:", svm_pipeline)
    
    y_test_pred = svm_pipeline.predict(test_df["clean_text"])
    y_test_prob = svm_pipeline.predict_proba(test_df["clean_text"])[:, 1]
    
    metrics = compute_metrics(test_df["label"].values, y_test_pred, y_test_prob, "Linear SVM")
    print("Test Metrics:\\n", pd.Series(metrics))
else:
    print("SVM model not trained yet. Run 'python -m src.train svm' in terminal.")
""")
])

with open(NOTEBOOKS_DIR / "03_baseline_models.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb03, f, indent=2)
print("Wrote 03_baseline_models.ipynb")


# ─────────────────────────────────────────────────────────────
# 04_xgboost_stylometric.ipynb
# ─────────────────────────────────────────────────────────────
nb04 = make_notebook([
    markdown_cell("""# 04. XGBoost with Engineered Stylometric Features
## PRG4206 Individual Assignment: AI-Generated Text Detection System

This notebook investigates **Model 3: XGBoost Classifier** trained entirely on 21 engineered stylometric and linguistic features.

### Research Question:
*Can AI-generated text be effectively detected solely through writing style (lexical richness, punctuation patterns, sentence complexity) independent of specific vocabulary / n-grams?*
"""),
    code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import XGB_DIR
from src.data_loader import load_splits
from src.features import extract_stylometric_matrix, get_stylometric_feature_names
from src.evaluate import compute_metrics

print("Imports ready.")
"""),
    markdown_cell("""### 1. Load Trained XGBoost Model & Scaler"""),
    code_cell("""xgb_model_path = XGB_DIR / "xgb_model.joblib"
scaler_path = XGB_DIR / "xgb_scaler.joblib"
fi_path = XGB_DIR / "feature_importances.csv"

if xgb_model_path.exists() and fi_path.exists():
    model = joblib.load(xgb_model_path)
    scaler = joblib.load(scaler_path)
    fi_df = pd.read_csv(fi_path)
    print("Top 10 Feature Importances in XGBoost:")
    print(fi_df.head(10).to_string(index=False))
    
    # Plot top features
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df.head(12), x="importance", y="feature", palette="Blues_r")
    plt.title("Top 12 Stylometric Features in XGBoost Detection")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()
else:
    print("XGBoost model not yet trained. Run 'python -m src.train_xgboost' in terminal.")
""")
])

with open(NOTEBOOKS_DIR / "04_xgboost_stylometric.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb04, f, indent=2)
print("Wrote 04_xgboost_stylometric.ipynb")


# ─────────────────────────────────────────────────────────────
# 05_transformer.ipynb
# ─────────────────────────────────────────────────────────────
nb05 = make_notebook([
    markdown_cell("""# 05. Transformer Architecture: DistilBERT Sequence Classifier
## PRG4206 Individual Assignment: AI-Generated Text Detection System

This notebook documents the implementation and fine-tuning of **Model 4: DistilBERT Sequence Classifier** (`distilbert-base-uncased`).

### Key Highlights:
- Pre-trained 66M parameter distilled transformer model
- Context window: 256 tokens
- AdamW optimizer with linear warmup learning rate schedule
- Evaluated on CPU / Apple Silicon MPS
"""),
    code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import BERT_DIR
import json

history_path = BERT_DIR / "training_history.csv"
params_path = BERT_DIR / "best_params.json"

if history_path.exists():
    history = pd.read_csv(history_path)
    print("DistilBERT Training History:")
    print(history)
else:
    print("DistilBERT training history will be recorded once training commences.")
""")
])

with open(NOTEBOOKS_DIR / "05_transformer.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb05, f, indent=2)
print("Wrote 05_transformer.ipynb")


# ─────────────────────────────────────────────────────────────
# 06_model_comparison.ipynb
# ─────────────────────────────────────────────────────────────
nb06 = make_notebook([
    markdown_cell("""# 06. Quantitative Model Comparison & Statistical Significance
## PRG4206 Individual Assignment: AI-Generated Text Detection System

This notebook synthesizes all experimental evaluations across:
1. Logistic Regression (TF-IDF)
2. Linear SVM (TF-IDF)
3. XGBoost (Stylometric Features)
4. DistilBERT (Contextual Transformer)

Metrics evaluated on identical held-out test partition ($N = 11,900$).
Statistical validation via **McNemar's Test** and **Bootstrap Confidence Intervals**.
"""),
    code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import METRICS_DIR, FIGURES_DIR
from src.evaluate import load_all_metrics

metrics_df = load_all_metrics()
print("Saved Model Metrics:")
metrics_df
"""),
    markdown_cell("""### 1. Comparative Metrics Bar Chart"""),
    code_cell("""if not metrics_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=metrics_df, x="model", y="f1", palette="Blues_d", ax=axes[0])
    axes[0].set_title("F1-Score Comparison on Held-Out Test Set")
    axes[0].set_ylim(0, 1.05)
    
    sns.barplot(data=metrics_df, x="model", y="accuracy", palette="Greens_d", ax=axes[1])
    axes[1].set_title("Accuracy Comparison on Held-Out Test Set")
    axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show()
"""),
    markdown_cell("""### 2. McNemar's Test for Paired Comparison"""),
    code_cell("""mcnemar_path = METRICS_DIR / "mcnemar_results.json"
if mcnemar_path.exists():
    with open(mcnemar_path) as f:
        import json
        mcnemar_res = json.load(f)
    print(json.dumps(mcnemar_res, indent=2))
else:
    print("Run 'python -m src.statistical_tests' to perform pairwise significance tests.")
""")
])

with open(NOTEBOOKS_DIR / "06_model_comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb06, f, indent=2)
print("Wrote 06_model_comparison.ipynb")


# ─────────────────────────────────────────────────────────────
# 07_xai.ipynb
# ─────────────────────────────────────────────────────────────
nb07 = make_notebook([
    markdown_cell("""# 07. Explainable AI (XAI)
## PRG4206 Individual Assignment: AI-Generated Text Detection System

This notebook visualizes and interprets model decisions using:
1. **Global feature coefficients** from linear TF-IDF models
2. **SHAP (SHapley Additive exPlanations)** TreeExplainer on XGBoost
3. **LIME (Local Interpretable Model-agnostic Explanations)** on individual texts
"""),
    code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.config import LR_DIR, XGB_DIR, FIGURES_DIR
from src.explainability import get_lr_feature_weights

lr_model_path = LR_DIR / "lr_pipeline.joblib"
if lr_model_path.exists():
    lr_pipeline = joblib.load(lr_model_path)
    sample_text = ("In conclusion, artificial intelligence represents a transformative technology that requires "
                   "careful ethical considerations, robust governance frameworks, and comprehensive regulation.")
    exps = get_lr_feature_weights(lr_pipeline, sample_text, top_n=10)
    print("Top feature weights for sample text:")
    print(pd.DataFrame(exps))
"""),
    markdown_cell("""### Visualizing Saved SHAP Figures"""),
    code_cell("""from IPython.display import Image, display
shap_figs = list(FIGURES_DIR.glob("*shap*.png"))
for p in shap_figs:
    print("Figure:", p.name)
    display(Image(filename=str(p)))
""")
])

with open(NOTEBOOKS_DIR / "07_xai.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb07, f, indent=2)
print("Wrote 07_xai.ipynb")


# ─────────────────────────────────────────────────────────────
# 08_error_analysis.ipynb
# ─────────────────────────────────────────────────────────────
nb08 = make_notebook([
    markdown_cell("""# 08. Detailed Error Analysis and Domain-Wise Investigation
## PRG4206 Individual Assignment: AI-Generated Text Detection System

This notebook performs quantitative and qualitative error analysis:
1. False Positive (FP) and False Negative (FN) characterization
2. Text length error distribution
3. Domain-stratified accuracy, precision, recall, and F1
4. Qualitative inspection of challenging failure cases
"""),
    code_cell("""import sys
from pathlib import Path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from src.config import ERROR_ANALYSIS_DIR

domain_files = list(ERROR_ANALYSIS_DIR.glob("*_domain_metrics.csv"))
for f in domain_files:
    print("Domain Metrics for", f.stem)
    df_dom = pd.read_csv(f)
    print(df_dom.to_string(index=False))
    print("\\n" + "-"*50 + "\\n")
"""),
    markdown_cell("""### Inspecting Sample Misclassifications"""),
    code_cell("""ea_files = list(ERROR_ANALYSIS_DIR.glob("*_error_analysis.csv"))
if ea_files:
    ea_df = pd.read_csv(ea_files[0])
    fps = ea_df[ea_df["error_type"] == "FP"].head(3)
    fns = ea_df[ea_df["error_type"] == "FN"].head(3)
    
    print("--- SAMPLE FALSE POSITIVES (Human written, flagged as AI) ---")
    for _, r in fps.iterrows():
        print(f"Domain: {r['source']} | Words: {r['word_count']} | Conf: {r['confidence']:.2f}")
        print("Text snippet:", r['text'][:200], "...\\n")
        
    print("--- SAMPLE FALSE NEGATIVES (AI generated, missed as Human) ---")
    for _, r in fns.iterrows():
        print(f"Domain: {r['source']} | Words: {r['word_count']} | Conf: {r['confidence']:.2f}")
        print("Text snippet:", r['text'][:200], "...\\n")
""")
])

with open(NOTEBOOKS_DIR / "08_error_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb08, f, indent=2)
print("Wrote 08_error_analysis.ipynb")

print("All 8 notebooks generated successfully!")
