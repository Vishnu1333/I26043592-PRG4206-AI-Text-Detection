# AI-Generated Text Detection System
### PRG4206 Individual Assignment

**A Comparative and Explainable Machine Learning System for AI-Generated Text Detection**

---

## Problem Statement

This project develops a machine learning system capable of distinguishing
human-written text from AI-generated text, using the HC3 (Human ChatGPT
Comparison Corpus) benchmark dataset. The system investigates whether
classical ML and transformer-based models can reliably detect AI-generated
text and which textual characteristics contribute most to the classification.

> ⚠️ This is a **research prototype**. AI-text detection can produce false
> positives and false negatives. Results must not be used as definitive proof
> of AI authorship.

---

## Research Questions

| # | Question |
|---|---|
| RQ1 | How accurately can ML models distinguish human vs AI text on HC3? |
| RQ2 | How do classical models compare with transformer-based models? |
| RQ3 | Which textual/linguistic features contribute most to detection? |
| RQ4 | How does performance vary across HC3 domains? |
| RQ5 | What types of text are hardest to classify correctly? |

---

## Dataset

| Item | Detail |
|---|---|
| Name | HC3 — Human ChatGPT Comparison Corpus |
| Source | [HuggingFace: Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) |
| License | CC BY-SA 4.0 |
| Paper | Guo et al. (2023), arXiv:2301.07597 |
| Labels | 0 = Human-Written, 1 = AI-Generated |

---

## Models

| # | Model | Features | Notes |
|---|---|---|---|
| 1 | Logistic Regression | TF-IDF word n-grams | Baseline |
| 2 | Linear SVM | TF-IDF word n-grams | Classical NLP |
| 3 | XGBoost | 21 stylometric features | Writing-style analysis |
| 4 | DistilBERT | Full text (tokenised) | Optional training script only; not included in the verified three-model results or bundled as a trained model |

---

## Project Structure

```
ai-text-detector/
├── data/
│   ├── raw/          ← downloaded HC3 parquet
│   └── processed/    ← flat HC3 (79,331 rows) + train/val/test splits
├── notebooks/        ← 8 executed EDA and analysis notebooks
├── src/
│   ├── config.py         ← all settings & chunking thresholds
│   ├── data_loader.py    ← HC3 download + stratified split
│   ├── preprocessing.py  ← text cleaning
│   ├── features.py       ← TF-IDF + 21 stylometric features
│   ├── train.py          ← LR + SVM training
│   ├── train_xgboost.py  ← XGBoost training
│   ├── train_transformer.py ← DistilBERT fine-tuning
│   ├── evaluate.py       ← metrics + plots
│   ├── inference.py      ← unified prediction API
│   ├── document_parser.py← document extraction (.txt, .pdf, .docx) & chunking
│   ├── explainability.py ← SHAP + LIME
│   ├── error_analysis.py ← FP/FN analysis + domain analysis
│   ├── statistical_tests.py ← paired McNemar test
│   └── verify_statistics.py ← independent statistical verification runner
├── models/           ← saved model files (joblib / HuggingFace)
├── configs/          ← config.yaml + best_params.json
├── results/
│   ├── metrics/      ← verified model comparison & McNemar test artifacts
│   ├── figures/      ← 25 publication-grade generated plots
│   ├── predictions/  ← test-set predictions CSV (N=11,900)
│   └── error_analysis/ ← FP/FN analysis CSV files
├── app/
│   ├── app.py        ← Streamlit multi-page web application
│   ├── components.py ← UI cards, charts, and deep-dive tables
│   └── ui_utils.py   ← presets & model loading helpers
├── tests/            ← 67 automated unit and regression tests
├── docs/             ← academic report data & presentation guides
└── requirements.txt
```

---

## Installation (macOS / Linux)

```bash
# 1. Unzip and navigate to the project directory
cd I26043592_Vishnu_Vardhan_Code

# 2. Create virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## Installation (Windows PowerShell)

```powershell
# 1. Unzip and navigate to project directory
cd I26043592_Vishnu_Vardhan_Code

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## Quick-Start Verification (Pre-Trained Models & Tests)

The three primary trained models (`Linear SVM`, `Logistic Regression`, and `XGBoost`), their preprocessing pipelines, test predictions, and held-out test split (`test.parquet`) are **pre-bundled in this submission package**. You can immediately verify all results without re-training:

```bash
# 1. Run the full automated test suite (67 unit and regression tests)
pytest tests/ -v

# 2. Run the independent statistical verification runner
python -m src.verify_statistics

# 3. Launch the Streamlit web application
streamlit run app/app.py
```

Sample demonstration documents in `.txt`, `.docx`, and `.pdf` formats are provided in `sample_documents/` for testing the document parser and smart chunking feature.

---

## Workflow — Step by Step (Re-Training from Scratch)

If you wish to re-download the HC3 corpus and retrain all models from scratch:

### Step 1: Download and prepare the HC3 dataset
```bash
python -m src.data_loader
```

### Step 2: Run unit tests
```bash
python -m pytest tests/ -v
```

### Step 3: Train Logistic Regression (baseline)
```bash
python -m src.train lr
```

### Step 4: Train Linear SVM
```bash
python -m src.train svm
```

### Step 5: Train XGBoost (stylometric)
```bash
python -m src.train_xgboost
```

### Step 6: Train DistilBERT (requires torch)
```bash
python -m src.train_transformer
```

### Step 7: Run XAI (SHAP/LIME)
```bash
python -m src.explainability
```

### Step 8: Run error analysis
```bash
python -m src.error_analysis
```

### Step 9: Run statistical comparison
```bash
python -m src.statistical_tests
```

### Step 10: Launch Streamlit UI
```bash
streamlit run app/app.py
```

---

## Streamlit UI & Document Analysis Architecture

The web interface features five primary navigation views:

1. **🏠 Home / Detection**: Single-text analysis with real-time surface statistics, class confidence meters, and local TF-IDF word attributions. Includes pre-loaded demonstration presets.
2. **📄 Document Analysis**:
   - **Supported Formats**: Plain Text (`.txt`), Adobe PDF (`.pdf`), Microsoft Word (`.docx`).
   - **Text Extraction**: Multi-encoding fallback for text, full multi-page parsing for PDF, paragraph ordering for DOCX.
   - **Smart Chunking**: Configurable chunk boundaries (`min_chunk_words=50`, `max_chunk_words=250`) that preserve paragraph and sentence integrity while avoiding under-sized or truncated fragments.
   - **Document Aggregation**: Calculates word-weighted AI probability, aggregate confidence, AI chunk percentage, and class determination.
   - **Visualizations**: Interactive donut breakdown chart and chronological chunk-by-chunk probability timeline.
   - **Passage Deep Dive**: Inspects top 3 most AI-like and top 3 most Human-like passages, plus full chunk table with CSV download.
3. **📊 Model Performance**: Verified held-out test metrics table, McNemar significance cards, ROC curves, confusion matrices, and domain-wise breakdowns across 5 HC3 categories.
4. **🔍 Explainability (XAI)**: Global SHAP TreeExplainer plots for 21 stylometric features, linear feature weights, and interactive local explanations.
5. **ℹ️ About / Methodology**: Dataset schema, experimental controls, limitations, and ethical guidelines.

---

## Evaluation Metrics

All models are evaluated on the same held-out test set (15% of HC3):

- **Accuracy**
- **Precision**, **Recall**, **F1-score**
- **Macro-F1**, **Weighted-F1**
- **ROC-AUC**
- **PR-AUC**
- **Matthews Correlation Coefficient (MCC)**
- **Balanced Accuracy**
- **Specificity**

Confusion matrices, ROC curves, and PR curves are saved to `results/figures/`.

---

## XAI

- **SHAP LinearExplainer** — Logistic Regression feature contributions
- **SHAP TreeExplainer** — XGBoost feature importances
- **LIME** — word-level local explanations for any model
- Figures saved to `results/figures/`

---

## Reproducibility

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Python version | 3.11 |
| Train/Val/Test split | 70% / 15% / 15% |
| Stratification | Yes |
| TF-IDF fitted on | Training data only |
| Hyperparameter tuning | Training + validation (never test) |

---

## Limitations

- Detector may perform poorly on adversarially paraphrased AI text.
- Domain shift: models trained on HC3 domains may not generalise to others.
- Short texts (< 50 words) are inherently harder to classify reliably.
- DistilBERT is trained for up to 256 tokens; longer texts are truncated.
- HC3 contains ChatGPT responses only; performance on other LLMs may vary.
- False positives risk unjustly flagging human authors.

---

## Ethical Considerations

This tool is a research prototype and:
- Must NOT be used as the sole basis for academic misconduct decisions.
- May produce false accusations against innocent authors (false positives).
- Performance degrades on domains not represented in HC3.
- Results must be reviewed by a qualified human.
- Privacy: users' input text is processed locally and not stored externally.

---

## Results

> **Note:** All results below are generated from actual experiment runs.
> See `results/metrics/` for complete metric files.

Model comparison table is auto-generated and saved to:
`results/metrics/model_comparison.csv`

---

## References

1. Guo et al. (2023). *How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection.* arXiv:2301.07597
2. Mitchell et al. (2023). *DetectGPT: Zero-Shot Machine-Generated Text Detection.* ICML 2023.
3. Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108
4. Lundberg & Lee (2017). *A unified approach to interpreting model predictions (SHAP).* NeurIPS.
5. Ribeiro et al. (2016). *"Why Should I Trust You?" Explaining the Predictions of Any Classifier (LIME).* KDD.

---

## Future Work

- Ensemble models combining TF-IDF + stylometric + transformer signals.
- Robustness evaluation against paraphrasing and adversarial attacks.
- Extension to non-ChatGPT LLMs (GPT-4, Llama, Gemini).
- Calibration analysis and reliability diagrams.
- Real-time API endpoint wrapping the inference pipeline.


Note: the package includes a DistilBERT training script for optional future experimentation, but the verified metrics and the Colab verification notebook cover the three bundled models only: Logistic Regression, Linear SVM, and XGBoost. No trained DistilBERT artefact is included.

---

## Google Colab Verification and Output PDF

This package includes `notebooks/00_PRG4206_Colab_Run_and_Outputs.ipynb`. Open it in Google Colab and run the cells from top to bottom. When prompted, upload this project ZIP. The notebook installs `requirements-colab.txt`, runs the test suite, independently verifies the saved test-set statistics, checks predictions from Logistic Regression / Linear SVM / XGBoost, and smoke-tests TXT/PDF/DOCX document analysis.

After execution, it writes:

- `outputs/colab_runtime.json` — the timestamped runtime record and actual command/prediction results.
- `outputs/PRG4206_Project_Outputs_Summary.pdf` — a PDF of metrics, figures, paired statistical tests, live inference outputs, document parser checks, and runtime status.
- `I26043592_Vishnu_Vardhan_Code.zip` — a new archive containing the project and generated outputs.

The `outputs/PRG4206_Project_Outputs_Summary.pdf` included in the initial archive is a **pre-run summary from the saved evaluation artefacts**, clearly marked as not yet being proof of a Colab run. Run the notebook to replace it with a report containing the actual Colab runtime record. The notebook verifies saved models; it does not retrain the models or claim a new benchmark score.

See `GITHUB_COLAB_CANVAS_GUIDE.md` for steps to publish the repository and submit the links/files in Canvas.
