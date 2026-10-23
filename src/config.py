# ============================================================
# config.py — Central configuration for AI Text Detector
# PRG4206 | AI-Generated Text Detection System
# ============================================================
"""Central configuration module. All paths and hyperparameters
are defined here so nothing is hard-coded in other modules."""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE: int = 42

# ── Project root (resolves relative to this file) ────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Data paths ───────────────────────────────────────────────
DATA_DIR:      Path = PROJECT_ROOT / "data"
RAW_DIR:       Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# ── Model paths ──────────────────────────────────────────────
MODELS_DIR:  Path = PROJECT_ROOT / "models"
LR_DIR:      Path = MODELS_DIR / "logistic_regression"
SVM_DIR:     Path = MODELS_DIR / "svm"
XGB_DIR:     Path = MODELS_DIR / "xgboost"
BERT_DIR:    Path = MODELS_DIR / "distilbert"

# ── Results paths ────────────────────────────────────────────
RESULTS_DIR:       Path = PROJECT_ROOT / "results"
METRICS_DIR:       Path = RESULTS_DIR / "metrics"
FIGURES_DIR:       Path = RESULTS_DIR / "figures"
PREDICTIONS_DIR:   Path = RESULTS_DIR / "predictions"
ERROR_ANALYSIS_DIR:Path = RESULTS_DIR / "error_analysis"

# ── Config paths ─────────────────────────────────────────────
CONFIGS_DIR:      Path = PROJECT_ROOT / "configs"
BEST_PARAMS_PATH: Path = CONFIGS_DIR / "best_params.json"

# ── Dataset ──────────────────────────────────────────────────
HF_DATASET_NAME:   str  = "Hello-SimpleAI/HC3"
HF_DATASET_SUBSET: str  = "all"          # 'all' = full English dataset
HC3_LABEL_MAP:     dict = {0: "human", 1: "ai"}
MAX_SAMPLES: int | None = None           # None = use full dataset

# ── Data split ───────────────────────────────────────────────
TRAIN_SIZE:      float = 0.70
VAL_SIZE:        float = 0.15
TEST_SIZE:       float = 0.15
STRATIFY:        bool  = True

# ── TF-IDF ───────────────────────────────────────────────────
TFIDF_MAX_FEATURES: int   = 50_000
TFIDF_MIN_DF:       int   = 2
TFIDF_MAX_DF:       float = 0.95
TFIDF_SUBLINEAR_TF: bool  = True
TFIDF_NGRAM_RANGE:  tuple = (1, 2)

# ── Logistic Regression search space ─────────────────────────
LR_PARAM_GRID: dict = {
    "clf__C":            [0.01, 0.1, 1, 10, 100],
    "clf__class_weight": [None, "balanced"],
    "clf__solver":       ["liblinear", "saga"],
    "clf__max_iter":     [1000],
}

# ── SVM search space ─────────────────────────────────────────
SVM_PARAM_GRID: dict = {
    "clf__estimator__C":            [0.01, 0.1, 1, 10],
    "clf__estimator__class_weight": [None, "balanced"],
}

# ── XGBoost search space ─────────────────────────────────────
XGB_PARAM_GRID: dict = {
    "n_estimators":      [100, 300, 500],
    "max_depth":         [3, 5, 7],
    "learning_rate":     [0.01, 0.05, 0.1, 0.3],
    "subsample":         [0.7, 0.8, 1.0],
    "colsample_bytree":  [0.7, 0.8, 1.0],
}

# ── DistilBERT ───────────────────────────────────────────────
DISTILBERT_MODEL_NAME: str = "distilbert-base-uncased"
DISTILBERT_MAX_LENGTH: int = 256
DISTILBERT_BATCH_SIZE: int = 16
DISTILBERT_EPOCHS:     int = 3
DISTILBERT_LR:       float = 2e-5
DISTILBERT_WEIGHT_DECAY: float = 0.01

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS: int = 5

# ── Document Analysis & Chunking ─────────────────────────────
DOC_MIN_CHUNK_WORDS: int = 50
DOC_MAX_CHUNK_WORDS: int = 250
DOC_CHUNK_OVERLAP:   int = 0
DOC_MIN_TEXT_WORDS:  int = 10

# ── Ensure directories exist ─────────────────────────────────
def ensure_dirs() -> None:
    """Create all required project directories."""
    for d in [
        RAW_DIR, PROCESSED_DIR,
        LR_DIR, SVM_DIR, XGB_DIR, BERT_DIR,
        METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, ERROR_ANALYSIS_DIR,
        CONFIGS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("✅ All project directories verified.")
    print(f"   PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"   DATA_DIR     : {DATA_DIR}")
    print(f"   MODELS_DIR   : {MODELS_DIR}")
    print(f"   RESULTS_DIR  : {RESULTS_DIR}")
