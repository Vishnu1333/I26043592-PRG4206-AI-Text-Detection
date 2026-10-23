"""
app/ui_utils.py — Helper utilities, metrics loaders, and presets for the Streamlit UI.
PRG4206 | AI-Generated Text Detection System
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.config import (
    BERT_DIR,
    ERROR_ANALYSIS_DIR,
    LR_DIR,
    METRICS_DIR,
    SVM_DIR,
    XGB_DIR,
)

# ── Available model detection (Linear SVM listed first as default) ───────────

_MODEL_FILE_MAP = {
    "Linear SVM":          SVM_DIR / "svm_pipeline.joblib",
    "Logistic Regression": LR_DIR  / "lr_pipeline.joblib",
    "XGBoost":             XGB_DIR / "xgb_model.joblib",
    "DistilBERT":          BERT_DIR / "best_model" / "config.json",
}


def get_available_models() -> List[str]:
    """Return a list of model display names for which saved model files exist.

    Returns:
        List of available model names, with Linear SVM as the preferred default.
    """
    return [name for name, path in _MODEL_FILE_MAP.items() if path.exists()]


# ── Metrics loading ──────────────────────────────────────────────────────────

def load_model_metrics() -> Optional[pd.DataFrame]:
    """Load model comparison table from results/metrics/model_comparison.csv.

    Returns:
        DataFrame with benchmark metrics, or None if unavailable.
    """
    path = METRICS_DIR / "model_comparison.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass

    # Fallback to individual metrics CSVs
    if not METRICS_DIR.exists():
        return None
    frames = []
    for p in sorted(METRICS_DIR.glob("*_metrics.csv")):
        try:
            df = pd.read_csv(p)
            if "model" in df.columns:
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined[~combined["model"].str.contains("val|e[0-9]+", case=False, na=False)]


def load_statistical_tests_df() -> Optional[pd.DataFrame]:
    """Load verified pairwise McNemar test results.

    Returns:
        DataFrame from results/metrics/statistical_tests_verified.csv or None.
    """
    path = METRICS_DIR / "statistical_tests_verified.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def load_confusion_matrices_df() -> Optional[pd.DataFrame]:
    """Load verified confusion matrices table.

    Returns:
        DataFrame from results/metrics/confusion_matrices_verified.csv or None.
    """
    path = METRICS_DIR / "confusion_matrices_verified.csv"
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def load_domain_metrics_map() -> Dict[str, pd.DataFrame]:
    """Load domain metrics breakdown CSVs for each model.

    Returns:
        Dictionary mapping model key to domain metrics DataFrame.
    """
    domain_map = {}
    for p in sorted(ERROR_ANALYSIS_DIR.glob("*_domain_metrics.csv")):
        try:
            key = p.stem.replace("_domain_metrics", "").replace("_", " ").title()
            domain_map[key] = pd.read_csv(p)
        except Exception:
            continue
    return domain_map


# ── Sample text presets for interactive demonstration ────────────────────────

def get_sample_presets() -> Dict[str, str]:
    """Return realistic test presets for demonstration in the UI."""
    return {
        "— Select a preset or type below —": "",
        "Authentic Human Casual (Reddit ELI5)": (
            "I've been working as an auto mechanic for about twelve years now. Honestly, the single "
            "biggest mistake people make is skipping their regular oil changes. They think, 'Oh it's fine, "
            "it's only 2,000 miles past due,' but what actually happens is oil sludge builds up inside the "
            "engine block, starving the camshaft bearings and lifters. Do the basic maintenance, guys! "
            "A thirty-dollar oil change avoids a five-thousand-dollar engine rebuild every time."
        ),
        "Authentic Human Academic Writing": (
            "The phenomenological approach adopted in this qualitative investigation emphasizes the subjective "
            "lived experiences of healthcare practitioners within emergency departments. Semi-structured interviews "
            "were transcribed verbatim and analyzed thematically using Braun and Clarke's reflexive framework. "
            "Analysis revealed three overarching themes: systemic workload compression, emotional fatigue, and "
            "informal coping strategies mediated through peer debriefing."
        ),
        "AI-Generated Explanatory (ChatGPT Structured)": (
            "Artificial intelligence (AI) has significantly transformed computational linguistics and natural "
            "language processing. Firstly, modern deep learning architectures leverage multi-head self-attention "
            "mechanisms to capture complex contextual dependencies across long textual sequences. Furthermore, "
            "automated text generation models can synthesize coherent responses across diverse domains. In conclusion, "
            "rigorous empirical validation protocols and ethical frameworks remain paramount."
        ),
        "AI-Generated Financial Overview (ChatGPT)": (
            "In summary, portfolio diversification is a fundamental risk-management strategy in modern financial theory. "
            "By allocating capital across uncorrelated asset classes, such as equities, fixed-income securities, and "
            "commodities, investors can minimize unsystematic volatility while optimizing risk-adjusted returns over extended "
            "investment horizons. However, thorough due diligence and consistent periodic rebalancing remain essential."
        ),
    }
