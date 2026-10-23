"""
src/inference.py — Unified inference pipeline for all models.
PRG4206 | AI-Generated Text Detection System

Exposes:
    predict_text(text, model_name) → structured prediction dict
"""

import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BERT_DIR,
    DISTILBERT_MAX_LENGTH,
    DISTILBERT_MODEL_NAME,
    LR_DIR,
    RANDOM_STATE,
    SVM_DIR,
    XGB_DIR,
)
from src.features import extract_stylometric_features, get_stylometric_feature_names
from src.preprocessing import clean_text
from src.utils import get_logger

logger = get_logger(__name__)

# ── Label map ────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Human-Written", 1: "AI-Generated"}


# ── Model loaders (cached) ───────────────────────────────────────────────────

_CACHE: dict = {}


def _load_lr_pipeline():
    if "lr" not in _CACHE:
        path = LR_DIR / "lr_pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Logistic Regression model not found at {path}. "
                "Run: python -m src.train lr"
            )
        _CACHE["lr"] = joblib.load(path)
        logger.info("LR pipeline loaded.")
    return _CACHE["lr"]


def _load_svm_pipeline():
    if "svm" not in _CACHE:
        path = SVM_DIR / "svm_pipeline.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"SVM model not found at {path}. "
                "Run: python -m src.train svm"
            )
        _CACHE["svm"] = joblib.load(path)
        logger.info("SVM pipeline loaded.")
    return _CACHE["svm"]


def _load_xgb():
    if "xgb" not in _CACHE:
        model_path  = XGB_DIR / "xgb_model.joblib"
        scaler_path = XGB_DIR / "xgb_scaler.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at {model_path}. "
                "Run: python -m src.train_xgboost"
            )
        _CACHE["xgb"] = {
            "model":  joblib.load(model_path),
            "scaler": joblib.load(scaler_path),
        }
        logger.info("XGBoost model loaded.")
    return _CACHE["xgb"]


def _load_distilbert():
    if "bert" not in _CACHE:
        model_dir = BERT_DIR / "best_model"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"DistilBERT model not found at {model_dir}. "
                "Run: python -m src.train_transformer"
            )
        try:
            import torch
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            )
            tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
            model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
            model.eval()
            _CACHE["bert"] = {"model": model, "tokenizer": tokenizer}
            logger.info("DistilBERT model loaded.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load DistilBERT: {exc}") from exc
    return _CACHE["bert"]


# ── Text statistics ───────────────────────────────────────────────────────────

def get_text_statistics(text: str) -> dict:
    """Compute simple surface statistics for display in the UI.

    Args:
        text: Input text string.

    Returns:
        Dictionary with word_count, char_count, sentence_count.
    """
    words     = text.split()
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    return {
        "word_count":     len(words),
        "char_count":     len(text),
        "sentence_count": max(1, len(sentences)),
    }


# ── Prediction functions per model ───────────────────────────────────────────

def _predict_lr(text: str) -> dict:
    pipeline = _load_lr_pipeline()
    cleaned  = clean_text(text)
    pred     = int(pipeline.predict([cleaned])[0])
    prob     = pipeline.predict_proba([cleaned])[0][pred]
    return {
        "label":      LABEL_MAP[pred],
        "label_id":   pred,
        "confidence": round(float(prob), 4),
        "model":      "Logistic Regression",
    }


def _predict_svm(text: str) -> dict:
    pipeline = _load_svm_pipeline()
    cleaned  = clean_text(text)
    pred     = int(pipeline.predict([cleaned])[0])
    try:
        prob = float(pipeline.predict_proba([cleaned])[0][pred])
    except AttributeError:
        prob = abs(float(pipeline.decision_function([cleaned])[0]))
    return {
        "label":      LABEL_MAP[pred],
        "label_id":   pred,
        "confidence": round(prob, 4),
        "model":      "Linear SVM",
    }


def _predict_xgb(text: str) -> dict:
    loaded  = _load_xgb()
    model   = loaded["model"]
    scaler  = loaded["scaler"]
    feats   = extract_stylometric_features(text)
    feat_names = get_stylometric_feature_names()
    X = pd.DataFrame([feats])[feat_names]
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    prob = float(model.predict_proba(X_scaled)[0][pred])
    return {
        "label":      LABEL_MAP[pred],
        "label_id":   pred,
        "confidence": round(prob, 4),
        "model":      "XGBoost",
    }


def _predict_bert(text: str) -> dict:
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch not installed.")

    loaded    = _load_distilbert()
    model     = loaded["model"]
    tokenizer = loaded["tokenizer"]
    cleaned   = clean_text(text)
    inputs    = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=DISTILBERT_MAX_LENGTH,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred = int(np.argmax(probs))
    return {
        "label":      LABEL_MAP[pred],
        "label_id":   pred,
        "confidence": round(float(probs[pred]), 4),
        "model":      "DistilBERT",
    }


# ── Public API ───────────────────────────────────────────────────────────────

MODEL_NAME_MAP = {
    "logistic regression": "lr",
    "lr": "lr",
    "linear svm": "svm",
    "svm": "svm",
    "xgboost": "xgb",
    "xgb": "xgb",
    "distilbert": "bert",
    "bert": "bert",
}

_PREDICT_FN = {
    "lr":   _predict_lr,
    "svm":  _predict_svm,
    "xgb":  _predict_xgb,
    "bert": _predict_bert,
}


def predict_text(text: str, model_name: str = "lr") -> dict:
    """Run inference on a raw text string.

    This is the main entry point used by the Streamlit UI and tests.

    Args:
        text:       Raw text to classify.
        model_name: One of ``"lr"``, ``"svm"``, ``"xgb"``, ``"bert"``
                    (or their long-form equivalents).

    Returns:
        Dictionary with keys:
            ``label``      — "Human-Written" or "AI-Generated"
            ``label_id``   — 0 or 1
            ``confidence`` — float in [0, 1]
            ``model``      — display name of the model
            ``text_stats`` — word/char/sentence counts
            ``explanation``— list of (feature, value) pairs (if available)

    Raises:
        ValueError:       If ``model_name`` is not recognised.
        FileNotFoundError:If the requested model file is not found.
        RuntimeError:     If a dependency (e.g., PyTorch) is missing.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        raise ValueError("Input text is too short. Please provide at least 10 characters.")

    key = MODEL_NAME_MAP.get(model_name.lower().strip())
    if key is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {list(MODEL_NAME_MAP.keys())}"
        )

    result = _PREDICT_FN[key](text)
    result["text_stats"] = get_text_statistics(text)
    result["explanation"] = []   # Populated by explainability.py when called

    return result


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sample = (
        "Machine learning models have demonstrated remarkable performance on "
        "a wide variety of natural language processing tasks, including text "
        "classification, sentiment analysis, and question answering."
    )
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "lr"
    try:
        result = predict_text(sample, model_arg)
        print("\n=== Inference Result ===")
        for k, v in result.items():
            print(f"  {k:12s}: {v}")
    except FileNotFoundError as exc:
        print(f"⚠️  Model not trained yet: {exc}")
