"""
src/explainability.py — XAI module: SHAP and LIME explanations.
PRG4206 | AI-Generated Text Detection System

Implements:
  - SHAP explanations for LR / SVM (linear explainer / TreeExplainer)
  - SHAP TreeExplainer for XGBoost
  - LIME text explainer as fallback
  - Token-level attention for DistilBERT (optional)
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, ensure_dirs
from src.utils import get_logger

logger = get_logger(__name__)


# ── SHAP helpers ─────────────────────────────────────────────────────────────

def explain_lr_shap(
    model,               # fitted sklearn Pipeline (TF-IDF + LR)
    texts: list[str],
    n_samples: int = 100,
    max_display: int = 20,
    save_name: str = "lr_shap",
) -> Optional[object]:
    """Generate SHAP explanations for the Logistic Regression pipeline.

    Uses ``shap.LinearExplainer`` on the LR component, transforming texts
    via the fitted TF-IDF vectoriser first.

    Args:
        model:       Fitted sklearn Pipeline containing 'tfidf' and 'clf' steps.
        texts:       List of text strings to explain.
        n_samples:   Number of background samples for SHAP estimation.
        max_display: Number of top features to display.
        save_name:   Filename prefix for saved plots.

    Returns:
        SHAP Explanation object, or ``None`` if SHAP is unavailable.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping SHAP explanation.")
        return None

    ensure_dirs()
    tfidf_step = model.named_steps["tfidf"]
    clf_step   = model.named_steps["clf"]
    feature_names = tfidf_step.get_feature_names_out()

    X_transformed = tfidf_step.transform(texts).toarray()

    # Use background samples for efficiency
    bg = shap.maskers.Independent(X_transformed, max_samples=min(n_samples, len(texts)))
    explainer = shap.LinearExplainer(clf_step, bg)
    shap_values = explainer(X_transformed)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values.values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    path = FIGURES_DIR / f"{save_name}_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("SHAP summary plot saved → %s", path)
    return shap_values


def explain_xgb_shap(
    model,               # fitted XGBClassifier
    scaler,              # fitted StandardScaler
    texts: list[str],
    feature_names: Optional[list[str]] = None,
    max_display: int = 20,
    save_name: str = "xgb_shap",
) -> Optional[object]:
    """Generate SHAP TreeExplainer explanations for the XGBoost model.

    Args:
        model:         Fitted XGBClassifier.
        scaler:        Fitted StandardScaler used during training.
        texts:         List of text strings to explain.
        feature_names: Stylometric feature names.
        max_display:   Number of top features to display.
        save_name:     Filename prefix for saved plots.

    Returns:
        SHAP Explanation object, or ``None`` if SHAP is unavailable.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping XGBoost SHAP.")
        return None

    from src.features import extract_stylometric_matrix, get_stylometric_feature_names

    ensure_dirs()
    if feature_names is None:
        feature_names = get_stylometric_feature_names()

    X_raw = extract_stylometric_matrix(pd.Series(texts), show_progress=False)
    X     = scaler.transform(X_raw)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Summary plot (class 1 = AI)
    shap_vals_class1 = shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_class1,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    path = FIGURES_DIR / f"{save_name}_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("XGBoost SHAP summary saved → %s", path)
    return shap_values


def explain_single_text_lime(
    predict_fn,
    text: str,
    class_names: list[str] = ["Human", "AI"],
    num_features: int = 10,
    save_name: str = "lime_explanation",
) -> Optional[list]:
    """Explain a single text prediction using LIME.

    Args:
        predict_fn:   Function that takes a list of strings and returns
                      a 2D probability array (n_texts × n_classes).
        text:         Text string to explain.
        class_names:  Display names for classes.
        num_features: Number of top features to include.
        save_name:    Filename prefix for saved HTML.

    Returns:
        List of (word, weight) tuples for the predicted class, or ``None``.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        logger.warning("LIME not installed. Skipping LIME explanation.")
        return None

    ensure_dirs()
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text,
        predict_fn,
        num_features=num_features,
        num_samples=500,
    )

    # Save HTML
    html_path = FIGURES_DIR / f"{save_name}.html"
    exp.save_to_file(str(html_path))
    logger.info("LIME explanation saved → %s", html_path)

    # Return as simple list of (word, weight)
    return exp.as_list()


def get_lr_feature_weights(
    model,
    text: str,
    top_n: int = 10,
) -> list[dict]:
    """Return the top-n contributing TF-IDF terms for a single prediction.

    Supports both Logistic Regression and Calibrated Linear SVM pipelines.

    Args:
        model: Fitted pipeline containing 'tfidf' and 'clf' steps.
        text:  Text string to explain.
        top_n: Number of features to return.

    Returns:
        List of dicts with keys ``feature``, ``weight``, ``direction``.
    """
    tfidf = model.named_steps["tfidf"]
    clf   = model.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    X = tfidf.transform([text])

    # Extract coefficients for class 1 (AI)
    if hasattr(clf, "coef_"):
        coef = clf.coef_[0]
    elif hasattr(clf, "calibrated_classifiers_"):
        coefs = [cc.estimator.coef_[0] for cc in clf.calibrated_classifiers_]
        coef = np.mean(coefs, axis=0)
    else:
        logger.warning("Classifier does not expose linear coefficients.")
        return []

    scores = np.array(X.toarray()[0]) * coef
    top_idx = np.argsort(np.abs(scores))[::-1][:top_n]

    results = []
    for idx in top_idx:
        if scores[idx] != 0:
            results.append(
                {
                    "feature":   feature_names[idx],
                    "weight":    round(float(scores[idx]), 4),
                    "direction": "AI" if scores[idx] > 0 else "Human",
                }
            )
    return results


get_linear_feature_weights = get_lr_feature_weights


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import joblib
    from src.config import LR_DIR, XGB_DIR
    from src.data_loader import load_splits

    logger.info("Loading splits and models for XAI demonstration …")
    try:
        train_df, val_df, test_df = load_splits()
        lr_model = joblib.load(LR_DIR / "lr_pipeline.joblib")
        sample_texts = test_df["clean_text"].head(200).tolist()

        logger.info("Generating LR SHAP explanations …")
        explain_lr_shap(lr_model, sample_texts)

        xgb_model  = joblib.load(XGB_DIR / "xgb_model.joblib")
        xgb_scaler = joblib.load(XGB_DIR / "xgb_scaler.joblib")
        logger.info("Generating XGBoost SHAP explanations …")
        explain_xgb_shap(xgb_model, xgb_scaler, test_df["raw_text"].head(200).tolist())

        print("\n✅ XAI complete. Figures saved to results/figures/")
    except FileNotFoundError as exc:
        print(f"⚠️  Models not trained yet: {exc}")
