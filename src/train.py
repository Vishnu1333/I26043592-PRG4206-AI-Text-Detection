"""
src/train.py — Training pipelines for Logistic Regression and Linear SVM.
PRG4206 | AI-Generated Text Detection System

Implements:
  - TF-IDF + Logistic Regression (Model 1)
  - TF-IDF + LinearSVC           (Model 2)

Both models:
  - Fit TF-IDF ONLY on training data (no leakage)
  - Tune via GridSearchCV with stratified 5-fold CV
  - Save best estimator to disk with joblib
  - Evaluate on the held-out test set
"""

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    CV_FOLDS,
    LR_DIR,
    LR_PARAM_GRID,
    RANDOM_STATE,
    SVM_DIR,
    SVM_PARAM_GRID,
    ensure_dirs,
)
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    print_classification_report,
    save_metrics,
    save_predictions,
)
from src.features import build_tfidf_vectorizer
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _get_probs(estimator: Pipeline, X_test) -> Optional[np.ndarray]:
    """Try to extract class-1 probabilities from any sklearn estimator."""
    try:
        return estimator.predict_proba(X_test)[:, 1]
    except AttributeError:
        try:
            return estimator.decision_function(X_test)
        except AttributeError:
            return None


def _record_experiment(
    params: dict,
    meta: dict,
) -> None:
    """Append one experiment row to results/experiments.csv."""
    from src.config import RESULTS_DIR
    ensure_dirs()
    row = {**meta, **params}
    exp_path = RESULTS_DIR / "experiments.csv"
    df_new = pd.DataFrame([row])
    if exp_path.exists():
        df_existing = pd.read_csv(exp_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(exp_path, index=False)


# ── Model 1: Logistic Regression ─────────────────────────────────────────────

def train_logistic_regression(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
    param_grid: Optional[dict] = None,
) -> dict:
    """Train, tune, and evaluate TF-IDF + Logistic Regression.

    Process:
    1. Build TF-IDF + LR pipeline.
    2. GridSearch over ``param_grid`` using stratified 5-fold CV
       (training data only).
    3. Evaluate best model on val and test sets.
    4. Save model, metrics, and predictions.

    Args:
        train_df:   Training DataFrame.
        val_df:     Validation DataFrame.
        test_df:    Test DataFrame (evaluated once, at end).
        text_col:   Column containing text features.
        label_col:  Column containing binary labels.
        param_grid: Override hyperparameter search space.

    Returns:
        Dictionary with ``best_params``, ``val_metrics``, ``test_metrics``.
    """
    set_seed(RANDOM_STATE)
    ensure_dirs()
    logger.info("=== Training Logistic Regression ===")

    if param_grid is None:
        param_grid = LR_PARAM_GRID

    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].values
    X_val   = val_df[text_col].tolist()
    y_val   = val_df[label_col].values
    X_test  = test_df[text_col].tolist()
    y_test  = test_df[label_col].values

    # Build pipeline (TF-IDF fitted only on train)
    pipeline = Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf",   LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    best_model  = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_f1  = round(grid.best_score_, 4)

    logger.info("Best params: %s", best_params)
    logger.info("Best CV F1 : %.4f", best_cv_f1)
    logger.info("Training time: %.1f s", train_time)

    # ── Validation evaluation ─────────────────────────────────
    y_val_pred = best_model.predict(X_val)
    y_val_prob = _get_probs(best_model, X_val)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob, "LR-val")
    print_classification_report(y_val, y_val_pred, "Logistic Regression (val)")

    # ── Test evaluation ───────────────────────────────────────
    y_test_pred = best_model.predict(X_test)
    y_test_prob = _get_probs(best_model, X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, "Logistic Regression")
    print_classification_report(y_test, y_test_pred, "Logistic Regression (test)")

    # ── Plots ─────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_test_pred, "Logistic Regression")
    plot_roc_curve(y_test, y_test_prob, "Logistic Regression")
    plot_pr_curve(y_test, y_test_prob, "Logistic Regression")

    # ── Save model ────────────────────────────────────────────
    ensure_dirs()
    model_path = LR_DIR / "lr_pipeline.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Model saved → %s", model_path)

    # ── Save params ───────────────────────────────────────────
    params_out = {
        "model": "Logistic Regression",
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "train_time_s": train_time,
    }
    with open(LR_DIR / "best_params.json", "w") as fh:
        json.dump(params_out, fh, indent=2)

    # ── Save metrics & predictions ────────────────────────────
    save_metrics(test_metrics, "logistic_regression")
    save_predictions(test_df, y_test_pred, y_test_prob, "Logistic Regression")

    _record_experiment(
        params_out,
        {
            "experiment_id": f"lr_{int(time.time())}",
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "random_seed": RANDOM_STATE,
            **test_metrics,
        },
    )

    return {
        "model":       best_model,
        "best_params": best_params,
        "cv_f1":       best_cv_f1,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


# ── Model 2: Linear SVM ───────────────────────────────────────────────────────

def train_linear_svm(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
    param_grid: Optional[dict] = None,
) -> dict:
    """Train, tune, and evaluate TF-IDF + LinearSVC.

    LinearSVC does not natively produce probabilities; it is wrapped in
    :class:`~sklearn.calibration.CalibratedClassifierCV` so that
    calibrated probability estimates are available for ROC-AUC computation.

    Args:
        train_df:   Training DataFrame.
        val_df:     Validation DataFrame.
        test_df:    Test DataFrame.
        text_col:   Text column name.
        label_col:  Label column name.
        param_grid: Override hyperparameter search space.

    Returns:
        Dictionary with ``best_params``, ``val_metrics``, ``test_metrics``.
    """
    set_seed(RANDOM_STATE)
    ensure_dirs()
    logger.info("=== Training Linear SVM ===")

    if param_grid is None:
        param_grid = SVM_PARAM_GRID

    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].values
    X_val   = val_df[text_col].tolist()
    y_val   = val_df[label_col].values
    X_test  = test_df[text_col].tolist()
    y_test  = test_df[label_col].values

    # Pipeline: TF-IDF → Calibrated LinearSVC
    base_svm = LinearSVC(random_state=RANDOM_STATE, max_iter=5000)
    calibrated_svm = CalibratedClassifierCV(base_svm, cv=3)

    pipeline = Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf",   calibrated_svm),
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    best_model  = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_f1  = round(grid.best_score_, 4)

    logger.info("Best params: %s", best_params)
    logger.info("Best CV F1 : %.4f", best_cv_f1)
    logger.info("Training time: %.1f s", train_time)

    # ── Validation ────────────────────────────────────────────
    y_val_pred = best_model.predict(X_val)
    y_val_prob = _get_probs(best_model, X_val)
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob, "SVM-val")
    print_classification_report(y_val, y_val_pred, "Linear SVM (val)")

    # ── Test ──────────────────────────────────────────────────
    y_test_pred = best_model.predict(X_test)
    y_test_prob = _get_probs(best_model, X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, "Linear SVM")
    print_classification_report(y_test, y_test_pred, "Linear SVM (test)")

    # ── Plots ─────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_test_pred, "Linear SVM")
    plot_roc_curve(y_test, y_test_prob, "Linear SVM")
    plot_pr_curve(y_test, y_test_prob, "Linear SVM")

    # ── Save model ────────────────────────────────────────────
    model_path = SVM_DIR / "svm_pipeline.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Model saved → %s", model_path)

    params_out = {
        "model": "Linear SVM",
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "train_time_s": train_time,
    }
    with open(SVM_DIR / "best_params.json", "w") as fh:
        json.dump(params_out, fh, indent=2)

    save_metrics(test_metrics, "linear_svm")
    save_predictions(test_df, y_test_pred, y_test_prob, "Linear SVM")

    _record_experiment(
        params_out,
        {
            "experiment_id": f"svm_{int(time.time())}",
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "random_seed": RANDOM_STATE,
            **test_metrics,
        },
    )

    return {
        "model":       best_model,
        "best_params": best_params,
        "cv_f1":       best_cv_f1,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


# ── CLI entry points ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from src.data_loader import load_hc3, split_dataset
    from src.preprocessing import preprocess_dataframe

    logger.info("Loading and preprocessing HC3 dataset …")
    df = load_hc3()
    df = preprocess_dataframe(df)
    train_df, val_df, test_df = split_dataset(df)

    mode = sys.argv[1] if len(sys.argv) > 1 else "lr"
    if mode == "lr":
        results = train_logistic_regression(train_df, val_df, test_df)
    elif mode == "svm":
        results = train_linear_svm(train_df, val_df, test_df)
    else:
        logger.error("Unknown mode '%s'. Use 'lr' or 'svm'.", mode)
        sys.exit(1)

    print(f"\n✅ Training complete. Test F1: {results['test_metrics']['f1']}")
