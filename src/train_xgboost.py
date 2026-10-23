"""
src/train_xgboost.py — XGBoost model with stylometric features.
PRG4206 | AI-Generated Text Detection System

Model 3: XGBoost + engineered stylometric features
  - Does NOT use TF-IDF text features.
  - Instead uses a rich set of linguistic/surface-level features.
  - This investigates whether AI text can be identified from writing STYLE
    alone, independently of vocabulary.
"""

import json
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.config import (
    CV_FOLDS,
    RANDOM_STATE,
    XGB_DIR,
    XGB_PARAM_GRID,
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
from src.features import extract_stylometric_matrix
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def train_xgboost(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_col: str = "raw_text",
    label_col:   str = "label",
    n_iter:      int = 30,
    param_distributions: Optional[dict] = None,
) -> dict:
    """Train, tune, and evaluate the XGBoost stylometric classifier.

    Process:
    1. Extract stylometric features from train / val / test texts.
    2. Scale features (StandardScaler fitted ONLY on train).
    3. Run RandomizedSearchCV over XGBoost hyperparameters using
       stratified 5-fold CV (training data only — no leakage).
    4. Evaluate best model on val, then test.
    5. Save model, scaler, feature importances, metrics, predictions.

    Args:
        train_df:             Training DataFrame.
        val_df:               Validation DataFrame.
        test_df:              Test DataFrame.
        feature_col:          Text column to use for feature extraction.
                              Default ``"raw_text"`` (preserves punctuation).
        label_col:            Label column name.
        n_iter:               Number of RandomizedSearchCV iterations.
        param_distributions:  Override hyperparameter search space.

    Returns:
        Dictionary with model, best_params, val_metrics, test_metrics.
    """
    if not XGB_AVAILABLE:
        raise RuntimeError(
            "XGBoost is not installed. Run: pip install xgboost"
        )

    set_seed(RANDOM_STATE)
    ensure_dirs()
    logger.info("=== Training XGBoost (Stylometric) ===")

    if param_distributions is None:
        param_distributions = XGB_PARAM_GRID

    # ── Extract stylometric features ──────────────────────────
    logger.info("Extracting stylometric features for train (%d rows) …", len(train_df))
    X_train_raw = extract_stylometric_matrix(train_df[feature_col])
    logger.info("Extracting stylometric features for val   (%d rows) …", len(val_df))
    X_val_raw   = extract_stylometric_matrix(val_df[feature_col])
    logger.info("Extracting stylometric features for test  (%d rows) …", len(test_df))
    X_test_raw  = extract_stylometric_matrix(test_df[feature_col])

    y_train = train_df[label_col].values
    y_val   = val_df[label_col].values
    y_test  = test_df[label_col].values

    # ── Scale features (fit on train only) ───────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    # ── XGBoost with class weighting ─────────────────────────
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    best_model  = search.best_estimator_
    best_params = search.best_params_
    best_cv_f1  = round(search.best_score_, 4)

    logger.info("Best params: %s", best_params)
    logger.info("Best CV F1 : %.4f", best_cv_f1)
    logger.info("Training time: %.1f s", train_time)

    # ── Validation ────────────────────────────────────────────
    y_val_pred = best_model.predict(X_val)
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_prob, "XGBoost-val")
    print_classification_report(y_val, y_val_pred, "XGBoost (val)")

    # ── Test ──────────────────────────────────────────────────
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob, "XGBoost")
    print_classification_report(y_test, y_test_pred, "XGBoost (test)")

    # ── Plots ─────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_test_pred, "XGBoost")
    plot_roc_curve(y_test, y_test_prob, "XGBoost")
    plot_pr_curve(y_test, y_test_prob, "XGBoost")

    # ── Feature importances ───────────────────────────────────
    from src.features import get_stylometric_feature_names
    feature_names = get_stylometric_feature_names()
    importances   = best_model.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    fi_path = XGB_DIR / "feature_importances.csv"
    fi_df.to_csv(fi_path, index=False)
    logger.info("Feature importances saved → %s", fi_path)

    # ── Save model + scaler ───────────────────────────────────
    ensure_dirs()
    joblib.dump(best_model, XGB_DIR / "xgb_model.joblib")
    joblib.dump(scaler,     XGB_DIR / "xgb_scaler.joblib")
    logger.info("XGBoost model saved → %s", XGB_DIR)

    params_out = {
        "model": "XGBoost",
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "train_time_s": train_time,
    }
    with open(XGB_DIR / "best_params.json", "w") as fh:
        json.dump(params_out, fh, indent=2)

    save_metrics(test_metrics, "xgboost")
    save_predictions(test_df, y_test_pred, y_test_prob, "XGBoost")

    return {
        "model":        best_model,
        "scaler":       scaler,
        "best_params":  best_params,
        "cv_f1":        best_cv_f1,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "feature_importances": fi_df,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader import load_hc3, split_dataset
    from src.preprocessing import preprocess_dataframe

    logger.info("Loading and preprocessing HC3 …")
    df = load_hc3()
    df = preprocess_dataframe(df)
    train_df, val_df, test_df = split_dataset(df)

    results = train_xgboost(train_df, val_df, test_df)
    print(f"\n✅ XGBoost complete. Test F1: {results['test_metrics']['f1']}")
    print("\nTop 10 features:")
    print(results["feature_importances"].head(10).to_string(index=False))
