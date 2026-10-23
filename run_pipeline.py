"""
run_pipeline.py — Complete ML pipeline orchestrator.
PRG4206 | AI-Generated Text Detection System

This script runs the full pipeline from data download through model training
and evaluation. Each phase can be run selectively.

Usage:
    python run_pipeline.py --phase all          # Run everything
    python run_pipeline.py --phase data         # Download + split only
    python run_pipeline.py --phase eda          # EDA only
    python run_pipeline.py --phase lr           # Train LR
    python run_pipeline.py --phase svm          # Train SVM
    python run_pipeline.py --phase xgb          # Train XGBoost
    python run_pipeline.py --phase bert         # Train DistilBERT
    python run_pipeline.py --phase evaluate     # Evaluate all trained models
    python run_pipeline.py --phase xai          # XAI analysis
    python run_pipeline.py --phase errors       # Error analysis
    python run_pipeline.py --phase stats        # Statistical comparison
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PREDICTIONS_DIR, ensure_dirs
from src.utils import get_logger, set_seed

logger = get_logger("pipeline")
RANDOM_STATE = 42


def load_data():
    """Phase: Download HC3 and create splits."""
    from src.data_loader import load_hc3, split_dataset
    from src.preprocessing import preprocess_dataframe

    logger.info("=== PHASE: DATA ===")
    df = load_hc3()
    df = preprocess_dataframe(df)
    train_df, val_df, test_df = split_dataset(df)
    logger.info("Data phase complete.")
    return train_df, val_df, test_df


def load_existing_splits():
    """Load pre-processed splits from disk."""
    from src.data_loader import load_splits
    from src.preprocessing import preprocess_dataframe
    try:
        train_df, val_df, test_df = load_splits()
        train_df = preprocess_dataframe(train_df)
        val_df   = preprocess_dataframe(val_df)
        test_df  = preprocess_dataframe(test_df)
        return train_df, val_df, test_df
    except FileNotFoundError:
        logger.warning("Splits not found. Running data phase first.")
        return load_data()


def run_eda(df=None):
    """Phase: Exploratory Data Analysis."""
    logger.info("=== PHASE: EDA ===")
    from notebooks.eda_script import run_eda as _eda
    if df is None:
        from src.data_loader import load_hc3
        df = load_hc3()
    _eda()
    logger.info("EDA phase complete.")


def run_lr(train_df, val_df, test_df):
    """Phase: Train Logistic Regression."""
    logger.info("=== PHASE: LOGISTIC REGRESSION ===")
    from src.train import train_logistic_regression
    set_seed(RANDOM_STATE)
    return train_logistic_regression(train_df, val_df, test_df)


def run_svm(train_df, val_df, test_df):
    """Phase: Train Linear SVM."""
    logger.info("=== PHASE: LINEAR SVM ===")
    from src.train import train_linear_svm
    set_seed(RANDOM_STATE)
    return train_linear_svm(train_df, val_df, test_df)


def run_xgb(train_df, val_df, test_df):
    """Phase: Train XGBoost."""
    logger.info("=== PHASE: XGBOOST ===")
    from src.train_xgboost import train_xgboost
    set_seed(RANDOM_STATE)
    return train_xgboost(train_df, val_df, test_df)


def run_bert(train_df, val_df, test_df):
    """Phase: Fine-tune DistilBERT."""
    logger.info("=== PHASE: DISTILBERT ===")
    try:
        from src.train_transformer import train_distilbert
        set_seed(RANDOM_STATE)
        return train_distilbert(train_df, val_df, test_df)
    except RuntimeError as exc:
        logger.error("DistilBERT training failed: %s", exc)
        logger.warning("Continuing without DistilBERT. Other model results remain valid.")
        return None


def run_evaluate():
    """Phase: Load metrics and generate comparison table."""
    logger.info("=== PHASE: EVALUATION SUMMARY ===")
    from src.evaluate import load_all_metrics, save_comparison_table, plot_model_comparison

    df = load_all_metrics()
    if df.empty:
        logger.warning("No metrics found. Train models first.")
        return

    save_comparison_table(df)
    print("\n" + "="*60)
    print(" MODEL COMPARISON TABLE (Test Set)")
    print("="*60)
    cols = [c for c in ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
            if c in df.columns]
    print(df[cols].to_string(index=False))

    for metric in ["f1", "roc_auc", "accuracy"]:
        if metric in df.columns:
            plot_model_comparison(df.to_dict("records"), metric=metric)


def run_xai():
    """Phase: Generate XAI explanations."""
    logger.info("=== PHASE: XAI ===")
    import joblib
    from src.config import LR_DIR, XGB_DIR
    from src.data_loader import load_splits
    from src.preprocessing import preprocess_dataframe
    from src.explainability import explain_lr_shap, explain_xgb_shap

    try:
        _, _, test_df = load_splits()
        test_df = preprocess_dataframe(test_df)

        try:
            lr = joblib.load(LR_DIR / "lr_pipeline.joblib")
            explain_lr_shap(lr, test_df["clean_text"].head(200).tolist())
            logger.info("LR SHAP complete.")
        except FileNotFoundError:
            logger.warning("LR model not found. Skipping LR SHAP.")

        try:
            xgb_model  = joblib.load(XGB_DIR / "xgb_model.joblib")
            xgb_scaler = joblib.load(XGB_DIR / "xgb_scaler.joblib")
            explain_xgb_shap(xgb_model, xgb_scaler,
                             test_df["raw_text"].head(200).tolist())
            logger.info("XGBoost SHAP complete.")
        except FileNotFoundError:
            logger.warning("XGBoost model not found. Skipping XGB SHAP.")

    except FileNotFoundError as exc:
        logger.error("Splits not found: %s. Run data phase first.", exc)


def _get_xgb_preds(model, df):
    import joblib
    from src.config import XGB_DIR
    from src.features import extract_stylometric_matrix, get_stylometric_feature_names
    scaler = joblib.load(XGB_DIR / "xgb_scaler.joblib")
    feat_names = get_stylometric_feature_names()
    X_raw = extract_stylometric_matrix(df["raw_text"], show_progress=False)
    X = scaler.transform(X_raw)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs


def run_errors():
    """Phase: Error analysis for all trained models."""
    logger.info("=== PHASE: ERROR ANALYSIS ===")
    import joblib
    import numpy as np
    from src.config import LR_DIR, SVM_DIR, XGB_DIR
    from src.data_loader import load_splits
    from src.preprocessing import preprocess_dataframe
    from src.error_analysis import (classify_errors, summarise_errors,
                                     domain_wise_metrics, plot_error_word_length_dist,
                                     plot_domain_f1_bar)
    from src.features import extract_stylometric_matrix

    try:
        _, _, test_df = load_splits()
        test_df = preprocess_dataframe(test_df)

        for model_path, model_name, get_preds in [
            (LR_DIR / "lr_pipeline.joblib",  "Logistic Regression",
             lambda m, df: (m.predict(df["clean_text"]),
                            m.predict_proba(df["clean_text"])[:, 1])),
            (SVM_DIR / "svm_pipeline.joblib", "Linear SVM",
             lambda m, df: (m.predict(df["clean_text"]),
                            m.predict_proba(df["clean_text"])[:, 1])),
            (XGB_DIR / "xgb_model.joblib", "XGBoost",
             lambda m, df: _get_xgb_preds(m, df)),
        ]:
            if not model_path.exists():
                logger.warning("%s model not found — skipping.", model_name)
                continue
            model = joblib.load(model_path)
            y_pred, y_prob = get_preds(model, test_df)
            ea = classify_errors(test_df, test_df["label"].values, y_pred, y_prob, model_name)
            summarise_errors(ea, model_name)
            dm = domain_wise_metrics(ea, model_name)
            plot_error_word_length_dist(ea, model_name)
            plot_domain_f1_bar(dm, model_name)

    except FileNotFoundError as exc:
        logger.error("Files not found: %s", exc)


def run_stats():
    """Phase: Statistical comparison."""
    logger.info("=== PHASE: STATISTICAL TESTS ===")
    from src.statistical_tests import run_all_comparisons
    run_all_comparisons(PREDICTIONS_DIR)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Text Detector — Pipeline Runner"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "data", "eda", "lr", "svm", "xgb", "bert",
                 "evaluate", "xai", "errors", "stats"],
        default="all",
        help="Pipeline phase to run (default: all)",
    )
    args = parser.parse_args()

    ensure_dirs()
    set_seed(RANDOM_STATE)
    t0 = time.time()

    if args.phase in ("all", "data"):
        train_df, val_df, test_df = load_data()
    else:
        # Try to load from disk for non-data phases
        if args.phase in ("lr", "svm", "xgb", "bert", "eda"):
            train_df, val_df, test_df = load_existing_splits()

    if args.phase in ("all", "eda"):
        run_eda()

    if args.phase in ("all", "lr"):
        run_lr(train_df, val_df, test_df)

    if args.phase in ("all", "svm"):
        run_svm(train_df, val_df, test_df)

    if args.phase in ("all", "xgb"):
        run_xgb(train_df, val_df, test_df)

    if args.phase in ("all", "bert"):
        run_bert(train_df, val_df, test_df)

    if args.phase in ("all", "evaluate"):
        run_evaluate()

    if args.phase in ("all", "xai"):
        run_xai()

    if args.phase in ("all", "errors"):
        run_errors()

    if args.phase in ("all", "stats"):
        run_stats()

    elapsed = round(time.time() - t0, 1)
    logger.info("=== Pipeline complete in %.1f s ===", elapsed)
    print(f"\n✅ Phase '{args.phase}' completed in {elapsed} s.")


if __name__ == "__main__":
    main()
