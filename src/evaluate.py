"""
src/evaluate.py — Model evaluation: metrics, plots, and comparison tables.
PRG4206 | AI-Generated Text Detection System
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)

from src.config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, ensure_dirs
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

# Pretty plot style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Metric calculation ───────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "model",
) -> dict:
    """Compute a comprehensive set of binary classification metrics.

    Args:
        y_true:     Ground-truth binary labels (0 / 1).
        y_pred:     Predicted binary labels.
        y_prob:     Predicted probabilities for class 1 (optional).
                    Required for ROC-AUC and PR-AUC.
        model_name: Name used in log messages and saved files.

    Returns:
        Dictionary of metric name → value.
    """
    metrics: dict = {
        "model":             model_name,
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                round(f1_score(y_true, y_pred, zero_division=0), 4),
        "macro_f1":          round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1":       round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "mcc":               round(matthews_corrcoef(y_true, y_pred), 4),
        "specificity":       round(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0), 4
        ),
        "roc_auc":  None,
        "pr_auc":   None,
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
            metrics["pr_auc"]  = round(average_precision_score(y_true, y_prob), 4)
        except Exception as exc:
            logger.warning("Could not compute ROC-AUC/PR-AUC: %s", exc)

    # Log summary
    logger.info(
        "[%s] Acc=%.4f | F1=%.4f | Prec=%.4f | Rec=%.4f | ROC-AUC=%s",
        model_name,
        metrics["accuracy"],
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        metrics["roc_auc"],
    )
    return metrics


def save_metrics(metrics: dict, model_name: str) -> Path:
    """Save metrics dictionary to CSV and JSON.

    Args:
        metrics:    Dictionary of metric name → value.
        model_name: Model identifier used in the filename.

    Returns:
        Path to the saved JSON file.
    """
    ensure_dirs()
    fname = model_name.lower().replace(" ", "_")
    json_path = METRICS_DIR / f"{fname}_metrics.json"
    csv_path  = METRICS_DIR / f"{fname}_metrics.csv"

    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    logger.info("Metrics saved → %s", json_path)
    return json_path


def save_predictions(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    model_name: str,
) -> Path:
    """Persist test-set predictions alongside metadata columns.

    Args:
        df:         Test-set DataFrame (must include ``label``, ``source``).
        y_pred:     Predicted labels.
        y_prob:     Predicted probabilities (or ``None``).
        model_name: Model identifier.

    Returns:
        Path to the saved predictions CSV.
    """
    ensure_dirs()
    out = df[["text", "label", "source"]].copy().reset_index(drop=True)
    out["predicted_label"] = y_pred
    out["confidence"]      = y_prob if y_prob is not None else np.nan
    out["model"]           = model_name
    out["label_name"]      = out["label"].map({0: "human", 1: "ai"})
    out["predicted_name"]  = out["predicted_label"].map({0: "human", 1: "ai"})
    out["correct"]         = (out["label"] == out["predicted_label"]).astype(int)

    fname = model_name.lower().replace(" ", "_")
    csv_path = PREDICTIONS_DIR / f"{fname}_predictions.csv"
    out.to_csv(csv_path, index=False)
    logger.info("Predictions saved → %s", csv_path)
    return csv_path


# ── Plotting helpers ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    labels: list = ["Human", "AI"],
) -> Path:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true:     Ground-truth labels.
        y_pred:     Predicted labels.
        model_name: Model name (used in title and filename).
        labels:     Display labels for classes.

    Returns:
        Path to the saved figure.
    """
    ensure_dir(FIGURES_DIR)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    path  = FIGURES_DIR / f"{fname}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", path)
    return path


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> Optional[Path]:
    """Plot and save an ROC curve.

    Args:
        y_true:     Ground-truth labels.
        y_prob:     Predicted probabilities for class 1.
        model_name: Model name.

    Returns:
        Path to the saved figure, or ``None`` if probabilities are unavailable.
    """
    if y_prob is None:
        logger.warning("ROC curve skipped — no probabilities for %s.", model_name)
        return None
    ensure_dir(FIGURES_DIR)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    path  = FIGURES_DIR / f"{fname}_roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved → %s", path)
    return path


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> Optional[Path]:
    """Plot and save a Precision-Recall curve.

    Args:
        y_true:     Ground-truth labels.
        y_prob:     Predicted probabilities for class 1.
        model_name: Model name.

    Returns:
        Path to the saved figure, or ``None`` if probabilities unavailable.
    """
    if y_prob is None:
        return None
    ensure_dir(FIGURES_DIR)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    path  = FIGURES_DIR / f"{fname}_pr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("PR curve saved → %s", path)
    return path


def plot_model_comparison(
    metrics_list: list[dict],
    metric: str = "f1",
) -> Path:
    """Bar chart comparing all models on a given metric.

    Args:
        metrics_list: List of metric dicts (one per model).
        metric:       Metric column to compare. Default ``"f1"``.

    Returns:
        Path to the saved figure.
    """
    ensure_dir(FIGURES_DIR)
    df = pd.DataFrame(metrics_list)[["model", metric]].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df["model"], df[metric], color=sns.color_palette("muted", len(df)))
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Model Comparison — {metric.upper()}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = FIGURES_DIR / f"model_comparison_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Model comparison chart saved → %s", path)
    return path


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_names: list = ["Human", "AI"],
) -> None:
    """Print a formatted sklearn classification report.

    Args:
        y_true:       Ground-truth labels.
        y_pred:       Predicted labels.
        model_name:   Model name for the header.
        target_names: Class display names.
    """
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(f"\n{'='*55}")
    print(f" Classification Report — {model_name}")
    print(f"{'='*55}")
    print(report)


# ── Aggregate comparison table ────────────────────────────────────────────────

def load_all_metrics() -> pd.DataFrame:
    """Load all saved metrics CSVs into one comparison DataFrame.

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    ensure_dirs()
    frames = []
    for p in sorted(METRICS_DIR.glob("*_metrics.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:
            logger.warning("Could not read %s: %s", p, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_comparison_table(df: pd.DataFrame) -> Path:
    """Save the model comparison table.

    Args:
        df: Comparison DataFrame (output of :func:`load_all_metrics`).

    Returns:
        Path to the saved CSV.
    """
    ensure_dirs()
    path = METRICS_DIR / "model_comparison.csv"
    df.to_csv(path, index=False)
    logger.info("Comparison table saved → %s", path)
    return path
