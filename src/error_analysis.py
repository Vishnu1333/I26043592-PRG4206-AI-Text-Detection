"""
src/error_analysis.py — Error analysis and domain-wise evaluation.
PRG4206 | AI-Generated Text Detection System

Identifies:
  - False Positives (human → predicted AI)
  - False Negatives (AI → predicted human)
Analyses errors by domain, text length, and linguistic characteristics.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import ERROR_ANALYSIS_DIR, FIGURES_DIR, ensure_dirs
from src.utils import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── Error classification ──────────────────────────────────────────────────────

def classify_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    model_name: str,
) -> pd.DataFrame:
    """Build an error-analysis DataFrame.

    Columns in output:
        text, true_label, predicted_label, confidence, domain (source),
        error_type (TP / TN / FP / FN), word_count, char_count.

    Args:
        df:         Test-set DataFrame (must include 'text', 'source').
        y_true:     Ground-truth labels.
        y_pred:     Predicted labels.
        y_prob:     Predicted probabilities for class 1 (or None).
        model_name: Model name for output filenames.

    Returns:
        Error-analysis DataFrame.
    """
    ensure_dirs()
    out = df[["text", "source"]].copy().reset_index(drop=True)
    out["true_label"]      = y_true
    out["predicted_label"] = y_pred
    out["confidence"]      = y_prob if y_prob is not None else np.nan

    # Error type
    def _error_type(row):
        t, p = row["true_label"], row["predicted_label"]
        if t == 0 and p == 0:
            return "TN"   # True Negative (human → human)
        if t == 1 and p == 1:
            return "TP"   # True Positive (AI → AI)
        if t == 0 and p == 1:
            return "FP"   # False Positive (human → AI)
        return "FN"        # False Negative (AI → human)

    out["error_type"] = out.apply(_error_type, axis=1)
    out["true_name"]  = out["true_label"].map({0: "human", 1: "ai"})
    out["pred_name"]  = out["predicted_label"].map({0: "human", 1: "ai"})
    out["word_count"] = out["text"].str.split().str.len()
    out["char_count"] = out["text"].str.len()

    fname = model_name.lower().replace(" ", "_")
    save_path = ERROR_ANALYSIS_DIR / f"{fname}_error_analysis.csv"
    out.to_csv(save_path, index=False)
    logger.info("Error analysis CSV saved → %s", save_path)
    return out


# ── Quantitative error summary ────────────────────────────────────────────────

def summarise_errors(ea_df: pd.DataFrame, model_name: str) -> dict:
    """Print and return a summary of error counts and rates.

    Args:
        ea_df:      Error-analysis DataFrame from :func:`classify_errors`.
        model_name: Model name for display.

    Returns:
        Dictionary with FP, FN, FP_rate, FN_rate, etc.
    """
    total = len(ea_df)
    tp = (ea_df["error_type"] == "TP").sum()
    tn = (ea_df["error_type"] == "TN").sum()
    fp = (ea_df["error_type"] == "FP").sum()
    fn = (ea_df["error_type"] == "FN").sum()

    n_human = (ea_df["true_label"] == 0).sum()
    n_ai    = (ea_df["true_label"] == 1).sum()

    summary = {
        "model":       model_name,
        "total":       total,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "FP_rate":     round(fp / max(1, n_human), 4),
        "FN_rate":     round(fn / max(1, n_ai), 4),
    }

    print(f"\n{'='*55}")
    print(f" Error Summary — {model_name}")
    print(f"{'='*55}")
    print(f"  Total samples : {total:,}")
    print(f"  True Pos (TP) : {tp:,}  (AI correctly detected)")
    print(f"  True Neg (TN) : {tn:,}  (Human correctly passed)")
    print(f"  False Pos (FP): {fp:,}  (Human flagged as AI)")
    print(f"  False Neg (FN): {fn:,}  (AI missed as Human)")
    print(f"  FP Rate       : {summary['FP_rate']:.4f}")
    print(f"  FN Rate       : {summary['FN_rate']:.4f}")
    return summary


# ── Domain-wise analysis ──────────────────────────────────────────────────────

def domain_wise_metrics(ea_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Compute per-domain accuracy, precision, recall, F1.

    Args:
        ea_df:      Error-analysis DataFrame.
        model_name: Model name for display and filename.

    Returns:
        DataFrame with per-domain metrics.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    records = []
    for domain, grp in ea_df.groupby("source"):
        y_t = grp["true_label"].values
        y_p = grp["predicted_label"].values
        if len(y_t) < 2:
            continue
        records.append(
            {
                "domain":    domain,
                "n_samples": len(y_t),
                "accuracy":  round(accuracy_score(y_t, y_p), 4),
                "precision": round(precision_score(y_t, y_p, zero_division=0), 4),
                "recall":    round(recall_score(y_t, y_p, zero_division=0), 4),
                "f1":        round(f1_score(y_t, y_p, zero_division=0), 4),
                "fp":        int(((y_t == 0) & (y_p == 1)).sum()),
                "fn":        int(((y_t == 1) & (y_p == 0)).sum()),
            }
        )

    df_domain = pd.DataFrame(records).sort_values("f1")
    fname = model_name.lower().replace(" ", "_")
    save_path = ERROR_ANALYSIS_DIR / f"{fname}_domain_metrics.csv"
    df_domain.to_csv(save_path, index=False)
    logger.info("Domain metrics saved → %s", save_path)

    print(f"\n{'='*60}")
    print(f" Domain-Wise Metrics — {model_name}")
    print(f"{'='*60}")
    print(df_domain.to_string(index=False))
    return df_domain


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_error_word_length_dist(
    ea_df: pd.DataFrame, model_name: str
) -> Path:
    """Plot word-count distribution for FP and FN errors.

    Args:
        ea_df:      Error-analysis DataFrame.
        model_name: Model name.

    Returns:
        Path to saved figure.
    """
    ensure_dirs()
    errors = ea_df[ea_df["error_type"].isin(["FP", "FN"])]
    if errors.empty:
        logger.warning("No errors to plot for %s.", model_name)
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=errors,
        x="word_count",
        hue="error_type",
        bins=40,
        alpha=0.6,
        kde=True,
        ax=ax,
    )
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error Word-Count Distribution — {model_name}")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    path  = FIGURES_DIR / f"{fname}_error_word_dist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Error word-length distribution saved → %s", path)
    return path


def plot_domain_f1_bar(
    df_domain: pd.DataFrame, model_name: str
) -> Path:
    """Bar chart of per-domain F1 scores.

    Args:
        df_domain:  Domain metrics DataFrame.
        model_name: Model name.

    Returns:
        Path to saved figure.
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        df_domain["domain"],
        df_domain["f1"],
        color=sns.color_palette("muted", len(df_domain)),
    )
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"Domain-Wise F1 — {model_name}")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    path  = FIGURES_DIR / f"{fname}_domain_f1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Domain F1 bar chart saved → %s", path)
    return path


# ── Qualitative sampling ──────────────────────────────────────────────────────

def sample_errors(
    ea_df: pd.DataFrame,
    error_type: str = "FP",
    n: int = 5,
) -> pd.DataFrame:
    """Return a sample of error examples for qualitative review.

    Args:
        ea_df:      Error-analysis DataFrame.
        error_type: ``"FP"`` or ``"FN"``.
        n:          Number of samples.

    Returns:
        Sampled DataFrame with text and metadata.
    """
    subset = ea_df[ea_df["error_type"] == error_type]
    if subset.empty:
        logger.info("No %s errors found.", error_type)
        return pd.DataFrame()
    return subset.sample(min(n, len(subset)), random_state=42)[
        ["text", "true_name", "pred_name", "confidence", "source", "word_count"]
    ]


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import joblib
    from src.config import LR_DIR
    from src.data_loader import load_splits
    from src.preprocessing import preprocess_dataframe

    logger.info("Running error analysis on Logistic Regression predictions …")
    try:
        _, _, test_df = load_splits()
        test_df = preprocess_dataframe(test_df)
        lr = joblib.load(LR_DIR / "lr_pipeline.joblib")
        y_pred = lr.predict(test_df["clean_text"].tolist())
        y_prob = lr.predict_proba(test_df["clean_text"].tolist())[:, 1]

        ea = classify_errors(test_df, test_df["label"].values, y_pred, y_prob, "Logistic Regression")
        summarise_errors(ea, "Logistic Regression")
        dm = domain_wise_metrics(ea, "Logistic Regression")
        plot_error_word_length_dist(ea, "Logistic Regression")
        plot_domain_f1_bar(dm, "Logistic Regression")

        print("\n--- Sample False Positives (human → AI) ---")
        print(sample_errors(ea, "FP").to_string(index=False))
        print("\n--- Sample False Negatives (AI → human) ---")
        print(sample_errors(ea, "FN").to_string(index=False))

        print("\n✅ Error analysis complete.")
    except FileNotFoundError as exc:
        print(f"⚠️  Files not found: {exc}")
