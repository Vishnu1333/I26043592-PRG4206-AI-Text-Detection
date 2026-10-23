"""
src/statistical_tests.py — Statistical comparison between models.
PRG4206 | AI-Generated Text Detection System

Implements:
  - McNemar's test for paired classifier comparison
  - Bootstrap confidence intervals for metric estimates
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest

from src.config import METRICS_DIR, ensure_dirs
from src.utils import get_logger

logger = get_logger(__name__)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a: str = "Model A",
    model_b: str = "Model B",
) -> dict:
    """Perform McNemar's test to compare two classifiers.

    McNemar's test checks whether two classifiers have significantly
    different error patterns on the SAME test set.

    H₀: The two classifiers make errors on the same samples
        (no significant difference).
    H₁: The classifiers differ significantly in their error patterns.

    Args:
        y_true:    Ground-truth labels.
        y_pred_a:  Predictions from classifier A.
        y_pred_b:  Predictions from classifier B.
        model_a:   Name of classifier A.
        model_b:   Name of classifier B.

    Returns:
        Dictionary with contingency table, chi2 statistic, p-value,
        and interpretation.
    """
    # Contingency table:
    # b00: both correct, b01: A correct B wrong, b10: A wrong B correct, b11: both wrong
    b00 = int(((y_pred_a == y_true) & (y_pred_b == y_true)).sum())
    b01 = int(((y_pred_a == y_true) & (y_pred_b != y_true)).sum())
    b10 = int(((y_pred_a != y_true) & (y_pred_b == y_true)).sum())
    b11 = int(((y_pred_a != y_true) & (y_pred_b != y_true)).sum())

    # McNemar's test uses discordant cells: b = b01, c = b10
    # b01: Model A correct, Model B wrong
    # b10: Model A wrong, Model B correct
    b = b01
    c = b10
    discordant_total = b + c

    if discordant_total == 0:
        chi2_stat = 0.0
        p_value = 1.0
        exact_p = 1.0
    else:
        # Continuity-corrected chi-square: (|b - c| - 1)^2 / (b + c)
        diff = abs(b - c)
        chi2_stat = float(((diff - 1) ** 2) / discordant_total) if diff > 0 else 0.0
        p_value = float(chi2.sf(chi2_stat, df=1))
        exact_p = float(binomtest(min(b, c), discordant_total, 0.5, alternative="two-sided").pvalue)

    alpha = 0.05
    significant = bool(p_value < alpha)

    result = {
        "test":             "McNemar's Test",
        "model_a":          model_a,
        "model_b":          model_b,
        "contingency":      {"b00": b00, "b01": b01, "b10": b10, "b11": b11},
        "discordant_total": discordant_total,
        "chi2":             round(float(chi2_stat), 4),
        "p_value":          round(float(p_value), 6),
        "exact_p_value":    round(float(exact_p), 6),
        "alpha":            alpha,
        "significant":      significant,
        "interpretation": (
            f"The difference between {model_a} and {model_b} is "
            f"{'statistically significant' if significant else 'NOT statistically significant'} "
            f"at α={alpha} (McNemar χ²={chi2_stat:.4f}, p={p_value:.4f}, exact p={exact_p:.4f})."
        ),
    }

    print(f"\n{'='*55}")
    print(f" McNemar's Test: {model_a} vs {model_b}")
    print(f"{'='*55}")
    print(f"  Both correct (b00) : {b00:,}")
    print(f"  A only corr. (b01) : {b01:,}")
    print(f"  B only corr. (b10) : {b10:,}")
    print(f"  Both wrong   (b11) : {b11:,}")
    print(f"  Discordant total   : {discordant_total:,}")
    print(f"  χ² statistic (corr): {chi2_stat:.4f}")
    print(f"  p-value (asymp)    : {p_value:.6f}")
    print(f"  p-value (exact)    : {exact_p:.6f}")
    print(f"  Significant        : {significant}")
    print(f"  {result['interpretation']}")
    return result


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    model_name: str = "model",
) -> dict:
    """Estimate bootstrap confidence interval for a metric.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        metric_fn:   Function (y_true, y_pred) → float.
        n_bootstrap: Number of bootstrap iterations.
        ci:          Confidence level. Default 0.95 (95% CI).
        model_name:  Name for display.

    Returns:
        Dictionary with point_estimate, lower, upper.
    """
    rng = np.random.RandomState(42)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    scores = np.array(scores)
    alpha  = (1 - ci) / 2
    lower  = float(np.percentile(scores, 100 * alpha))
    upper  = float(np.percentile(scores, 100 * (1 - alpha)))
    point  = float(metric_fn(y_true, y_pred))

    result = {
        "model":          model_name,
        "metric":         metric_fn.__name__,
        "point_estimate": round(point, 4),
        "ci_lower":       round(lower, 4),
        "ci_upper":       round(upper, 4),
        "ci_level":       ci,
        "n_bootstrap":    n_bootstrap,
    }
    logger.info(
        "[%s] %s = %.4f (%.0f%% CI: %.4f – %.4f)",
        model_name,
        metric_fn.__name__,
        point,
        100 * ci,
        lower,
        upper,
    )
    return result


def run_all_comparisons(predictions_dir: Path) -> None:
    """Load all saved predictions and run pairwise McNemar tests.

    Args:
        predictions_dir: Directory containing ``*_predictions.csv`` files.
    """
    ensure_dirs()
    preds = {}
    for p in sorted(predictions_dir.glob("*_predictions.csv")):
        df = pd.read_csv(p)
        model_key = p.stem.replace("_predictions", "")
        preds[model_key] = df

    if len(preds) < 2:
        logger.warning("Need at least 2 prediction files for comparison. Found: %d", len(preds))
        return

    # Verify all predictions are on the same test set (same number of rows)
    sizes = {k: len(v) for k, v in preds.items()}
    if len(set(sizes.values())) > 1:
        logger.warning(
            "Prediction files have different sizes: %s. "
            "McNemar's test requires the same test set.",
            sizes,
        )

    results = []
    keys = list(preds.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ka, kb = keys[i], keys[j]
            dfa, dfb = preds[ka], preds[kb]
            min_len = min(len(dfa), len(dfb))
            true_col = "label" if "label" in dfa.columns else "true_label"
            y_true = dfa[true_col].values[:min_len]
            y_pred_a = dfa["predicted_label"].values[:min_len]
            y_pred_b = dfb["predicted_label"].values[:min_len]
            r = mcnemar_test(y_true, y_pred_a, y_pred_b, ka, kb)
            results.append(r)

    out_path = METRICS_DIR / "mcnemar_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("McNemar results saved → %s", out_path)


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import PREDICTIONS_DIR
    run_all_comparisons(PREDICTIONS_DIR)
    print("\n✅ Statistical comparison complete.")
