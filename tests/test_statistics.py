"""
tests/test_statistics.py — Automated Unit & Regression Tests for Statistical Validation.
PRG4206 | AI-Generated Text Detection System

Tests:
  - Metric calculations (accuracy, precision, recall, f1, mcc, etc.)
  - Confusion matrix totals (TN + FP + FN + TP == N)
  - Prediction alignment (same samples, same ground truth, same row ordering)
  - McNemar contingency table calculation (b00, b01, b10, b11)
  - McNemar statistic calculation (continuity-corrected formula)
  - Discordant total equality: discordant_total == model_a_only + model_b_only
  - p-value mathematical bounds (0 <= p <= 1)
  - Alpha significance decisions (alpha = 0.05)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, chi2

from src.statistical_tests import mcnemar_test
from src.verify_statistics import (
    load_prediction_files,
    verify_alignment,
    recalculate_metrics,
    build_confusion_matrices_table,
    run_pairwise_mcnemar,
    calculate_effect_sizes,
)


@pytest.fixture(scope="module")
def preds():
    return load_prediction_files()


class TestPredictionAlignment:
    """Tests verifying test set sample alignment and class distributions."""

    def test_all_prediction_files_exist(self, preds):
        assert "linear_svm" in preds
        assert "logistic_regression" in preds
        assert "xgboost" in preds

    def test_sample_counts_equal_11900(self, preds):
        for model_name, df in preds.items():
            assert len(df) == 11900, f"{model_name} does not have 11,900 samples!"

    def test_class_distribution_exact(self, preds):
        for model_name, df in preds.items():
            counts = df["label"].value_counts().to_dict()
            assert counts[0] == 7963, f"{model_name} Human count is {counts[0]}, expected 7,963"
            assert counts[1] == 3937, f"{model_name} AI count is {counts[1]}, expected 3,937"

    def test_alignment_integrity(self, preds):
        alignment = verify_alignment(preds)
        assert alignment["all_aligned"] is True
        assert alignment["rows_equal"] is True
        assert alignment["labels_identical"] is True
        assert alignment["texts_identical"] is True
        assert alignment["sources_identical"] is True


class TestMetricCalculations:
    """Tests verifying mathematical accuracy of recalculated classification metrics."""

    def test_metrics_range_bounds(self, preds):
        for model_name, df in preds.items():
            m = recalculate_metrics(df, model_name)
            for metric_key in ["accuracy", "precision", "recall", "f1", "macro_f1",
                               "weighted_f1", "balanced_accuracy", "specificity", "roc_auc", "pr_auc"]:
                assert 0.0 <= m[metric_key] <= 1.0, f"{model_name} {metric_key}={m[metric_key]} out of [0, 1]!"
            assert -1.0 <= m["mcc"] <= 1.0, f"{model_name} mcc={m['mcc']} out of [-1, 1]!"

    def test_synthetic_perfect_and_imperfect_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        df_perf = pd.DataFrame({"label": y_true, "predicted_label": y_pred, "confidence": [0.1, 0.2, 0.8, 0.9]})
        m_perf = recalculate_metrics(df_perf, "perfect")
        assert m_perf["accuracy"] == 1.0
        assert m_perf["f1"] == 1.0
        assert m_perf["mcc"] == 1.0


class TestConfusionMatrices:
    """Tests verifying confusion matrix sum equals test set size."""

    def test_confusion_matrix_totals_match_test_size(self, preds):
        conf_table = build_confusion_matrices_table(preds)
        for _, row in conf_table.iterrows():
            total = row["tn"] + row["fp"] + row["fn"] + row["tp"]
            assert total == 11900
            assert row["actual_human"] == 7963
            assert row["actual_ai"] == 3937
            assert row["total_verified"] is True


class TestMcNemarCalculations:
    """Tests verifying McNemar contingency logic, chi-square formula, and exact p-values."""

    def test_contingency_table_sums_to_sample_size(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        pred_a = np.array([0, 0, 1, 0, 0, 1])  # error at index 3
        pred_b = np.array([0, 1, 1, 1, 0, 1])  # error at index 1
        res = mcnemar_test(y_true, pred_a, pred_b, "A", "B")

        c = res["contingency"]
        assert c["b00"] + c["b01"] + c["b10"] + c["b11"] == len(y_true)
        assert c["b01"] == 1  # A correct, B wrong at index 1
        assert c["b10"] == 1  # A wrong, B correct at index 3
        assert res["discordant_total"] == c["b01"] + c["b10"]
        assert res["discordant_total"] == 2

    def test_mcnemar_exact_formula(self):
        # When b01=24, b10=14:
        # diff = 10, (|diff| - 1)^2 = 81
        # discordant_total = 38
        # chi2 = 81 / 38 = 2.1315789...
        b01, b10 = 24, 14
        discordant_total = b01 + b10
        diff = abs(b01 - b10)
        chi2_calc = ((diff - 1) ** 2) / discordant_total
        p_calc = float(chi2.sf(chi2_calc, df=1))
        exact_p = float(binomtest(min(b01, b10), discordant_total, 0.5, alternative="two-sided").pvalue)

        assert round(chi2_calc, 4) == 2.1316
        assert round(p_calc, 4) == 0.1443
        assert round(exact_p, 4) == 0.1433
        assert p_calc > 0.05, "Linear SVM vs LR must NOT be statistically significant at alpha=0.05!"

    def test_mcnemar_p_value_bounds_and_discordant_equality(self):
        preds = load_prediction_files()
        stat_df, detailed = run_pairwise_mcnemar(preds)

        for _, row in stat_df.iterrows():
            # Check discordant total equality rule
            assert row["discordant_total"] == row["model_a_only_correct"] + row["model_b_only_correct"]

            # Check p-value bounds
            assert 0.0 <= row["p_value"] <= 1.0
            assert 0.0 <= row["exact_p_value"] <= 1.0

            # Check total contingency equals 11,900
            total_pairs = (
                row["both_correct"]
                + row["model_a_only_correct"]
                + row["model_b_only_correct"]
                + row["both_wrong"]
            )
            assert total_pairs == 11900

            # Check alpha significance decision
            if row["p_value"] < row["alpha"]:
                assert row["significant"] is True
                assert row["interpretation"] == "Statistically significant difference"
            else:
                assert row["significant"] is False
                assert row["interpretation"] == "No statistically significant difference"

    def test_linear_svm_vs_lr_is_not_significant(self):
        preds = load_prediction_files()
        stat_df, _ = run_pairwise_mcnemar(preds)
        svm_lr = stat_df[stat_df["comparison"] == "Linear SVM vs Logistic Regression"].iloc[0]
        assert svm_lr["statistic"] == 2.1316
        assert bool(svm_lr["significant"]) is False
        assert svm_lr["interpretation"] == "No statistically significant difference"

    def test_linear_models_vs_xgboost_are_significant(self):
        preds = load_prediction_files()
        stat_df, _ = run_pairwise_mcnemar(preds)
        svm_xgb = stat_df[stat_df["comparison"] == "Linear SVM vs XGBoost"].iloc[0]
        lr_xgb = stat_df[stat_df["comparison"] == "Logistic Regression vs XGBoost"].iloc[0]

        assert bool(svm_xgb["significant"]) is True
        assert svm_xgb["p_value"] < 0.0001
        assert bool(lr_xgb["significant"]) is True
        assert lr_xgb["p_value"] < 0.0001


class TestEffectSizes:
    """Tests verifying effect size calculations."""

    def test_effect_sizes_computation(self):
        preds = load_prediction_files()
        effects = calculate_effect_sizes(preds)
        assert len(effects) == 3
        for eff in effects:
            assert "diff_accuracy" in eff
            assert "diff_f1" in eff
            assert "practical_interpretation" in eff


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
