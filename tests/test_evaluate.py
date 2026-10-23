"""
tests/test_evaluate.py — Unit tests for evaluation metrics and reporting.
PRG4206 | AI-Generated Text Detection System
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.evaluate import compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        
        m = compute_metrics(y_true, y_pred, y_prob, "test_model")
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["mcc"] == 1.0
        assert m["specificity"] == 1.0

    def test_imbalanced_metrics(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.4])
        
        m = compute_metrics(y_true, y_pred, y_prob, "test_model")
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        assert -1.0 <= m["mcc"] <= 1.0
        assert m["roc_auc"] is not None

    def test_none_probabilities_graceful(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        m = compute_metrics(y_true, y_pred, None, "test_model")
        assert m["accuracy"] == 0.75
        assert m["roc_auc"] is None
        assert m["pr_auc"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
