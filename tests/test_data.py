"""
tests/test_data.py — Unit tests for the data loading module.
PRG4206 | AI-Generated Text Detection System
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.data_loader import _flatten_hc3, split_dataset


class TestFlattenHc3:
    def _make_raw(self):
        return pd.DataFrame(
            [
                {
                    "question": "What is AI?",
                    "human_answers": ["AI stands for artificial intelligence."],
                    "chatgpt_answers": [
                        "Artificial intelligence (AI) refers to the simulation of "
                        "human intelligence processes by computer systems."
                    ],
                    "source": "medicine",
                },
                {
                    "question": "Describe Python.",
                    "human_answers": ["Python is a programming language.", "I love Python!"],
                    "chatgpt_answers": ["Python is a high-level, interpreted language."],
                    "source": "open_qa",
                },
            ]
        )

    def test_output_has_required_columns(self):
        df = _flatten_hc3(self._make_raw())
        for col in ["text", "label", "label_name", "source"]:
            assert col in df.columns

    def test_human_label_is_zero(self):
        df = _flatten_hc3(self._make_raw())
        human = df[df["label_name"] == "human"]
        assert (human["label"] == 0).all()

    def test_ai_label_is_one(self):
        df = _flatten_hc3(self._make_raw())
        ai = df[df["label_name"] == "ai"]
        assert (ai["label"] == 1).all()

    def test_row_count(self):
        """Expect 1+2 human + 1+1 AI = 5 rows."""
        df = _flatten_hc3(self._make_raw())
        assert len(df) == 5

    def test_empty_answers_skipped(self):
        raw = pd.DataFrame(
            [
                {
                    "question": "Q",
                    "human_answers": ["", None],
                    "chatgpt_answers": ["Real AI response."],
                    "source": "test",
                }
            ]
        )
        df = _flatten_hc3(raw)
        assert len(df) == 1
        assert df.iloc[0]["label"] == 1


class TestSplitDataset:
    def _make_flat_df(self, n=200):
        import numpy as np
        np.random.seed(42)
        labels = np.array([0] * (n // 2) + [1] * (n // 2))
        return pd.DataFrame(
            {
                "text":   [f"Sample text number {i}." for i in range(n)],
                "label":  labels,
                "source": ["test"] * n,
            }
        )

    def test_split_sizes_add_up(self):
        df = self._make_flat_df(200)
        train, val, test = split_dataset(df, save_splits=False)
        assert len(train) + len(val) + len(test) == 200

    def test_test_is_stratified(self):
        df = self._make_flat_df(200)
        _, _, test = split_dataset(df, save_splits=False)
        ratio = (test["label"] == 1).mean()
        assert 0.4 < ratio < 0.6, "Test set class ratio should be near 0.5"

    def test_no_overlap(self):
        df = self._make_flat_df(200)
        train, val, test = split_dataset(df, save_splits=False)
        train_texts = set(train["text"])
        val_texts   = set(val["text"])
        test_texts  = set(test["text"])
        assert train_texts.isdisjoint(test_texts), "Train/test overlap detected!"
        assert val_texts.isdisjoint(test_texts),   "Val/test overlap detected!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
