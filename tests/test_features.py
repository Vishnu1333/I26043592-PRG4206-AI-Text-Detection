"""
tests/test_features.py — Unit tests for the feature extraction module.
PRG4206 | AI-Generated Text Detection System
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.features import (
    build_tfidf_vectorizer,
    extract_stylometric_features,
    extract_stylometric_matrix,
    get_stylometric_feature_names,
)


class TestTfidfVectorizer:
    def test_builds_without_error(self):
        vec = build_tfidf_vectorizer()
        assert vec is not None

    def test_fits_and_transforms(self):
        # min_df=1 so small corpus works in tests
        vec = build_tfidf_vectorizer(max_features=100, min_df=1)
        corpus = [
            "hello world text",
            "machine learning is fun",
            "AI text detection system",
            "deep learning model training",
            "natural language processing task",
        ]
        X = vec.fit_transform(corpus)
        assert X.shape[0] == 5

    def test_no_fit_leakage_on_test(self):
        """Verify TF-IDF can transform unseen texts after fitting on train."""
        vec = build_tfidf_vectorizer(max_features=50, min_df=1)
        train = [
            "hello world text here",
            "foo bar baz qux",
            "machine learning nlp",
            "text classification model",
        ]
        vec.fit(train)
        test = ["completely different unseen vocabulary here"]
        X = vec.transform(test)
        assert X.shape[0] == 1


class TestStylometricFeatures:
    SAMPLE_AI = (
        "Artificial intelligence has demonstrated remarkable capabilities "
        "across diverse domains. The systematic application of machine learning "
        "techniques has yielded significant improvements in natural language "
        "processing, computer vision, and decision-making systems."
    )
    SAMPLE_HUMAN = (
        "so I was thinking about AI today... it's kind of crazy how fast "
        "things have changed! Like, 5 years ago nobody was talking about "
        "ChatGPT and now it's everywhere. what do you think?"
    )

    def test_returns_dict(self):
        result = extract_stylometric_features(self.SAMPLE_AI)
        assert isinstance(result, dict)

    def test_all_expected_features_present(self):
        result = extract_stylometric_features(self.SAMPLE_AI)
        for name in get_stylometric_feature_names():
            assert name in result, f"Missing feature: {name}"

    def test_word_count_positive(self):
        result = extract_stylometric_features(self.SAMPLE_AI)
        assert result["word_count"] > 0

    def test_empty_text_returns_zeros(self):
        result = extract_stylometric_features("")
        assert all(v == 0.0 for v in result.values())

    def test_matrix_shape(self):
        texts = pd.Series([self.SAMPLE_AI, self.SAMPLE_HUMAN])
        df = extract_stylometric_matrix(texts, show_progress=False)
        assert df.shape == (2, len(get_stylometric_feature_names()))

    def test_no_nan_values(self):
        texts = pd.Series([self.SAMPLE_AI, self.SAMPLE_HUMAN, "Short."])
        df = extract_stylometric_matrix(texts, show_progress=False)
        assert not df.isnull().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
