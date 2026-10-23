"""
tests/test_inference.py — Unit tests for the inference pipeline.
PRG4206 | AI-Generated Text Detection System
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.inference import get_text_statistics, predict_text, LABEL_MAP


class TestTextStatistics:
    def test_word_count(self):
        stats = get_text_statistics("Hello world, how are you today?")
        assert stats["word_count"] == 6

    def test_char_count(self):
        text = "Hello"
        stats = get_text_statistics(text)
        assert stats["char_count"] == 5

    def test_sentence_count_multiple(self):
        text = "First sentence. Second sentence! Third one?"
        stats = get_text_statistics(text)
        assert stats["sentence_count"] >= 3


class TestPredictTextInput:
    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            predict_text("Hi", "lr")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            predict_text("A" * 100, "nonexistent_model_xyz")

    def test_model_not_found_graceful(self):
        """When model file is absent (e.g. distilbert), FileNotFoundError should be raised."""
        from src.inference import _CACHE
        _CACHE.pop("bert", None)

        with pytest.raises(FileNotFoundError):
            predict_text("This is a test text with enough words for inference.", "distilbert")


class TestLabelMap:
    def test_human_label(self):
        assert LABEL_MAP[0] == "Human-Written"

    def test_ai_label(self):
        assert LABEL_MAP[1] == "AI-Generated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
