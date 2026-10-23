"""
tests/test_preprocessing.py — Unit tests for the preprocessing module.
PRG4206 | AI-Generated Text Detection System
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.preprocessing import (
    clean_text,
    normalise_unicode,
    normalise_whitespace,
    preprocess_dataframe,
    remove_html_tags,
)


class TestNormaliseUnicode:
    def test_zero_width_removed(self):
        text = "hello\u200bworld"
        result = normalise_unicode(text)
        assert "\u200b" not in result

    def test_regular_text_unchanged(self):
        text = "Hello, World!"
        assert normalise_unicode(text) == "Hello, World!"


class TestNormaliseWhitespace:
    def test_multiple_spaces_collapsed(self):
        assert normalise_whitespace("hello   world") == "hello world"

    def test_excessive_newlines_collapsed(self):
        result = normalise_whitespace("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_leading_trailing_stripped(self):
        assert normalise_whitespace("  hello  ") == "hello"


class TestRemoveHtmlTags:
    def test_tags_removed(self):
        result = remove_html_tags("<b>bold</b> text")
        assert "<b>" not in result
        assert "bold" in result

    def test_no_html_unchanged(self):
        text = "No HTML here."
        result = remove_html_tags(text)
        assert "No HTML here." in result


class TestCleanText:
    def test_url_replaced(self):
        result = clean_text("Visit https://example.com for more.")
        assert "https://example.com" not in result
        assert "URL" in result

    def test_html_removed(self):
        result = clean_text("<p>Hello</p> World")
        assert "<p>" not in result
        assert "Hello" in result

    def test_punctuation_preserved(self):
        """Punctuation must NOT be stripped — it's a detection signal."""
        result = clean_text("AI models, unlike humans, use commas frequently.")
        assert "," in result

    def test_empty_input_returns_empty(self):
        assert clean_text("") == ""

    def test_none_returns_empty(self):
        assert clean_text(None) == ""  # type: ignore


class TestPreprocessDataframe:
    def test_clean_text_column_added(self):
        df = pd.DataFrame({"text": ["Hello world.", "AI text here."], "label": [0, 1]})
        result = preprocess_dataframe(df)
        assert "clean_text" in result.columns
        assert "raw_text" in result.columns

    def test_empty_rows_dropped(self):
        df = pd.DataFrame({"text": ["Hello.", "", "   "], "label": [0, 1, 0]})
        result = preprocess_dataframe(df)
        assert len(result) == 1

    def test_original_length_preserved(self):
        df = pd.DataFrame(
            {"text": ["Short text.", "Another longer text with more words."], "label": [0, 1]}
        )
        result = preprocess_dataframe(df)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
