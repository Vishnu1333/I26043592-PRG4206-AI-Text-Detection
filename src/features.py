"""
src/features.py — Feature extraction pipelines.
PRG4206 | AI-Generated Text Detection System

Three feature representations:
  1. TF-IDF (word n-grams + optional char n-grams) — for LR and SVM
  2. Stylometric / engineered features               — for XGBoost
  3. (Transformer tokenisation handled in train_transformer.py)
"""

import re
import string
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.config import (
    TFIDF_MAX_DF,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ── 1. TF-IDF vectoriser factory ────────────────────────────────────────────

def build_tfidf_vectorizer(
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    max_features: int  = TFIDF_MAX_FEATURES,
    min_df:       int  = TFIDF_MIN_DF,
    max_df:       float = TFIDF_MAX_DF,
    sublinear_tf: bool = TFIDF_SUBLINEAR_TF,
    analyzer:     str  = "word",
) -> TfidfVectorizer:
    """Create a TF-IDF vectoriser with project defaults.

    Args:
        ngram_range:  Tuple (min_n, max_n) for n-gram range.
        max_features: Maximum number of features. Limits memory use.
        min_df:       Minimum document frequency.
        max_df:       Maximum document frequency (fraction).
        sublinear_tf: Apply ``1 + log(tf)`` scaling.
        analyzer:     ``"word"`` or ``"char"`` or ``"char_wb"``.

    Returns:
        Configured :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        analyzer=analyzer,
        strip_accents="unicode",
        decode_error="replace",
    )


# ── 2. Stylometric feature extraction ────────────────────────────────────────

# Punctuation sets
_PUNCT_ALL  = set(string.punctuation)
_STOP_WORDS: Optional[set] = None  # loaded lazily


def _load_stopwords() -> set:
    """Lazily load NLTK English stop words."""
    global _STOP_WORDS
    if _STOP_WORDS is None:
        try:
            from nltk.corpus import stopwords
            import nltk
            try:
                _STOP_WORDS = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                nltk.download("punkt", quiet=True)
                nltk.download("averaged_perceptron_tagger", quiet=True)
                _STOP_WORDS = set(stopwords.words("english"))
        except Exception:
            logger.warning("NLTK stopwords unavailable; using empty set.")
            _STOP_WORDS = set()
    return _STOP_WORDS


def extract_stylometric_features(text: str) -> dict:
    """Extract stylometric and surface-level features from a text.

    Features extracted:
        char_count, word_count, sentence_count, avg_word_length,
        avg_sentence_length, type_token_ratio, vocab_richness,
        punct_ratio, comma_ratio, period_ratio,
        question_mark_ratio, exclamation_ratio,
        uppercase_ratio, digit_ratio, whitespace_ratio,
        stop_word_ratio, repeated_word_ratio,
        unique_word_ratio, avg_paragraph_length,
        function_word_ratio.

    All ratios are per-character or per-word as appropriate.

    Args:
        text: Raw (or lightly cleaned) text string.

    Returns:
        Dictionary mapping feature name → float value.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {k: 0.0 for k in _FEATURE_NAMES}

    # ── Character-level ──────────────────────────────────────
    char_count       = len(text)
    uppercase_count  = sum(1 for c in text if c.isupper())
    digit_count      = sum(1 for c in text if c.isdigit())
    whitespace_count = sum(1 for c in text if c.isspace())
    punct_count      = sum(1 for c in text if c in _PUNCT_ALL)
    comma_count      = text.count(",")
    period_count     = text.count(".")
    q_mark_count     = text.count("?")
    excl_count       = text.count("!")

    # ── Word-level ───────────────────────────────────────────
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return {k: 0.0 for k in _FEATURE_NAMES}

    word_lengths      = [len(w) for w in words]
    avg_word_length   = np.mean(word_lengths) if word_lengths else 0.0

    lower_words       = [w.lower().strip(string.punctuation) for w in words]
    unique_words      = set(lower_words)
    type_token_ratio  = len(unique_words) / word_count

    stop_words        = _load_stopwords()
    stop_word_count   = sum(1 for w in lower_words if w in stop_words)

    word_freq         = {}
    for w in lower_words:
        word_freq[w] = word_freq.get(w, 0) + 1
    repeated_word_count = sum(1 for c in word_freq.values() if c > 1)

    # Function words (overlap with stop words + determiners etc.)
    function_words    = stop_words
    function_word_count = sum(1 for w in lower_words if w in function_words)

    # Vocabulary richness (unique / sqrt(2 * word_count)) — Guiraud's R
    vocab_richness    = len(unique_words) / max(1, np.sqrt(2 * word_count))

    # ── Sentence-level ───────────────────────────────────────
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count    = max(1, len(sentences))
    avg_sentence_length = word_count / sentence_count

    # ── Paragraph-level ──────────────────────────────────────
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraph_count   = max(1, len(paragraphs))
    avg_paragraph_length = word_count / paragraph_count

    # ── Assemble feature dict ────────────────────────────────
    features = {
        "char_count":           char_count,
        "word_count":           word_count,
        "sentence_count":       sentence_count,
        "avg_word_length":      avg_word_length,
        "avg_sentence_length":  avg_sentence_length,
        "type_token_ratio":     type_token_ratio,
        "vocab_richness":       vocab_richness,
        "unique_word_ratio":    len(unique_words) / word_count,
        "punct_ratio":          punct_count / char_count,
        "comma_ratio":          comma_count / max(1, word_count),
        "period_ratio":         period_count / max(1, word_count),
        "question_mark_ratio":  q_mark_count / max(1, sentence_count),
        "exclamation_ratio":    excl_count / max(1, sentence_count),
        "uppercase_ratio":      uppercase_count / char_count,
        "digit_ratio":          digit_count / char_count,
        "whitespace_ratio":     whitespace_count / char_count,
        "stop_word_ratio":      stop_word_count / word_count,
        "repeated_word_ratio":  repeated_word_count / max(1, len(unique_words)),
        "function_word_ratio":  function_word_count / word_count,
        "avg_paragraph_length": avg_paragraph_length,
        "paragraph_count":      paragraph_count,
    }
    return features


# Build feature name list from a dummy call (used for zero-fill above)
_FEATURE_NAMES = list(
    extract_stylometric_features("Dummy sentence to get feature names.").keys()
)


def extract_stylometric_matrix(
    texts: pd.Series,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Apply :func:`extract_stylometric_features` to a Series of texts.

    Args:
        texts:         Pandas Series of text strings.
        show_progress: Show a tqdm progress bar. Default ``True``.

    Returns:
        DataFrame where each row corresponds to one text and each column
        is one stylometric feature.
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(texts, desc="Stylometric features", disable=not show_progress)
    except ImportError:
        iterator = texts

    records = [extract_stylometric_features(t) for t in iterator]
    df = pd.DataFrame(records, columns=_FEATURE_NAMES)
    return df


# ── 3. Feature name list ─────────────────────────────────────────────────────

def get_stylometric_feature_names() -> list[str]:
    """Return the ordered list of stylometric feature names.

    Returns:
        List of feature name strings.
    """
    return list(_FEATURE_NAMES)


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_texts = pd.Series(
        [
            "This is a human-written response. It is somewhat informal, and it "
            "uses contractions like it's and doesn't. The sentences vary in length!",
            "ChatGPT generates highly structured, well-organized responses. "
            "Each sentence tends to be complete and grammatically correct. "
            "The vocabulary is diverse, and the tone is consistently formal.",
        ]
    )
    feat_df = extract_stylometric_matrix(sample_texts, show_progress=False)
    print("Stylometric feature shape:", feat_df.shape)
    print(feat_df.T.to_string())
    print("\n✅ Feature extraction test passed.")
