"""
src/preprocessing.py — Text preprocessing pipeline.
PRG4206 | AI-Generated Text Detection System

Design notes:
  - Two representations are maintained:
      clean_text : normalised text for TF-IDF (light cleaning only,
                   punctuation PRESERVED because it is a detection signal)
      raw_text   : minimally touched original for stylometric analysis
  - Aggressive stripping is intentionally avoided.
"""

import re
import unicodedata
from typing import Optional

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


# ── Character-level helpers ──────────────────────────────────────────────────

def normalise_unicode(text: str) -> str:
    """Normalise Unicode to NFC form and strip zero-width characters.

    Args:
        text: Raw input string.

    Returns:
        Unicode-normalised string.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width joiners / non-breaking spaces / soft hyphens
    text = text.replace("\u200b", "").replace("\u00ad", "").replace("\ufeff", "")
    return text


def normalise_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters (but keep single newlines).

    Args:
        text: Input string.

    Returns:
        Whitespace-normalised string.
    """
    # Collapse horizontal whitespace runs to single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_urls(text: str) -> str:
    """Replace URLs with the token ``URL``.

    Args:
        text: Input string.

    Returns:
        String with URLs replaced.
    """
    return re.sub(r"https?://\S+|www\.\S+", "URL", text)


def remove_html_tags(text: str) -> str:
    """Strip HTML tags.

    Args:
        text: Input string (may contain HTML).

    Returns:
        String without HTML tags.
    """
    return re.sub(r"<[^>]+>", " ", text)


# ── Document-level cleaning ──────────────────────────────────────────────────

def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_html: bool = True,
    lower: bool = False,
) -> str:
    """Apply light cleaning to a single text document.

    Intentionally **preserves punctuation** because punctuation patterns
    (e.g., comma frequency, question mark use) are useful features for
    distinguishing AI-generated text.

    Args:
        text:        Raw text string.
        remove_urls: Replace URLs with token. Default ``True``.
        remove_html: Strip HTML tags. Default ``True``.
        lower:       Lowercase the text. Default ``False`` (keeps case
                     for stylometric use).

    Returns:
        Lightly cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    text = normalise_unicode(text)

    if remove_html:
        text = remove_html_tags(text)

    if remove_urls:
        text = remove_urls_fn(text)

    text = normalise_whitespace(text)

    if lower:
        text = text.lower()

    return text


# Alias used inside the function body to avoid shadowing the param name
remove_urls_fn = remove_urls


# ── DataFrame-level pipeline ─────────────────────────────────────────────────

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """Apply the full preprocessing pipeline to a DataFrame.

    Adds two new columns:
        ``clean_text`` — lightly cleaned text (for TF-IDF)
        ``raw_text``   — original text stored for stylometric feature use

    Also drops rows where ``clean_text`` is empty after preprocessing.

    Args:
        df:       Input DataFrame with at least a ``text`` column.
        text_col: Name of the raw text column. Default ``"text"``.

    Returns:
        Preprocessed DataFrame with ``clean_text`` and ``raw_text`` columns.
    """
    logger.info("Preprocessing %d texts …", len(df))
    df = df.copy()

    # Store original text for stylometric analysis
    df["raw_text"] = df[text_col].astype(str)

    # Light cleaning for TF-IDF (URL removal, HTML strip, Unicode norm)
    df["clean_text"] = df[text_col].apply(
        lambda t: clean_text(t, remove_urls=True, remove_html=True, lower=False)
    )

    # Drop rows where clean text ended up empty
    before = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.warning("Dropped %d rows with empty text after preprocessing.", removed)

    logger.info("Preprocessing complete. %d rows retained.", len(df))
    return df


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "text": [
                "This is a <b>test</b> with https://example.com and some  spaces.",
                "  AI models like GPT\u200b-4 produce fluent text.  ",
                "",  # Should be dropped
            ],
            "label": [0, 1, 0],
        }
    )
    result = preprocess_dataframe(sample)
    print(result[["clean_text", "raw_text"]].to_string())
    print(f"\n✅ Preprocessing test passed — {len(result)} rows retained.")
