"""
src/data_loader.py — HC3 dataset acquisition, verification, and splitting.
PRG4206 | AI-Generated Text Detection System

Dataset: HC3 — Human ChatGPT Comparison Corpus
Source:  https://huggingface.co/datasets/Hello-SimpleAI/HC3
License: CC BY-SA 4.0  (verified 2024)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CONFIGS_DIR,
    HF_DATASET_NAME,
    HF_DATASET_SUBSET,
    MAX_SAMPLES,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_DIR,
    STRATIFY,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    ensure_dirs,
)
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)

# ── Dataset metadata ─────────────────────────────────────────────────────────
DATASET_METADATA = {
    "name": "HC3 — Human ChatGPT Comparison Corpus",
    "hf_repo": "Hello-SimpleAI/HC3",
    "subset": "all",
    "license": "CC BY-SA 4.0",
    "paper": "How Close is ChatGPT to Human Experts? "
             "Comparison Corpus, Evaluation, and Detection. "
             "Guo et al., 2023. arXiv:2301.07597",
    "url": "https://huggingface.co/datasets/Hello-SimpleAI/HC3",
    "label_0": "human",
    "label_1": "ai",
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _flatten_hc3(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert HC3's nested list format into one row per answer.

    HC3 stores:
        ``human_answers`` — list of strings (one or more human responses)
        ``chatgpt_answers`` — list of strings (one or more ChatGPT responses)

    This function explodes those lists so every row is a single text with
    a binary label: 0 = human, 1 = ai.

    Args:
        raw_df: DataFrame with columns ``question``, ``human_answers``,
                ``chatgpt_answers``, and optionally ``source``.

    Returns:
        Flat DataFrame with columns:
        ``text``, ``label``, ``source``, ``question``, ``answer_index``.
    """
    records = []
    for _, row in raw_df.iterrows():
        source = row.get("source", "unknown")
        question = row.get("question", "")

        # HuggingFace may return numpy arrays — convert to list safely
        def _to_list(val):
            if val is None:
                return []
            if isinstance(val, (list, tuple)):
                return val
            try:
                return list(val)   # numpy array, etc.
            except Exception:
                return []

        for idx, answer in enumerate(_to_list(row.get("human_answers", []))):
            if isinstance(answer, str) and answer.strip():
                records.append(
                    {
                        "text": answer.strip(),
                        "label": 0,
                        "label_name": "human",
                        "source": source,
                        "question": question,
                        "answer_index": idx,
                    }
                )

        for idx, answer in enumerate(_to_list(row.get("chatgpt_answers", []))):
            if isinstance(answer, str) and answer.strip():
                records.append(
                    {
                        "text": answer.strip(),
                        "label": 1,
                        "label_name": "ai",
                        "source": source,
                        "question": question,
                        "answer_index": idx,
                    }
                )

    return pd.DataFrame(records)


def _dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact-duplicate texts and report statistics.

    Args:
        df: Flat HC3 DataFrame.

    Returns:
        Tuple of (deduplicated DataFrame, dedup statistics dict).
    """
    original_size = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = original_size - len(df)
    stats = {
        "original": original_size,
        "duplicates_removed": removed,
        "after_dedup": len(df),
    }
    if removed > 0:
        logger.warning("Removed %d exact duplicate texts.", removed)
    else:
        logger.info("No exact duplicates found.")
    return df, stats


def _add_text_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used in EDA and feature engineering.

    Columns added: ``char_count``, ``word_count``, ``sentence_count``.

    Args:
        df: DataFrame containing a ``text`` column.

    Returns:
        DataFrame with additional length columns.
    """
    import re

    df = df.copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    # Rough sentence count: split on . ? !
    df["sentence_count"] = df["text"].apply(
        lambda t: max(1, len(re.split(r"[.!?]+", t.strip())))
    )
    return df


# ── Public API ───────────────────────────────────────────────────────────────

def load_hc3(
    force_download: bool = False,
    max_samples: Optional[int] = MAX_SAMPLES,
) -> pd.DataFrame:
    """Load the HC3 dataset from HuggingFace (or from the local cache).

    Procedure:
    1. Attempt to load from ``data/processed/hc3_flat.parquet`` (cached).
    2. If not found (or ``force_download=True``), download from HuggingFace.
    3. Flatten the nested answer lists into one row per answer.
    4. Remove empty texts, deduplicate, add length columns.
    5. Save the processed file to disk for future use.

    Args:
        force_download: If ``True``, skip the local cache and re-download.
        max_samples:    If set, randomly sample this many rows (stratified).
                        Used for quick experimentation. Default ``None`` (all).

    Returns:
        Flat, labelled DataFrame ready for modelling.

    Raises:
        RuntimeError: If the download fails and no local cache is available.
    """
    ensure_dirs()
    cache_path = PROCESSED_DIR / "hc3_flat.parquet"
    meta_path  = PROCESSED_DIR / "hc3_metadata.json"

    if cache_path.exists() and not force_download:
        logger.info("Loading HC3 from local cache: %s", cache_path)
        df = pd.read_parquet(cache_path)
        logger.info("Loaded %d rows from cache.", len(df))
        return _maybe_sample(df, max_samples)

    # ── Download ──────────────────────────────────────────────
    logger.info("Downloading HC3 from HuggingFace: %s …", HF_DATASET_NAME)
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. "
            "Run: pip install datasets"
        ) from exc

    try:
        hf_ds = load_dataset(HF_DATASET_NAME, HF_DATASET_SUBSET)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download HC3 from HuggingFace: {exc}"
        ) from exc

    # HC3 'all' typically has a single 'train' split
    if "train" in hf_ds:
        raw_df = hf_ds["train"].to_pandas()
    else:
        # Fallback: concatenate all available splits
        splits = list(hf_ds.keys())
        logger.warning("Unexpected splits: %s — concatenating all.", splits)
        raw_df = pd.concat(
            [hf_ds[s].to_pandas() for s in splits], ignore_index=True
        )

    logger.info("Raw dataset: %d rows, columns: %s", len(raw_df), list(raw_df.columns))

    # Save raw data
    ensure_dir(RAW_DIR)
    raw_df.to_parquet(RAW_DIR / "hc3_raw.parquet", index=False)
    logger.info("Raw data saved to %s", RAW_DIR / "hc3_raw.parquet")

    # ── Flatten ──────────────────────────────────────────────
    logger.info("Flattening nested answer lists …")
    df = _flatten_hc3(raw_df)
    logger.info("Flat dataset: %d rows", len(df))

    # ── Remove empty texts ───────────────────────────────────
    before = len(df)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info("Removed %d empty texts.", before - len(df))

    # ── Deduplicate ──────────────────────────────────────────
    df, dedup_stats = _dedup(df)

    # ── Add length columns ───────────────────────────────────
    df = _add_text_length_columns(df)

    # ── Save processed ───────────────────────────────────────
    ensure_dir(PROCESSED_DIR)
    df.to_parquet(cache_path, index=False)

    # ── Save metadata ────────────────────────────────────────
    meta = {
        **DATASET_METADATA,
        "download_date": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(df),
        "human_rows": int((df["label"] == 0).sum()),
        "ai_rows": int((df["label"] == 1).sum()),
        "domains": sorted(df["source"].unique().tolist()),
        "columns": list(df.columns),
        **dedup_stats,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    logger.info("Processed data saved to %s", cache_path)
    logger.info(
        "Dataset summary: %d total | %d human | %d AI",
        len(df),
        meta["human_rows"],
        meta["ai_rows"],
    )

    return _maybe_sample(df, max_samples)


def _maybe_sample(df: pd.DataFrame, max_samples: Optional[int]) -> pd.DataFrame:
    """Optionally reduce the dataset to *max_samples* rows (stratified)."""
    if max_samples is None or max_samples >= len(df):
        return df
    logger.warning(
        "⚠️  Sampling %d / %d rows (MAX_SAMPLES is set). "
        "Final results will reflect this subset only.",
        max_samples,
        len(df),
    )
    from sklearn.model_selection import train_test_split

    df_sampled, _ = train_test_split(
        df,
        train_size=max_samples,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    return df_sampled.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    save_splits: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train / validation / test sets.

    Split ratios (defined in ``config.py``):
        - Train : 70 %
        - Val   : 15 %
        - Test  : 15 %

    Stratification is applied on the ``label`` column.

    The test set is returned last and **must not be used for any tuning**.

    Args:
        df: Full flat HC3 DataFrame (output of :func:`load_hc3`).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    val_test_size = VAL_SIZE + TEST_SIZE  # 0.30
    relative_val  = VAL_SIZE / val_test_size  # 0.15 / 0.30 = 0.5

    stratify_col = df["label"] if STRATIFY else None

    train_df, temp_df = train_test_split(
        df,
        test_size=val_test_size,
        random_state=RANDOM_STATE,
        stratify=stratify_col,
    )
    stratify_temp = temp_df["label"] if STRATIFY else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        random_state=RANDOM_STATE,
        stratify=stratify_temp,
    )

    logger.info(
        "Split sizes — Train: %d | Val: %d | Test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # Save splits
    if save_splits:
        ensure_dir(PROCESSED_DIR)
        train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
        val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
        test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
        logger.info("Splits saved to %s", PROCESSED_DIR)

    return train_df, val_df, test_df


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-saved train/val/test splits from disk.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        FileNotFoundError: If splits have not been generated yet.
    """
    paths = [
        PROCESSED_DIR / "train.parquet",
        PROCESSED_DIR / "val.parquet",
        PROCESSED_DIR / "test.parquet",
    ]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}. "
                "Run data_loader.split_dataset() first."
            )
    train_df = pd.read_parquet(paths[0])
    val_df   = pd.read_parquet(paths[1])
    test_df  = pd.read_parquet(paths[2])
    logger.info(
        "Loaded splits — Train: %d | Val: %d | Test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== HC3 Data Loader ===")
    df = load_hc3()
    print(f"\n{'='*50}")
    print(f"Total rows   : {len(df):,}")
    print(f"Human rows   : {(df['label']==0).sum():,}")
    print(f"AI rows      : {(df['label']==1).sum():,}")
    print(f"Domains      : {sorted(df['source'].unique().tolist())}")
    print(f"Columns      : {list(df.columns)}")
    print(f"{'='*50}\n")
    train_df, val_df, test_df = split_dataset(df)
    print("✅ Dataset loaded and split successfully.")
