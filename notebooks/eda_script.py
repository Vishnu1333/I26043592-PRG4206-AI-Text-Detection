"""
notebooks/eda_script.py — Exploratory Data Analysis for HC3 dataset.
PRG4206 | AI-Generated Text Detection System

Run with:
    python notebooks/eda_script.py

Generates all EDA figures to results/figures/eda_*.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import FIGURES_DIR, ensure_dirs
from src.data_loader import load_hc3
from src.preprocessing import preprocess_dataframe
from src.utils import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
ensure_dirs()


def run_eda():
    # ── 1. Load dataset ───────────────────────────────────────
    logger.info("Loading HC3 dataset for EDA …")
    df = load_hc3()
    logger.info("Dataset loaded: %d rows", len(df))

    print("\n" + "="*60)
    print(" HC3 DATASET — EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\n{'Total samples':30s}: {len(df):>10,}")
    print(f"{'Human samples (label=0)':30s}: {(df['label']==0).sum():>10,}")
    print(f"{'AI samples (label=1)':30s}: {(df['label']==1).sum():>10,}")
    print(f"{'Unique domains':30s}: {df['source'].nunique():>10}")
    print(f"{'Missing text values':30s}: {df['text'].isna().sum():>10}")

    # ── 2. Class distribution ─────────────────────────────────
    logger.info("Plotting class distribution …")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["label"].value_counts().rename({0: "Human", 1: "AI"})
    bars = ax.bar(counts.index, counts.values,
                  color=["#3498db", "#e74c3c"])
    ax.bar_label(bars, fmt="%d")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution (HC3)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)

    # ── 3. Text length distribution ───────────────────────────
    logger.info("Plotting text length distributions …")
    df["word_count"] = df["text"].str.split().str.len()
    df["char_count"] = df["text"].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Word count
    for label_id, label_name, color in [(0, "Human", "#3498db"), (1, "AI", "#e74c3c")]:
        sub = df[df["label"] == label_id]["word_count"]
        axes[0].hist(sub.clip(0, 1000), bins=50, alpha=0.6,
                     label=f"{label_name} (median={sub.median():.0f})",
                     color=color)
    axes[0].set_xlabel("Word Count (clipped at 1000)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Word Count Distribution")
    axes[0].legend()

    # Character count
    for label_id, label_name, color in [(0, "Human", "#3498db"), (1, "AI", "#e74c3c")]:
        sub = df[df["label"] == label_id]["char_count"]
        axes[1].hist(sub.clip(0, 6000), bins=50, alpha=0.6,
                     label=f"{label_name} (median={sub.median():.0f})",
                     color=color)
    axes[1].set_xlabel("Character Count (clipped at 6000)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Character Count Distribution")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_length_distribution.png", dpi=150)
    plt.close(fig)

    # ── 4. Boxplot: word count by class ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    df["Class"] = df["label"].map({0: "Human", 1: "AI"})
    sns.boxplot(data=df, x="Class", y="word_count",
                palette={"Human": "#3498db", "AI": "#e74c3c"},
                showfliers=False, ax=ax)
    ax.set_title("Word Count by Class (outliers hidden)")
    ax.set_ylabel("Word Count")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_wordcount_boxplot.png", dpi=150)
    plt.close(fig)

    # ── 5. Domain distribution ────────────────────────────────
    logger.info("Plotting domain distribution …")
    domain_counts = df["source"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(domain_counts.index, domain_counts.values,
                   color=sns.color_palette("muted", len(domain_counts)))
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Count")
    ax.set_title("HC3 Domain Distribution")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_domain_distribution.png", dpi=150)
    plt.close(fig)

    # ── 6. Domain × class breakdown ───────────────────────────
    domain_class = df.groupby(["source", "Class"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    domain_class.plot(kind="bar", ax=ax,
                      color=["#3498db", "#e74c3c"])
    ax.set_xlabel("Domain")
    ax.set_ylabel("Count")
    ax.set_title("Domain × Class Distribution")
    ax.legend(title="Class")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_domain_class_breakdown.png", dpi=150)
    plt.close(fig)

    # ── 7. Summary statistics ─────────────────────────────────
    print("\n--- Text Length Summary ---")
    summary = df.groupby("Class")[["word_count", "char_count"]].describe()
    print(summary.to_string())

    print("\n--- Domain Distribution ---")
    print(domain_counts.to_string())

    print("\n--- Outlier Detection ---")
    short = (df["word_count"] < 10).sum()
    long  = (df["word_count"] > 500).sum()
    print(f"  Texts with < 10 words  : {short:,}")
    print(f"  Texts with > 500 words : {long:,}")

    print("\n✅ EDA complete. Figures saved to results/figures/")
    logger.info("EDA complete. %d figures saved to %s", 5, FIGURES_DIR)

    return df


if __name__ == "__main__":
    run_eda()
