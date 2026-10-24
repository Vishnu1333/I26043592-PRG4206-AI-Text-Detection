"""
app/components.py — Reusable, high-fidelity Streamlit UI components.
PRG4206 | AI-Generated Text Detection System
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from app.ui_utils import (
    load_confusion_matrices_df,
    load_domain_metrics_map,
    load_model_metrics,
    load_statistical_tests_df,
)
from src.config import FIGURES_DIR, METRICS_DIR


# ── Header & Banner ──────────────────────────────────────────────────────────

def render_global_styles():
    """Install the Flashform-inspired dark-olive and acid-lime design system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --ff-bg: #111606;
            --ff-bg-soft: #1a210e;
            --ff-panel: #1b2210;
            --ff-panel-2: #222b13;
            --ff-lime: #b1dd40;
            --ff-lime-bright: #c8f45c;
            --ff-ink: #171b0c;
            --ff-text: #eff2e7;
            --ff-muted: #a0a990;
            --ff-line: rgba(226, 239, 197, .16);
        }
        html, body, [class*="css"] {
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stApp {
            color: var(--ff-text);
            background:
                radial-gradient(ellipse at 85% -10%, rgba(177,221,64,.10), transparent 28%),
                radial-gradient(ellipse at 0% 35%, rgba(111,139,34,.07), transparent 26%),
                var(--ff-bg);
        }
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: transparent; }
        [data-testid="stHeader"] { border-bottom: 1px solid rgba(226,239,197,.08); }
        .main .block-container {
            max-width: 1390px;
            padding-top: 1rem;
            padding-bottom: 3.5rem;
        }
        [data-testid="stSidebar"] { background: #151b0a; border-right: 1px solid var(--ff-line); }
        [data-testid="stSidebar"] * { color: var(--ff-text); }
        h1, h2, h3, h4 {
            color: var(--ff-text);
            font-family: 'Space Grotesk', 'DM Sans', sans-serif;
            letter-spacing: -.045em;
            font-weight: 650;
        }
        h1 { font-size: clamp(2.4rem, 5.8vw, 5.1rem); line-height: .98; }
        h2 { font-size: clamp(1.7rem, 3.1vw, 2.55rem); }
        h3 { font-size: clamp(1.25rem, 2vw, 1.75rem); }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li { color: #d4dcc6; line-height: 1.65; }
        [data-testid="stCaptionContainer"] { color: var(--ff-muted); font-size: .82rem; }
        [data-testid="stMetric"] {
            background: linear-gradient(150deg, #202913, #171e0c);
            border: 1px solid var(--ff-line);
            border-radius: 18px;
            padding: 1.1rem 1.1rem .9rem;
            box-shadow: 0 10px 28px rgba(0,0,0,.12);
        }
        [data-testid="stMetricLabel"] { color: #b1bd9a; }
        [data-testid="stMetricValue"] { color: var(--ff-lime); font-family: 'Space Grotesk', sans-serif; font-weight: 650; letter-spacing: -.035em; }
        [data-testid="stMetricDelta"] { color: var(--ff-lime-bright); }
        .stButton > button, [data-testid="stDownloadButton"] button {
            min-height: 2.8rem;
            border-radius: 999px;
            border: 1px solid var(--ff-lime);
            background: var(--ff-lime);
            color: var(--ff-ink);
            font-weight: 700;
            padding: .42rem 1.2rem;
            box-shadow: none;
            transition: transform 160ms ease, background 160ms ease, box-shadow 160ms ease;
        }
        .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
            background: var(--ff-lime-bright);
            border-color: var(--ff-lime-bright);
            color: var(--ff-ink);
            transform: translateY(-2px);
            box-shadow: 0 7px 22px rgba(177,221,64,.15);
        }
        .stButton > button:focus { box-shadow: 0 0 0 3px rgba(177,221,64,.25); }
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div {
            background: #151b0a;
            border-color: rgba(226,239,197,.24);
            border-radius: 12px;
            color: var(--ff-text);
            box-shadow: none;
        }
        [data-baseweb="select"] input,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea { color: var(--ff-text); }
        [data-baseweb="select"] svg { fill: var(--ff-lime); }
        [data-testid="stFileUploader"] {
            background: #191f0d;
            border: 1px dashed rgba(177,221,64,.5);
            border-radius: 16px;
            padding: .6rem;
        }
        [data-testid="stFileUploader"] section { background: transparent; border: 0; }
        [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid var(--ff-line); }
        [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 999px;
            padding: .55rem .85rem;
            color: #a0a990;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            background: var(--ff-lime);
            border-color: var(--ff-lime);
            color: var(--ff-ink);
        }
        [data-testid="stExpander"] {
            background: #1a210e;
            border: 1px solid var(--ff-line);
            border-radius: 14px;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--ff-line);
            border-radius: 12px;
            overflow: hidden;
        }
        [data-testid="stAlert"] { border-radius: 14px; }
        hr { border-color: var(--ff-line); }
        a { color: var(--ff-lime) !important; }
        [data-testid="stRadio"] [role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: center;
            gap: .18rem;
            width: 100%;
        }
        [data-testid="stRadio"] [role="radiogroup"] > label {
            flex: 0 1 auto;
            min-width: 0;
            white-space: nowrap !important;
            opacity: 1 !important;
            border-radius: 999px;
            padding: .42rem .62rem;
            margin: 0 !important;
            border: 1px solid transparent;
            background: transparent;
            color: #d7dfc8 !important;
            transition: background 140ms ease, border-color 140ms ease;
        }
        [data-testid="stRadio"] [role="radiogroup"] > label * { opacity: 1 !important; }
        [data-testid="stRadio"] [role="radiogroup"] > label p,
        [data-testid="stRadio"] [role="radiogroup"] > label div,
        [data-testid="stRadio"] [role="radiogroup"] > label span {
            color: #d7dfc8 !important;
            font-size: .82rem !important;
            white-space: nowrap !important;
        }
        [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
            background: var(--ff-lime) !important;
            border-color: var(--ff-lime) !important;
        }
        [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) * { color: var(--ff-ink) !important; }
        .ff-topline {
            display:flex; align-items:center; justify-content:space-between; gap:1rem;
            border-bottom:1px solid var(--ff-line); padding:.45rem 0 .9rem; margin-bottom:1.15rem;
        }
        .ff-brand {display:flex; align-items:center; gap:.65rem; color:var(--ff-text); font-size:.83rem; font-weight:700; letter-spacing:-.02em;}
        .ff-mark {display:inline-flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:50%;background:var(--ff-lime);color:var(--ff-ink);font-weight:800;font-size:.75rem;}
        .ff-status {font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ff-lime);border:1px solid rgba(177,221,64,.35);border-radius:999px;padding:.45rem .7rem;white-space:nowrap;}
        .ff-hero {
            position:relative; overflow:hidden; padding:clamp(1.6rem,4vw,3.8rem);
            margin:.25rem 0 1.5rem; border:1px solid var(--ff-line); border-radius:22px;
            background:linear-gradient(130deg,rgba(27,34,16,.98),rgba(17,22,6,.98));
        }
        .ff-hero:after {
            content:"";position:absolute;right:-105px;top:-145px;width:390px;height:390px;border-radius:50%;
            border:1px solid rgba(177,221,64,.22);box-shadow:0 0 0 28px rgba(177,221,64,.025),0 0 0 58px rgba(177,221,64,.02);
            pointer-events:none;
        }
        .ff-kicker {font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--ff-lime);margin-bottom:1.1rem;}
        .ff-hero h1 {position:relative;z-index:1;max-width:900px;font-size:clamp(2.5rem,6.3vw,6.15rem);line-height:.93;letter-spacing:-.075em;margin:0 0 1.1rem;color:var(--ff-lime);}
        .ff-hero h1 em {font-style:normal;color:var(--ff-text);}
        .ff-hero p {position:relative;z-index:1;max-width:690px;color:#c7d0b6;font-size:clamp(1rem,1.6vw,1.23rem);line-height:1.55;margin:0;}
        .ff-hero-foot {display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.65rem;position:relative;z-index:1;}
        .ff-chip {font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.04em;color:#e7eddd;border:1px solid var(--ff-line);border-radius:999px;padding:.42rem .7rem;background:rgba(255,255,255,.025);}
        .ff-band {background:var(--ff-lime);color:var(--ff-ink);border-radius:18px;padding:1.15rem 1.35rem;margin:.8rem 0 1.35rem;}
        .ff-band strong {font-family:'Space Grotesk',sans-serif;font-size:1.12rem;letter-spacing:-.03em;color:var(--ff-ink);}
        .ff-band p {color:#30391c !important;margin:.3rem 0 0;font-size:.9rem;line-height:1.5;}
        .ff-section-label {font-family:'DM Mono',monospace;color:var(--ff-lime);font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;margin-bottom:.35rem;}
        .ff-lime-card {background:var(--ff-lime);color:var(--ff-ink);border-radius:18px;padding:1.25rem 1.35rem;}
        .ff-lime-card h3,.ff-lime-card p,.ff-lime-card strong {color:var(--ff-ink) !important;}
        [data-testid="stForm"] {background:var(--ff-lime);border:0;border-radius:22px;padding:1.25rem 1.25rem .85rem;}
        [data-testid="stForm"] label, [data-testid="stForm"] p {color:var(--ff-ink) !important;}
        [data-testid="stForm"] [data-baseweb="input"] > div {background:#f6f8ed;border-color:rgba(23,27,12,.16);}
        [data-testid="stForm"] input {color:var(--ff-ink) !important;}
        [data-testid="stForm"] button {background:var(--ff-ink);border-color:var(--ff-ink);color:var(--ff-lime);}
        [data-testid="stForm"] button:hover {background:#283016;border-color:#283016;color:var(--ff-lime-bright);}
        .ff-login-title {font-family:'Space Grotesk',sans-serif;font-size:clamp(2.8rem,6.8vw,6.4rem);font-weight:700;letter-spacing:-.075em;line-height:.9;color:var(--ff-lime);margin:.4rem 0 1.25rem;}
        .ff-login-title span {color:var(--ff-text);}
        .ff-login-copy {max-width:510px;color:#c8d0b8;font-size:1.06rem;line-height:1.7;}
        .ff-mini {font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ff-muted);}
        @keyframes ff-drift {0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
        .ff-orbit {animation:ff-drift 7s ease-in-out infinite;float:right;color:var(--ff-lime);font-size:clamp(5rem,11vw,10rem);font-weight:400;line-height:1;opacity:.9;letter-spacing:-.1em;}
        @media (prefers-reduced-motion: reduce) {.ff-orbit {animation:none;}}
        @media (max-width: 850px) {
            .main .block-container {padding-top:.6rem;}
            [data-testid="stRadio"] [role="radiogroup"] {justify-content:flex-start;overflow-x:auto;}
            .ff-hero {border-radius:16px;}
            .ff-topline {align-items:flex-start;}
            .ff-status {font-size:.58rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the project masthead and Flashform-inspired hero."""
    st.markdown(
        """
        <div class="ff-topline">
          <div class="ff-brand"><span class="ff-mark">TS</span><span>TEXT SIGNAL<br><span style="color:#a0a990;font-size:.72rem;font-weight:400;">PRG4206 Research Console</span></span></div>
          <div class="ff-status">HC3 / MODEL RESEARCH</div>
        </div>
        <div class="ff-hero">
          <div class="ff-kicker">Language · Patterns · Evidence</div>
          <div class="ff-orbit" aria-hidden="true">✳</div>
          <h1>Words leave signals.<br><em>We make them visible.</em></h1>
          <p>Explore how three machine-learning models respond to a passage, compare their benchmark performance, and inspect the signals behind each prediction.</p>
          <div class="ff-hero-foot">
            <span class="ff-chip">79,331 cleaned HC3 samples</span>
            <span class="ff-chip">3 evaluated models</span>
            <span class="ff-chip">Human + AI text</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Prediction Result Display ────────────────────────────────────────────────

def render_prediction_result(result: Dict[str, Any]):
    """Render an acid-lime/dark-green result card using the model's actual output."""
    label = result.get("label", "Unknown")
    confidence = float(result.get("confidence", 0.0))
    model = result.get("model", "Unknown")
    is_ai = label == "AI-Generated"
    accent = "#b1dd40" if is_ai else "#c8f45c"
    panel = "#262515" if is_ai else "#1a2510"
    icon = "✳" if is_ai else "↗"
    ai_prob = confidence if is_ai else (1.0 - confidence)
    human_prob = 1.0 - ai_prob

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{panel},#171d0b);
                    border:1px solid rgba(177,221,64,.36);border-radius:18px;
                    padding:22px 24px;margin:18px 0 18px;
                    box-shadow:0 12px 32px rgba(0,0,0,.12);">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;">
            <div>
              <div style="font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:{accent};">
                MODEL OUTPUT / CLASSIFICATION
              </div>
              <div style="font-family:'Space Grotesk',sans-serif;color:#f0f3e7;font-size:clamp(1.55rem,3vw,2.2rem);
                          font-weight:700;letter-spacing:-.055em;margin:8px 0 5px;">
                {icon} {label}
              </div>
              <div style="font-size:.86rem;color:#a0a990;">Evaluated by <b style="color:#e9eddc;">{model}</b></div>
            </div>
            <div style="background:#b1dd40;border-radius:15px;padding:13px 17px;min-width:140px;">
              <div style="font-family:'DM Mono',monospace;font-size:.66rem;color:#31391c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:5px;">Confidence</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;line-height:1.12;font-weight:700;color:#171b0c;">{confidence * 100:.1f}%</div>
            </div>
          </div>
          <div style="margin-top:15px;font-size:.79rem;color:#aeb79d;line-height:1.6;">
            This is a model estimate based on learned HC3 patterns, not proof of who authored the passage.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Human probability** · {human_prob * 100:.1f}%")
        st.progress(human_prob)
    with c2:
        st.markdown(f"**AI probability** · {ai_prob * 100:.1f}%")
        st.progress(ai_prob)

    st.caption("Results may be wrong for short passages, formal human writing, or material from unfamiliar domains.")


# ── Text Statistics ──────────────────────────────────────────────────────────

def render_text_stats(stats: Dict[str, Any]):
    """Display word, character, and sentence counts in structured cards."""
    if not stats:
        return
    st.markdown("##### 📄 Text Surface Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Words", f"{stats.get('word_count', 0):,}")
    c2.metric("Characters", f"{stats.get('char_count', 0):,}")
    c3.metric("Sentences", f"{stats.get('sentence_count', 0):,}")


# ── Word Attribution / XAI Card ──────────────────────────────────────────────

def render_explanation_card(exps: List[Dict[str, Any]], model_name: str = "Linear Model"):
    """Display feature attribution weights for single text prediction."""
    if not exps:
        return

    st.markdown("##### 🔍 Local Explainability (Top Contributing Terms)")
    st.caption(f"Feature contributions derived from {model_name} TF-IDF weights.")

    ai_terms = [e for e in exps if e["direction"] == "AI"]
    human_terms = [e for e in exps if e["direction"] == "Human"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Terms pushing towards AI-Generated (Positive):**")
        if ai_terms:
            for item in ai_terms:
                st.markdown(
                    f"<span style='background-color: #fee2e2; color: #991b1b; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-family: monospace; display: inline-block; margin: 2px;'>"
                    f"{item['feature']} (+{item['weight']:.3f})</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.write("None strongly activated.")

    with c2:
        st.markdown("**Terms pushing towards Human-Written (Negative):**")
        if human_terms:
            for item in human_terms:
                st.markdown(
                    f"<span style='background-color: #dcfce7; color: #166534; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-family: monospace; display: inline-block; margin: 2px;'>"
                    f"{item['feature']} ({item['weight']:.3f})</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.write("None strongly activated.")


# ── Document Analysis Components ─────────────────────────────────────────────

def render_document_summary(analysis: Dict[str, Any]):
    """Render the document-level estimate in the same Flashform-inspired visual style."""
    overall = analysis["overall_prediction"]
    conf = analysis["overall_confidence"]
    is_ai = overall == "AI-Generated"
    accent = "#b1dd40"
    panel = "#282515" if is_ai else "#1b2410"
    icon = "✳" if is_ai else "↗"

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{panel},#171d0b);
                    border:1px solid rgba(177,221,64,.36);border-radius:18px;
                    padding:22px 24px;margin:16px 0 20px;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:18px;flex-wrap:wrap;">
            <div>
              <div style="font-family:'DM Mono',monospace;font-size:.68rem;font-weight:500;letter-spacing:.14em;text-transform:uppercase;color:{accent};">
                DOCUMENT OUTPUT / ESTIMATE
              </div>
              <div style="font-family:'Space Grotesk',sans-serif;color:#f0f3e7;font-size:clamp(1.55rem,3vw,2.2rem);
                          font-weight:700;letter-spacing:-.055em;margin:8px 0 5px;">
                {icon} {overall}
              </div>
              <div style="font-size:.86rem;color:#a0a990;">
                File <b style="color:#e9eddc;">{analysis.get('filename', 'document')}</b>
                &nbsp;·&nbsp; Model <b style="color:#e9eddc;">{analysis.get('model_used', 'Linear SVM')}</b>
              </div>
            </div>
            <div style="background:{accent};border-radius:15px;padding:13px 17px;min-width:145px;">
              <div style="font-family:'DM Mono',monospace;font-size:.66rem;color:#31391c;letter-spacing:.08em;text-transform:uppercase;margin-bottom:5px;">Confidence</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;line-height:1.12;font-weight:700;color:#171b0c;">{conf * 100:.1f}%</div>
            </div>
          </div>
          <div style="margin-top:15px;font-size:.79rem;color:#aeb79d;line-height:1.6;">
            Estimated by aggregating chunk predictions. This is not an independently validated document-level benchmark.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total words", f"{analysis['total_words']:,}")
    m2.metric("Text chunks", f"{analysis['total_chunks']:,}")
    m3.metric("AI-classified chunks", f"{analysis['ai_chunks']} ({analysis['ai_percentage']}%)")
    m4.metric("Human-classified chunks", f"{analysis['human_chunks']} ({100 - analysis['ai_percentage']:.1f}%)")


def render_document_visualizations(analysis: Dict[str, Any]):
    """Render interactive chunk distribution and chronological timeline charts."""
    st.markdown("#### 📊 Document Visualizations & Chunk Timeline")

    chunks = analysis["chunks"]
    df_chunks = pd.DataFrame(chunks)

    col1, col2 = st.columns([1, 2])

    with col1:
        # Pie / Donut Chart of AI vs Human Chunks
        st.markdown("##### Chunk Classification Breakdown")
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["Human Chunks", "AI Chunks"]
        counts = [analysis["human_chunks"], analysis["ai_chunks"]]
        colors = ["#22c55e", "#ef4444"]

        # Only plot non-zero wedges
        non_zero = [(l, c, cl) for l, c, cl in zip(labels, counts, colors) if c > 0]
        if non_zero:
            n_labels, n_counts, n_colors = zip(*non_zero)
            wedges, texts, autotexts = ax.pie(
                n_counts,
                labels=n_labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=n_colors,
                textprops=dict(color="#1e293b", weight="bold"),
                wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
            )
        ax.set_title("Chunk Distribution", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        # Sequential timeline of AI probability across chunk indices
        st.markdown("##### Chunk-by-Chunk AI Probability Timeline")
        fig, ax = plt.subplots(figsize=(8, 4))
        indices = df_chunks["chunk_index"].values
        probs = df_chunks["ai_probability"].values
        colors = ["#ef4444" if p >= 0.50 else "#22c55e" for p in probs]

        bars = ax.bar(indices, probs, color=colors, width=0.6, edgecolor="none", alpha=0.9)
        ax.axhline(0.50, color="#64748b", linestyle="--", linewidth=1.2, label="Decision Threshold (0.50)")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Chunk Index (Ordered Through Document)", fontsize=9, fontweight="bold")
        ax.set_ylabel("AI Probability", fontsize=9, fontweight="bold")
        ax.set_title("Sequential Chunk Scoring Timeline", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_chunk_deep_dive(analysis: Dict[str, Any]):
    """Render expandable deep dive sections for top AI, top Human, and full chunk table."""
    st.markdown("#### 🔬 Passage Inspection (Deep Dive)")

    tab_ai, tab_human, tab_all = st.tabs(
        ["🤖 Most AI-Like Passages", "✍️ Most Human-Like Passages", "📋 Complete Chunk Table"]
    )

    with tab_ai:
        if analysis["most_ai_chunks"]:
            for item in analysis["most_ai_chunks"]:
                with st.expander(
                    f"Chunk #{item['chunk_index']} — {item['source_info']} | "
                    f"AI Probability: {item['ai_probability']*100:.1f}% ({item['word_count']} words)"
                ):
                    st.write(item["text"])
        else:
            st.info("No AI-classified chunks identified.")

    with tab_human:
        if analysis["most_human_chunks"]:
            for item in analysis["most_human_chunks"]:
                with st.expander(
                    f"Chunk #{item['chunk_index']} — {item['source_info']} | "
                    f"Human Probability: {item['human_probability']*100:.1f}% ({item['word_count']} words)"
                ):
                    st.write(item["text"])
        else:
            st.info("No Human-classified chunks identified.")

    with tab_all:
        df_all = pd.DataFrame(analysis["chunks"])
        display_cols = ["chunk_index", "source_info", "word_count", "label", "confidence", "ai_probability"]
        available_cols = [c for c in display_cols if c in df_all.columns]
        st.dataframe(
            df_all[available_cols].style.format(
                {"confidence": "{:.4f}", "ai_probability": "{:.4f}"}
            ),
            use_container_width=True,
        )

        csv_data = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Chunk Predictions (CSV)",
            data=csv_data,
            file_name=f"{analysis['filename']}_chunk_analysis.csv",
            mime="text/csv",
        )


# ── Model Performance Dashboard ──────────────────────────────────────────────

def render_metrics_dashboard():
    """Render a clearer, decision-focused dashboard for the verified HC3 benchmark."""
    st.markdown("## 📊 Model Performance Dashboard")
    st.caption(
        "Held-out evaluation on 11,900 HC3 answer samples · 7,963 human-written · "
        "3,937 AI-generated · Same test examples for all models"
    )

    metrics_df = load_model_metrics()
    if metrics_df is None or metrics_df.empty:
        st.error("Verified model metrics could not be loaded. Check results/metrics/model_comparison.csv.")
        return

    df = metrics_df.copy()
    if "model" not in df.columns:
        st.error("The model comparison file is missing the required 'model' column.")
        return

    numeric_metrics = [
        "accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1",
        "balanced_accuracy", "mcc", "specificity", "roc_auc", "pr_auc"
    ]
    for col in numeric_metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["model"]).reset_index(drop=True)
    if df.empty:
        st.warning("No usable model rows were found in the saved metrics.")
        return

    # Headline values are read from the saved evaluation file, not hard-coded.
    best_accuracy = df.loc[df["accuracy"].idxmax()] if "accuracy" in df else None
    best_f1 = df.loc[df["f1"].idxmax()] if "f1" in df else None
    best_auc = df.loc[df["roc_auc"].idxmax()] if "roc_auc" in df else None

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if best_accuracy is not None:
            st.metric("Top test accuracy", f"{best_accuracy['accuracy']:.2%}",
                      help=f"Model: {best_accuracy['model']}")
            st.caption(str(best_accuracy["model"]))
    with k2:
        if best_f1 is not None:
            st.metric("Top F1-score", f"{best_f1['f1']:.4f}",
                      help=f"Model: {best_f1['model']}")
            st.caption(str(best_f1["model"]))
    with k3:
        if best_auc is not None:
            st.metric("Top ROC-AUC", f"{best_auc['roc_auc']:.4f}",
                      help=f"Model: {best_auc['model']}")
            st.caption(str(best_auc["model"]))
    with k4:
        st.metric("Held-out samples", "11,900")
        st.caption("One shared test set")

    st.divider()
    st.markdown("### Model comparison")
    st.write(
        "The chart compares the main classification measures. Accuracy shows overall correctness; "
        "precision and recall describe the AI-generated class; F1 balances precision and recall."
    )

    chart_metrics = [c for c in ["accuracy", "f1", "precision", "recall"] if c in df.columns]
    if chart_metrics:
        chart_df = df.set_index("model")[chart_metrics].copy()
        chart_df = chart_df.rename(columns={
            "accuracy": "Accuracy", "f1": "F1-score",
            "precision": "Precision", "recall": "Recall"
        }) * 100
        st.bar_chart(chart_df, use_container_width=True)
        st.caption("Values are percentages. The plotted values come from the saved held-out test metrics.")

    display_names = {
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
        "f1": "F1", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1",
        "balanced_accuracy": "Balanced accuracy", "mcc": "MCC",
        "specificity": "Specificity", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"
    }
    table_cols = [c for c in ["model"] + numeric_metrics if c in df.columns]
    table_df = df[table_cols].rename(columns=display_names)
    pct_cols = [display_names[c] for c in [
        "accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1",
        "balanced_accuracy", "specificity", "roc_auc", "pr_auc"
    ] if c in table_cols]
    format_map = {c: "{:.2%}" for c in pct_cols}
    if "MCC" in table_df.columns:
        format_map["MCC"] = "{:.4f}"
    st.dataframe(
        table_df.style.format(format_map).highlight_max(
            subset=[c for c in pct_cols + (["MCC"] if "MCC" in table_df.columns else [])
                    if c in table_df.columns],
            color="#e4f4ec"
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("How to read these results", expanded=True):
        st.markdown(
            """
            - **Linear SVM** has the highest observed accuracy (98.61%) and F1-score (0.9790).
            - **Logistic Regression** is very close and has slightly higher ROC-AUC and PR-AUC.
            - **XGBoost** uses 21 stylometric measurements rather than the TF-IDF vocabulary.
            - Strong benchmark scores do not prove authorship. Results can change for short passages,
              different domains, edited text, and newer language models.
            """
        )

    tab_domain, tab_stats, tab_curves = st.tabs(
        ["🌐 Domain performance", "🧪 Statistical tests", "📈 Diagnostic figures"]
    )

    with tab_domain:
        st.markdown("#### Performance by source domain")
        st.caption("Domain results are useful because a strong overall score can hide weaker categories.")
        domain_map = load_domain_metrics_map()
        domain_frames = []
        for model_name, dom_df in domain_map.items():
            if not {"domain", "f1"}.issubset(dom_df.columns):
                continue
            tmp = dom_df[["domain", "f1"]].copy()
            tmp["f1"] = pd.to_numeric(tmp["f1"], errors="coerce")
            tmp["Model"] = model_name
            domain_frames.append(tmp)

        if domain_frames:
            domains = pd.concat(domain_frames, ignore_index=True).dropna(subset=["f1"])
            pivot = domains.pivot_table(index="domain", columns="Model", values="f1", aggfunc="first")
            st.bar_chart(pivot * 100, use_container_width=True)
            st.dataframe(
                pivot.style.format("{:.2%}").highlight_max(axis=1, color="#e4f4ec"),
                use_container_width=True,
            )
            st.caption("Green cells indicate the highest F1-score within a domain.")
        else:
            st.info("Domain-level metrics are not available in this repository copy.")

    with tab_stats:
        st.markdown("#### Paired McNemar significance tests")
        st.write(
            "Each pair is evaluated on the same test examples. The test checks whether their "
            "disagreement in correctness is large enough to support a statistical difference."
        )
        stat_df = load_statistical_tests_df()
        if stat_df is not None and not stat_df.empty:
            pair_cols = [
                c for c in [
                    "comparison", "both_correct", "model_a_only_correct",
                    "model_b_only_correct", "both_wrong", "statistic",
                    "p_value", "exact_p_value", "significant", "interpretation"
                ] if c in stat_df.columns
            ]
            shown = stat_df[pair_cols].copy()
            for col in ["p_value", "exact_p_value"]:
                if col in shown.columns:
                    shown[col] = shown[col].apply(
                        lambda value: "< 0.000001" if pd.notna(value) and float(value) < 0.000001
                        else (f"{float(value):.6f}" if pd.notna(value) else "—")
                    )
            if "statistic" in shown.columns:
                shown["statistic"] = pd.to_numeric(shown["statistic"], errors="coerce").map(
                    lambda value: f"{value:.4f}" if pd.notna(value) else "—"
                )
            st.dataframe(shown, use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**SVM vs Logistic Regression**")
                st.metric("p-value", "0.1443")
                st.caption("Not statistically significant at α = 0.05.")
            with c2:
                st.markdown("**SVM vs XGBoost**")
                st.metric("p-value", "< 0.000001")
                st.caption("Statistically significant difference.")
            with c3:
                st.markdown("**Logistic Regression vs XGBoost**")
                st.metric("p-value", "< 0.000001")
                st.caption("Statistically significant difference.")
        else:
            st.info("Verified pairwise statistical results could not be loaded.")

    with tab_curves:
        st.markdown("#### ROC curves and feature-importance figure")
        st.caption("These saved plots complement the interactive comparison above.")
        c1, c2 = st.columns(2)
        with c1:
            svm_roc = FIGURES_DIR / "linear_svm_roc_curve.png"
            if svm_roc.exists():
                st.image(str(svm_roc), caption="Linear SVM ROC curve")
            else:
                st.info("Linear SVM ROC curve not found.")
        with c2:
            lr_roc = FIGURES_DIR / "logistic_regression_roc_curve.png"
            if lr_roc.exists():
                st.image(str(lr_roc), caption="Logistic Regression ROC curve")
            else:
                st.info("Logistic Regression ROC curve not found.")

        xgb_shap = FIGURES_DIR / "xgb_shap_summary.png"
        if xgb_shap.exists():
            st.image(str(xgb_shap), caption="XGBoost SHAP summary")

    st.info(
        "Research-use notice: these are benchmark estimates, not proof of authorship. "
        "A human review is required before making consequential decisions."
    )


# ── Explainability Deep Dive ─────────────────────────────────────────────────

def render_xai_deep_dive():
    """Render explainability deep dive tab with SHAP plots and methodology notes."""
    st.markdown("### 🔍 Explainable AI (XAI) Deep Dive")
    st.caption("Unpacking model decision mechanics through global and local feature attribution.")

    tab_shap, tab_linear = st.tabs(["🌳 XGBoost Stylometric SHAP", "📐 Linear Model TF-IDF Coefficients"])

    with tab_shap:
        st.markdown("#### SHAP TreeExplainer on 21 Stylometric Features")
        st.write(
            "SHAP (SHapley Additive exPlanations) values quantify how each stylometric feature pushes the "
            "model prediction away from the base value towards AI (positive SHAP) or Human (negative SHAP)."
        )
        xgb_shap_path = FIGURES_DIR / "xgb_shap_summary.png"
        if xgb_shap_path.exists():
            st.image(str(xgb_shap_path), caption="XGBoost SHAP Summary Plot (Test Split)")
        else:
            st.info("SHAP figure not found in results/figures/.")

    with tab_linear:
        st.markdown("#### Global TF-IDF Vocabulary Weights")
        st.write(
            "Linear models offer transparent interpretability through feature coefficients. "
            "Terms with high positive weights correlate strongly with ChatGPT stylistic markers."
        )
        lr_shap_path = FIGURES_DIR / "lr_shap_summary.png"
        if lr_shap_path.exists():
            st.image(str(lr_shap_path), caption="Logistic Regression Global Feature Coefficients")


# ── Methodology & About ──────────────────────────────────────────────────────

def render_methodology_tab():
    """Render academic methodology and project context."""
    st.markdown("### 📚 Methodology & Research Framework")
    st.markdown(
        """
        #### 1. Research Problem & Benchmark Dataset
        This research prototype investigates the discriminative capacity of classical machine learning and
        stylometric models to distinguish **Human-written text** from **AI-generated text (ChatGPT)**.
        - **Primary Benchmark**: **HC3 (Human ChatGPT Comparison Corpus, Guo et al., 2023)**.
        - **Dataset Scale**: **79,331 verified unique samples** (53,086 Human, 26,245 AI).
        - **Domain Diversity**: 5 heterogeneous categories: `reddit_eli5` (casual conversational),
          `finance` (technical business), `open_qa` (factual open-domain), `medicine` (clinical health),
          and `wiki_csai` (formal computer science encyclopedic writing).

        #### 2. Experimental Rigor & Leakage Controls
        - **Strict Stratification**: 70% Train (55,531), 15% Validation (11,900), and 15% Held-out Test (11,900).
        - **Zero Data Leakage**: Feature vectorizers (50,000 sublinear TF-IDF n-grams) and standard scalers were
          fitted exclusively on the training partition.
        - **Calibration**: Linear SVM is calibrated via Platt scaling (3-fold sigmoid calibration) to provide
          true posterior probabilities.

        #### 3. Critical Limitations
        - **Domain Generalization**: Performance is highest on text styles present in HC3. Accuracy can decrease
          under severe domain shift or novel LLMs (e.g., Claude 3.5, Gemini 1.5).
        - **Adversarial Paraphrasing**: Heavy rewriting via tools or prompt engineering can alter vocabulary n-grams.
        - **Non-Native English Vulnerability**: Non-native English text can exhibit lower lexical perplexity,
          occasionally increasing false-positive risk.
        """
    )


# ── Disclaimer ───────────────────────────────────────────────────────────────

def render_disclaimer():
    """Render prominent ethical and responsible AI notice."""
    st.markdown("---")
    st.warning(
        """
        ⚖️ **Ethical Notice & Responsible AI Disclaimer**  
        This detector is an **academic research prototype** created for PRG4206. AI text detection algorithms
        produce probabilistic estimates based on observed distributional patterns and **must NEVER be used as the
        sole basis for academic integrity penalties, disciplinary action, or employment decisions**.
        Independent human review and qualitative verification are mandatory.
        """
    )
