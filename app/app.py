"""
app/app.py — Streamlit research prototype web application.
PRG4206 | A Comparative and Explainable Machine Learning System for AI-Generated Text Detection

Run via:
    streamlit run app/app.py
"""

from pathlib import Path
import sys

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st

from app.components import (
    render_chunk_deep_dive,
    render_disclaimer,
    render_document_summary,
    render_document_visualizations,
    render_explanation_card,
    render_header,
    render_metrics_dashboard,
    render_methodology_tab,
    render_prediction_result,
    render_text_stats,
    render_xai_deep_dive,
)
from app.ui_utils import (
    get_available_models,
    get_sample_presets,
)
from src.config import (
    DOC_MAX_CHUNK_WORDS,
    DOC_MIN_CHUNK_WORDS,
    LR_DIR,
    SVM_DIR,
    ensure_dirs,
)
from src.document_parser import (
    analyze_document,
    extract_text_from_file,
)
from src.explainability import get_lr_feature_weights
from src.inference import predict_text

# ── App Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Detector | PRG4206",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ensure_dirs()


# ── Sidebar Navigation & Model Selector ──────────────────────────────────────

def render_sidebar():
    """Render a compact Apple-inspired top navigation and model selector."""
    brand_col, nav_col, model_col = st.columns([1.35, 4.2, 1.65])
    with brand_col:
        st.markdown(
            """
            <div class="apple-topbar">
              <span class="apple-brand-mark">AI</span>
              <span>Text Intelligence<br><span style="font-weight:400;color:#6e6e73;font-size:.72rem;">PRG4206 Research</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    nav_options = ["Detect", "Documents", "Performance", "Explainability", "Methodology"]
    with nav_col:
        nav_page = st.radio(
            "Main navigation",
            nav_options,
            horizontal=True,
            label_visibility="collapsed",
            index=0,
            key="prg_main_navigation",
        )

    available_models = get_available_models()
    with model_col:
        if not available_models:
            st.error("No saved models detected.")
            selected_model = None
        else:
            default_idx = available_models.index("Linear SVM") if "Linear SVM" in available_models else 0
            selected_model = st.selectbox(
                "Active model",
                options=available_models,
                index=default_idx,
                key="prg_active_model",
            )

    st.markdown(
        '<div style="height:1px;background:#e5e5ea;margin:.1rem 0 1.3rem 0;"></div>',
        unsafe_allow_html=True,
    )
    return nav_page, selected_model


# ── Page 1: Home / Detection (Single Text Input) ──────────────────────────────

def page_home_detection(selected_model: str):
    """Render the FinMind-inspired text detection workspace."""
    st.markdown('<div class="prg-section-label">LIVE WORKSPACE / TEXT ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown("### Inspect a passage")
    st.caption(
        "Run a saved model against a text sample, then review its output and the features "
        "that influenced the prediction. The result is a research estimate, not proof of authorship."
    )

    # Dashboard-style context row: all values are from the verified HC3 evaluation.
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Active model", selected_model)
    with c2:
        st.metric("Clean HC3 samples", "79,331", "5 source domains")
    with c3:
        st.metric("Best observed accuracy", "98.61%", "Linear SVM · held-out test")

    st.markdown(" ")
    left, right = st.columns([1.65, 0.85], gap="large")

    presets = get_sample_presets()
    with left:
        st.markdown("#### Text input")
        preset_choice = st.selectbox(
            "Load a sample passage",
            options=list(presets.keys()),
            index=0,
            help="Sample passages are provided to make it easy to test the interface.",
        )
        default_text = presets[preset_choice] if preset_choice in presets else ""
        text_input = st.text_area(
            "Text to analyze",
            value=default_text,
            height=230,
            placeholder="Paste a passage here. Use at least 10 characters to run a prediction.",
            help="Text is processed by the selected saved model in this application.",
        )
        words = len(text_input.split())
        chars = len(text_input)
        stat1, stat2 = st.columns(2)
        stat1.caption(f"✍️ {words:,} words")
        stat2.caption(f"⌨️ {chars:,} characters")

        col_btn, col_hint = st.columns([1, 1.55])
        with col_btn:
            analyze_btn = st.button("Analyze passage  →", type="primary", use_container_width=True)
        with col_hint:
            st.caption("Choose a model in the left sidebar.")

    with right:
        st.markdown(
            """
            <div style="background:linear-gradient(150deg,rgba(25,42,72,.86),rgba(13,24,41,.91));
                        border:1px solid rgba(145,169,219,.20);border-radius:17px;
                        padding:20px 19px;margin:4px 0 14px 0;box-shadow:0 12px 26px rgba(0,0,0,.12);">
              <div style="font-size:.7rem;font-weight:800;letter-spacing:.13em;color:#8fb4ff;text-transform:uppercase;margin-bottom:10px;">
                Analysis pipeline
              </div>
              <div style="font-size:1.05rem;font-weight:750;color:#f1f5ff;margin-bottom:12px;">
                From text to evidence
              </div>
              <div style="display:flex;gap:11px;margin:13px 0;align-items:flex-start;">
                <div style="background:rgba(119,151,255,.16);border:1px solid rgba(119,151,255,.28);border-radius:10px;padding:7px 9px;color:#a9c2ff;">01</div>
                <div><b style="color:#eaf0fc;">Normalize</b><div style="font-size:.83rem;color:#a8b7d0;margin-top:3px;">Clean formatting while retaining useful language signals.</div></div>
              </div>
              <div style="display:flex;gap:11px;margin:13px 0;align-items:flex-start;">
                <div style="background:rgba(76,214,232,.12);border:1px solid rgba(76,214,232,.25);border-radius:10px;padding:7px 9px;color:#80e5ef;">02</div>
                <div><b style="color:#eaf0fc;">Classify</b><div style="font-size:.83rem;color:#a8b7d0;margin-top:3px;">Score the passage using the selected saved model.</div></div>
              </div>
              <div style="display:flex;gap:11px;margin:13px 0;align-items:flex-start;">
                <div style="background:rgba(74,222,173,.12);border:1px solid rgba(74,222,173,.25);border-radius:10px;padding:7px 9px;color:#78e9c1;">03</div>
                <div><b style="color:#eaf0fc;">Explain</b><div style="font-size:.83rem;color:#a8b7d0;margin-top:3px;">Review local term contributions where supported.</div></div>
              </div>
              <div style="height:1px;background:rgba(145,169,219,.16);margin:15px 0 12px;"></div>
              <div style="font-size:.78rem;color:#b9c8df;line-height:1.6;">
                <b style="color:#eaf0fc;">Models available</b><br>
                Linear SVM · Logistic Regression · XGBoost
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "Use results cautiously. A detector can make mistakes, especially with short text, "
            "formal human writing, or material from a different domain."
        )

    if analyze_btn or (default_text and len(text_input.strip()) >= 10):
        if len(text_input.strip()) < 10:
            st.error("Input text is too short. Please provide at least 10 characters.")
            return

        with st.spinner(f"Scoring text with {selected_model} …"):
            try:
                model_code = "svm"
                if "logistic" in selected_model.lower():
                    model_code = "lr"
                elif "xgboost" in selected_model.lower():
                    model_code = "xgb"

                result = predict_text(text_input, model_name=model_code)
                render_prediction_result(result)
                render_text_stats(result.get("text_stats", {}))

                exps = []
                if model_code in ["svm", "lr"]:
                    pipeline_path = (
                        SVM_DIR / "svm_pipeline.joblib"
                        if model_code == "svm"
                        else LR_DIR / "lr_pipeline.joblib"
                    )
                    if pipeline_path.exists():
                        pipeline = joblib.load(pipeline_path)
                        exps = get_lr_feature_weights(pipeline, text_input, top_n=10)

                if exps:
                    render_explanation_card(exps, model_name=selected_model)

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


# ── Page 2: Document Analysis ────────────────────────────────────────────────

def page_document_analysis(selected_model: str):
    """Render document upload, multi-chunk segmentation, and document-level aggregation."""
    st.markdown("### 📄 Document Analysis & Segmentation")
    st.write(
        "Upload an academic essay, report, or article (`.txt`, `.pdf`, `.docx`). "
        "The document will be segmented into coherent paragraphs, analyzed chunk-by-chunk, "
        "and aggregated into a document-level estimate."
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a document file:",
        type=["txt", "pdf", "docx"],
        help="Upload UTF-8 text, readable PDF, or Microsoft Word (.docx) document.",
    )

    # Advanced chunking settings
    with st.expander("⚙️ Advanced Chunking Settings", expanded=False):
        c_min, c_max = st.columns(2)
        with c_min:
            min_chunk = st.slider(
                "Minimum Chunk Size (words)",
                min_value=20,
                max_value=150,
                value=DOC_MIN_CHUNK_WORDS,
                step=5,
                help="Small paragraphs below this size are merged with adjacent text to ensure reliable TF-IDF representation.",
            )
        with c_max:
            max_chunk = st.slider(
                "Maximum Chunk Size (words)",
                min_value=100,
                max_value=500,
                value=DOC_MAX_CHUNK_WORDS,
                step=10,
                help="Large blocks are split cleanly on sentence ends to fit model context bounds.",
            )

    if uploaded_file is not None:
        try:
            # 1. Extraction
            with st.spinner("Extracting document text and metadata …"):
                doc_data = extract_text_from_file(
                    file_source=uploaded_file,
                    filename=uploaded_file.name,
                )

            st.success(
                f"✅ Extracted **{doc_data['total_words']:,}** words and "
                f"**{doc_data['segment_count']:,}** structural segments from `{uploaded_file.name}`."
            )

            # 2. Analysis button
            if st.button("🔍 Run Full Document Analysis", type="primary"):
                with st.spinner(f"Segmenting into chunks and scoring with {selected_model} …"):
                    model_code = "svm"
                    if "logistic" in selected_model.lower():
                        model_code = "lr"
                    elif "xgboost" in selected_model.lower():
                        model_code = "xgb"

                    analysis = analyze_document(
                        doc_data=doc_data,
                        model_name=model_code,
                        min_chunk_words=min_chunk,
                        max_chunk_words=max_chunk,
                    )

                # 3. Render document-level summary card
                render_document_summary(analysis)

                # 4. Render visualizations (Donut chart & timeline)
                render_document_visualizations(analysis)

                # 5. Render passage inspection deep dive
                render_chunk_deep_dive(analysis)

        except ValueError as val_err:
            st.error(f"⚠️ Document Parsing Warning: {val_err}")
        except Exception as exc:
            st.error(f"❌ Error during document analysis: {exc}")


# ── Main Application Routing ─────────────────────────────────────────────────

def main():
    nav_page, selected_model = render_sidebar()
    render_header()

    if selected_model is None:
        st.warning("Please ensure the saved model files exist in the models folder before running inference.")
        render_disclaimer()
        return

    if nav_page == "Detect":
        page_home_detection(selected_model)
    elif nav_page == "Documents":
        page_document_analysis(selected_model)
    elif nav_page == "Performance":
        render_metrics_dashboard()
    elif nav_page == "Explainability":
        render_xai_deep_dive()
    elif nav_page == "Methodology":
        render_methodology_tab()

    render_disclaimer()


if __name__ == "__main__":
    main()
