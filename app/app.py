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
    brand_col, nav_col, model_col = st.columns([1.15, 5.9, 1.75])
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
            format_func=lambda value: {
                "Detect": "Detect",
                "Documents": "Documents",
                "Performance": "Results",
                "Explainability": "Explain",
                "Methodology": "About",
            }.get(value, value),
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
    """Render a clean Apple-inspired detection screen."""
    st.markdown('<div class="apple-section-label">TEXT ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown("### A closer look at your text")
    st.markdown(
        "Choose a model, paste a passage, and review what the classifier predicts. "
        "The result is an estimate based on the HC3 benchmark—not a determination of authorship."
    )

    presets = get_sample_presets()
    preset_choice = st.selectbox(
        "Try a sample passage",
        options=list(presets.keys()),
        index=0,
        help="Load an example if you want to test the interface quickly.",
    )
    default_text = presets[preset_choice] if preset_choice in presets else ""

    main_col, side_col = st.columns([1.75, 0.85], gap="large")
    with main_col:
        text_input = st.text_area(
            "Text to analyze",
            value=default_text,
            height=245,
            placeholder="Paste or type a passage here…",
            help="The selected saved model will classify the text as Human-Written or AI-Generated.",
        )
        n_words = len(text_input.split())
        n_chars = len(text_input)
        m1, m2 = st.columns(2)
        m1.caption(f"{n_words:,} words")
        m2.caption(f"{n_chars:,} characters")
        analyze_btn = st.button("Analyze text  →", type="primary", use_container_width=False)

    with side_col:
        st.markdown(
            """
            <div style="background:#f5f5f7;border-radius:18px;padding:21px 20px;margin:.3rem 0 1rem;">
              <div style="font-size:.72rem;color:#6e6e73;font-weight:650;letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;">
                Current configuration
              </div>
              <div style="font-size:1.12rem;color:#1d1d1f;font-weight:650;letter-spacing:-.02em;margin-bottom:6px;">
                Selected model
              </div>
              <div style="font-size:.95rem;color:#424245;line-height:1.5;margin-bottom:16px;">
                MODEL_NAME_PLACEHOLDER
              </div>
              <div style="height:1px;background:#d2d2d7;margin:12px 0;"></div>
              <div style="font-size:.84rem;color:#424245;line-height:1.6;">
                <b style="color:#1d1d1f;">How it works</b><br>
                1. Text is cleaned.<br>
                2. The selected model predicts a class.<br>
                3. Word-level signals are shown when available.
              </div>
            </div>
            """.replace("MODEL_NAME_PLACEHOLDER", selected_model),
            unsafe_allow_html=True,
        )
        st.caption("A model may misclassify formal human writing, short passages, or text from unfamiliar domains.")

    if analyze_btn:
        if len(text_input.strip()) < 10:
            st.warning("Please enter at least 10 characters before running an analysis.")
            return

        with st.spinner(f"Analyzing with {selected_model}…"):
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
