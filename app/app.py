"""
app/app.py — Streamlit research prototype web application.
PRG4206 | A Comparative and Explainable Machine Learning System for AI-Generated Text Detection

Run via:
    streamlit run app/app.py
"""

from pathlib import Path
import hmac
import os
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
    render_feature_card,
    render_footer,
    render_global_styles,
    render_header,
    render_metrics_dashboard,
    render_methodology_tab,
    render_page_heading,
    render_prediction_result,
    render_text_stats,
    render_xai_deep_dive,
    render_marquee,
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
    page_title="Text Signal | PRG4206",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ensure_dirs()


# ── Login, Top Navigation & Model Selector ───────────────────────────────────

def _get_login_credentials():
    """Read optional login credentials from Streamlit secrets or environment variables."""
    configured_user = os.getenv("PRG4206_LOGIN_USERNAME", "")
    configured_password = os.getenv("PRG4206_LOGIN_PASSWORD", "")
    try:
        auth_config = st.secrets.get("auth", {})
        configured_user = auth_config.get("username", configured_user)
        configured_password = auth_config.get("password", configured_password)
    except Exception:
        # Streamlit secrets are optional for the local/demo deployment.
        pass
    return str(configured_user or ""), str(configured_password or "")


def render_login_screen():
    """Render an attractive login gateway; support an explicitly labelled demo mode."""
    st.markdown(
        """
        <div class="ff-topline ff-animate-in">
          <div class="ff-brand"><span class="ff-mark">TS</span><span>TEXT SIGNAL<br><span style="color:#b8c2a4;font-size:.72rem;font-weight:400;">PRG4206 Research Console</span></span></div>
          <div class="ff-status">RESEARCH ACCESS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # The login page also gets the moving ticker; previously it appeared only after sign-in.
    render_marquee(["TEXT SIGNAL", "DETECT", "COMPARE", "EXPLAIN", "VALIDATE", "UNDERSTAND"])

    col_copy, col_form = st.columns([1.15, .85], gap="large")
    with col_copy:
        st.markdown('<div class="ff-mini">A machine-learning research workspace</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ff-login-title ff-animate-in">Read the text.<br><span class="ff-login-accent">Find the signal.</span><span class="ff-login-orbit" aria-hidden="true">✳</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ff-login-copy ff-animate-in ff-delay-2">Compare human-written and AI-generated text using three saved models. Explore the evidence behind a prediction, review benchmark results, and inspect long documents one section at a time.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="ff-hero-foot" style="margin-top:1.6rem;">
              <span class="ff-chip">01 / Text detection</span>
              <span class="ff-chip">02 / Model comparison</span>
              <span class="ff-chip">03 / Explainability</span>
            </div>
            <div style="height:1px;background:rgba(226,239,197,.16);margin:2rem 0 1rem;"></div>
            <div class="ff-mini">Built for PRG4206 · Individual Machine Learning Assignment</div>
            """,
            unsafe_allow_html=True,
        )

    with col_form:
        st.markdown(
            """
            <div class="ff-login-welcome ff-login-panel ff-animate-in ff-delay-3" style="margin-bottom:1rem;">
              <div class="ff-mini">WELCOME BACK</div>
              <h3 style="font-size:2rem;line-height:1.05;margin:.55rem 0 .45rem;">Your workspace,<br>ready when you are.</h3>
              <p style="margin:0;">Sign in to continue to the detection dashboard.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        expected_user, expected_password = _get_login_credentials()
        with st.form("text_signal_login", clear_on_submit=False):
            username = st.text_input("Email or username", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in  →", use_container_width=True)

        if submitted:
            if expected_user and expected_password:
                valid_user = hmac.compare_digest(username.strip(), expected_user)
                valid_password = hmac.compare_digest(password, expected_password)
                if valid_user and valid_password:
                    st.session_state["text_signal_authenticated"] = True
                    st.session_state["text_signal_access_mode"] = "Configured account"
                    st.rerun()
                else:
                    st.error("The username or password doesn't match the configured account.")
            else:
                st.error(
                    "Login credentials are not configured for this deployment yet. "
                    "Use demo access below, or configure an [auth] section in Streamlit secrets."
                )

        if not (expected_user and expected_password):
            st.markdown(
                '<p class="ff-mini" style="margin:.7rem 0 .35rem;color:#a0a990;">DEMO ENVIRONMENT · NO ACCOUNT SETUP</p>',
                unsafe_allow_html=True,
            )
            if st.button("Explore demo workspace  ↗", use_container_width=True, key="text_signal_demo_access"):
                st.session_state["text_signal_authenticated"] = True
                st.session_state["text_signal_access_mode"] = "Public demo"
                st.rerun()
            st.caption(
                "Demo access is open because no account credentials are configured. "
                "This is a UI access gate, not a substitute for production authentication."
            )

        st.markdown(
            '<div class="ff-mini" style="margin-top:1rem;">No personal document is saved by this login screen.</div>',
            unsafe_allow_html=True,
        )


def render_sidebar():
    """Render the compact app navigation, model selector, and sign-out control."""
    brand_col, nav_col, model_col, logout_col = st.columns([1.25, 5.0, 1.55, .65])
    with brand_col:
        st.markdown(
            """
            <div class="ff-brand" style="padding:.45rem 0;">
              <span class="ff-mark">TS</span>
              <span>TEXT SIGNAL<br><span style="color:#a0a990;font-size:.68rem;font-weight:400;">Research Console</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    nav_options = ["Overview", "Detect", "Docs", "Results", "Explain", "About"]
    with nav_col:
        nav_page = st.radio(
            "Main navigation",
            nav_options,
            horizontal=True,
            label_visibility="collapsed",
            index=0,
            key="text_signal_navigation",
        )

    available_models = get_available_models()
    with model_col:
        if not available_models:
            st.error("Saved models not found.")
            selected_model = None
        else:
            default_idx = available_models.index("Linear SVM") if "Linear SVM" in available_models else 0
            selected_model = st.selectbox(
                "Active model",
                options=available_models,
                index=default_idx,
                key="text_signal_active_model",
                label_visibility="collapsed",
            )
            # Keep the header compact; the selectbox retains its accessible "Active model" label.

    with logout_col:
        if st.button("Exit", key="text_signal_logout", use_container_width=True, help="Sign out of the research workspace"):
            st.session_state["text_signal_authenticated"] = False
            st.session_state.pop("text_signal_access_mode", None)
            st.rerun()

    st.markdown('<div style="height:1px;background:rgba(226,239,197,.16);margin:.25rem 0 1.2rem;"></div>', unsafe_allow_html=True)
    return nav_page, selected_model


# ── Page 0: Dashboard Overview ───────────────────────────────────────────────

def page_dashboard_overview(selected_model: str):
    """Render the first-screen dashboard summary after login."""
    render_page_heading(
        "00 / OVERVIEW",
        "A quick read of the system.",
        "An at-a-glance view of the benchmark, the three saved classifiers, and the tools available in this workspace.",
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Clean HC3 samples", "79,331", help="53,086 human-written and 26,245 AI-generated samples.")
    k2.metric("Held-out test set", "11,900", help="A stratified test split shared by the evaluated models.")
    k3.metric("Models evaluated", "03", help="Linear SVM, Logistic Regression, and XGBoost.")
    k4.metric("Top accuracy", "98.61%", help="Observed Linear SVM accuracy on the held-out benchmark.")

    st.markdown(" ")
    left, right = st.columns([1.2, .8], gap="large")
    with left:
        st.markdown('<div class="ff-section-label">MODEL SCORECARDS</div>', unsafe_allow_html=True)
        st.markdown("#### Three approaches. One benchmark.")
        model_cols = st.columns(3, gap="small")
        model_cards = [
            ("01", "Linear SVM", "98.61%", "F1  ·  0.9790", "TF-IDF + calibrated linear classifier"),
            ("02", "Logistic Regression", "98.53%", "F1  ·  0.9777", "TF-IDF baseline with class weighting"),
            ("03", "XGBoost", "96.95%", "F1  ·  0.9543", "21 engineered stylometric features"),
        ]
        for col, (number, name, accuracy, f1, detail) in zip(model_cols, model_cards):
            with col:
                render_feature_card(
                    f"MODEL {number} · {accuracy} ACCURACY",
                    name,
                    f"{f1} · {detail}",
                    accent=(number == "01"),
                )
    with right:
        st.markdown(
            """
            <div class="ff-lime-card" style="min-height:200px;">
              <div style="font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;">Start here</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.8rem;line-height:1;letter-spacing:-.055em;margin:14px 0 12px;">
                Try a passage.<br>Inspect the evidence.
              </div>
              <p style="font-size:.88rem;line-height:1.55;margin:0;">
                Run detection on a text sample, compare models, or upload a document for chunk-level review.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="margin-top:12px;border:1px solid rgba(226,239,197,.16);border-radius:16px;background:rgba(255,255,255,.02);padding:16px 17px;">
              <div class="ff-mini">RESEARCH NOTE</div>
              <p style="margin:.65rem 0 0;font-size:.84rem;">
                A high benchmark score does not establish who wrote a passage. Review errors and limitations before interpreting a prediction.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ff-section-label" style="margin-top:1.6rem;">WORKSPACE MAP</div>', unsafe_allow_html=True)
    route_cols = st.columns(5, gap="small")
    routes = [
        ("01", "Detect", "Paste text and inspect a prediction."),
        ("02", "Documents", "Upload TXT, PDF or DOCX."),
        ("03", "Results", "Compare metrics and domain scores."),
        ("04", "Explain", "Explore feature attribution."),
        ("05", "About", "Read the research method and caveats."),
    ]
    for col, (number, title, description) in zip(route_cols, routes):
        with col:
            render_feature_card(number, title, description)

    st.divider()
    render_page_heading(
        "WHY THIS MATTERS",
        "A score needs context.",
        "Human writing can be formal and structured; generated writing can be short or conversational. A classifier can be wrong, so the result should be read alongside the evidence and study limitations.",
    )
    challenge_col, approach_col = st.columns(2, gap="large")
    with challenge_col:
        render_feature_card(
            "THE CHALLENGE",
            "Style is not authorship.",
            "A high benchmark score does not guarantee reliable performance on unfamiliar domains, short passages, edited text, or newer language models.",
        )
    with approach_col:
        render_feature_card(
            "THE APPROACH",
            "Compare, then explain.",
            "Review model metrics, domain-level results, statistical tests and available text-attribution signals instead of relying on one label.",
            accent=True,
        )

    st.divider()
    render_page_heading(
        "THE WORKFLOW",
        "From passage to evidence.",
        "The existing app connects a text sample to a saved classifier and gives you context to review the model's response.",
    )
    step_cols = st.columns(4, gap="small")
    steps = [
        ("01 / INPUT", "Add a passage", "Paste text or select one of the supplied sample passages."),
        ("02 / MODEL", "Choose a classifier", "Use Linear SVM, Logistic Regression or XGBoost."),
        ("03 / OUTPUT", "Read the estimate", "Review the predicted class and the model confidence value."),
        ("04 / REVIEW", "Inspect the context", "Use feature explanations and limitations to interpret the result carefully."),
    ]
    for col, (number, title, description) in zip(step_cols, steps):
        with col:
            render_feature_card(number, title, description, accent=(number.startswith("04")))

    st.divider()
    render_page_heading(
        "WHAT YOU CAN EXPLORE",
        "One project. Four lenses.",
        "These sections are backed by existing project data and workflows rather than placeholder product features.",
    )
    lens_cols = st.columns(4, gap="small")
    lenses = [
        ("01 / BENCHMARK", "Model comparison", "Compare saved test metrics across three classifiers."),
        ("02 / GENERALIZATION", "Domain performance", "Look at how scores differ across HC3 source domains."),
        ("03 / EXPLAINABILITY", "Feature signals", "Inspect available SHAP plots and linear-model term contributions."),
        ("04 / DOCUMENTS", "Chunk analysis", "Upload TXT, PDF or DOCX for extraction and chunk-level estimates."),
    ]
    for col, (number, title, description) in zip(lens_cols, lenses):
        with col:
            render_feature_card(number, title, description)

    st.markdown(
        """
        <section class="ff-callout" aria-label="Important research limitation">
          <div class="ff-page-heading__eyebrow">A NOTE ON USE</div>
          <h3 style="margin:.25rem 0 .65rem;">This is a research signal, not a verdict.</h3>
          <p class="ff-page-heading__description" style="margin:0;">
            AI-text detectors can produce false positives and false negatives. Do not use this prototype as the sole basis for academic penalties, employment decisions or other consequential judgements; independent human review is essential.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


# ── Page 1: Home / Detection (Single Text Input) ──────────────────────────────

def page_home_detection(selected_model: str):
    """Render the primary detection workflow in the Flashform-inspired layout."""
    render_page_heading(
        "01 / DETECTION WORKSPACE",
        "Every passage has a pattern.",
        "Test a passage against the selected model, then examine the output and its strongest signals. The classification is a research estimate—not proof of authorship.",
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Active model", selected_model)
    k2.metric("Benchmark corpus", "HC3 · 2023")
    k3.metric("Cleaned samples", "79,331")

    presets = get_sample_presets()
    preset_choice = st.selectbox(
        "LOAD A SAMPLE PASSAGE",
        options=list(presets.keys()),
        index=0,
        help="Use a prepared sample to test the workflow without typing a passage.",
    )
    default_text = presets[preset_choice] if preset_choice in presets else ""

    main_col, side_col = st.columns([1.65, .85], gap="large")
    with main_col:
        st.markdown("#### The passage")
        text_input = st.text_area(
            "Text to analyze",
            value=default_text,
            height=255,
            placeholder="Paste or type a passage here…",
            help="The selected saved model will evaluate this text.",
            label_visibility="collapsed",
        )
        word_count = len(text_input.split())
        char_count = len(text_input)
        m1, m2 = st.columns(2)
        m1.caption(f"↳ {word_count:,} words")
        m2.caption(f"↳ {char_count:,} characters")
        analyze_btn = st.button("Run analysis  ↗", type="primary", use_container_width=True)

    with side_col:
        st.markdown(
            """
            <div class="ff-lime-card" style="min-height:210px;margin:.1rem 0 1rem;">
              <div style="font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.8rem;">
                HOW IT WORKS
              </div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.65rem;font-weight:700;letter-spacing:-.05em;line-height:1.05;margin-bottom:1rem;">
                Three models.<br>One clear view.
              </div>
              <p style="font-size:.88rem;line-height:1.55;margin:0;">
                Clean the input. Run the selected classifier. Review the signals that contributed to the result.
              </p>
            </div>
            <div style="border:1px solid rgba(226,239,197,.16);border-radius:16px;padding:16px 17px;background:rgba(255,255,255,.02);">
              <div class="ff-mini">AVAILABLE MODELS</div>
              <div style="margin-top:.7rem;line-height:2;color:#e6ecd9;font-size:.9rem;">
                <span style="color:#b1dd40;">01</span> &nbsp; Linear SVM<br>
                <span style="color:#b1dd40;">02</span> &nbsp; Logistic Regression<br>
                <span style="color:#b1dd40;">03</span> &nbsp; XGBoost
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Short passages, formal human writing, and unfamiliar domains can be misclassified.")

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

                explanations = []
                if model_code in ["svm", "lr"]:
                    pipeline_path = (
                        SVM_DIR / "svm_pipeline.joblib"
                        if model_code == "svm"
                        else LR_DIR / "lr_pipeline.joblib"
                    )
                    if pipeline_path.exists():
                        pipeline = joblib.load(pipeline_path)
                        explanations = get_lr_feature_weights(pipeline, text_input, top_n=10)

                if explanations:
                    render_explanation_card(explanations, model_name=selected_model)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


# ── Page 2: Document Analysis ────────────────────────────────────────────────

def page_document_analysis(selected_model: str):
    """Render document upload, multi-chunk segmentation, and document-level aggregation."""
    render_page_heading(
        "02 / DOCUMENT WORKSPACE",
        "Read beyond the first line.",
        "Upload an academic essay, report, or article (.txt, .pdf, .docx). The document will be segmented into coherent paragraphs, analyzed chunk-by-chunk, and aggregated into a document-level estimate.",
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
    render_global_styles()
    st.session_state.setdefault("text_signal_authenticated", False)

    if not st.session_state["text_signal_authenticated"]:
        render_login_screen()
        render_footer()
        return

    nav_page, selected_model = render_sidebar()
    render_header(compact=(nav_page != "Overview"))

    if selected_model is None:
        st.warning("Please ensure the saved model files exist in the models folder before running inference.")
        render_disclaimer()
        render_footer()
        return

    # Keep the contextual welcome strip on the Overview page only.
    if nav_page == "Overview":
        access_mode = st.session_state.get("text_signal_access_mode", "Configured account")
        st.markdown(
            f'<div class="ff-band"><strong>Workspace ready.</strong><p>Access mode: {access_mode} &nbsp; · &nbsp; Active model: {selected_model}. Select a section above to explore the detector.</p></div>',
            unsafe_allow_html=True,
        )

    if nav_page == "Overview":
        page_dashboard_overview(selected_model)
    elif nav_page == "Detect":
        page_home_detection(selected_model)
    elif nav_page == "Docs":
        page_document_analysis(selected_model)
    elif nav_page == "Results":
        render_metrics_dashboard()
    elif nav_page == "Explain":
        render_xai_deep_dive()
    elif nav_page == "About":
        render_methodology_tab()

    render_disclaimer()
    render_footer()


if __name__ == "__main__":
    main()
