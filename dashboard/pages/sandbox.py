"""
pages/sandbox.py — Inference Sandbox Page

Live pipeline inference on user-submitted restaurant reviews.
Renders pipeline log, translation comparison, and extracted pairs table.

Model: Stage 3 — Binary SVM + SentenceTransformer (Macro-F1 0.8922)
Files: models/model_proposed.pkl, models/ohe_proposed.pkl,
       models/transformer_reference.txt
"""

import sys
import os
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Path setup ────────────────────────────────────────────────────────────────
# Structure: ASBA_Project_Code/dashboard/pages/sandbox.py
# ROOT      → ASBA_Project_Code/          (two levels up)
# SRC       → ASBA_Project_Code/src/      (pipeline.py and friends live here)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC  = os.path.join(ROOT, "src")

for p in (ROOT, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import run_pipeline

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.65
MAX_CHARS            = 500
MODEL_LABEL          = "ABSA-V2.4-HYBRID"

# ── Model loading — cached so it only runs once per session ───────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    models_dir = os.path.join(ROOT, "models")

    model = joblib.load(os.path.join(models_dir, "model_proposed.pkl"))
    ohe   = joblib.load(os.path.join(models_dir, "ohe_proposed.pkl"))

    transformer_name = open(
        os.path.join(models_dir, "transformer_reference.txt")
    ).read().strip()
    embedder = SentenceTransformer(transformer_name)

    return model, ohe, embedder


# ── Sentiment feature extraction (mirrors training notebook exactly) ──────────

def extract_sentiment_features_single(text: str) -> np.ndarray:
    """
    Replicates extract_sentiment_features() from 03_model_training.ipynb
    for a single text string at inference time.
    Returns shape (1, 6) numpy array.
    """
    pos_words      = {'good', 'great', 'excellent', 'amazing', 'love', 'best',
                      'delicious', 'fresh', 'wonderful', 'fantastic', 'perfect'}
    neg_words      = {'bad', 'terrible', 'awful', 'horrible', 'disappoint',
                      'slow', 'poor', 'bland', 'worst', 'rude', 'dirty'}
    negation_words = {'not', 'no', 'never', "n't", 'none', 'neither', 'nor', 'without'}
    contrast_words = {'but', 'however', 'though', 'although', 'despite', 'yet', 'while'}

    words            = str(text).lower().split()
    has_pos          = int(any(w in pos_words for w in words))
    has_neg          = int(any(w in neg_words for w in words))
    mixed_sentiment  = int(has_pos and has_neg)
    negation_density = sum(1 for w in words if w in negation_words) / max(len(words), 1)
    review_length    = len(words)
    has_contrast     = int(any(w in contrast_words for w in words))

    return np.array([[mixed_sentiment, negation_density, review_length,
                      has_contrast, has_pos, has_neg]], dtype=float)


# ── Context window builder (mirrors notebook) ─────────────────────────────────

def build_context_window(text: str, opinion_word: str, window: int = 5) -> str:
    words = text.split()
    for i, w in enumerate(words):
        if opinion_word.lower() in w.lower():
            start = max(0, i - window)
            end   = min(len(words), i + window + 1)
            return ' '.join(words[start:end])
    return text  # fallback: full text


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(pipeline_result: dict, model, ohe, embedder) -> list[dict]:
    """
    Runs sentiment classification for each (category, opinion_word) pair.
    Returns list of dicts with keys:
        category, opinion_word, sentiment, confidence, low_confidence_flag
    """
    text_for_model = (
        pipeline_result.get("translated")
        or pipeline_result.get("normalised")
        or pipeline_result.get("original")
    )

    rows = []
    for category, opinion_word in pipeline_result["results"]:
        context   = build_context_window(text_for_model, opinion_word)
        embedding = embedder.encode([context])  # shape (1, 384)

        sent_feats  = extract_sentiment_features_single(text_for_model)  # (1, 6)
        cat_encoded = ohe.transform([[category]])                         # (1, 5) sparse

        emb_sparse  = csr_matrix(embedding)
        sent_sparse = csr_matrix(sent_feats)
        features    = hstack([emb_sparse, sent_sparse, cat_encoded])     # (1, 395)

        proba      = model.predict_proba(features)[0]
        confidence = float(proba.max())
        sentiment  = model.classes_[proba.argmax()]

        rows.append({
            "category":           category.upper(),
            "opinion_word":       opinion_word,
            "sentiment":          sentiment,
            "confidence":         confidence,
            "low_confidence_flag": confidence < CONFIDENCE_THRESHOLD,
        })

    return rows


# ── Pipeline log renderer ─────────────────────────────────────────────────────

PIPELINE_STEPS = [
    ("Retrieving",    "Fetched raw text input."),
    ("Normalising",   "Cleaning special characters."),
    ("Translating",   "Manglish to English via NMT."),
    ("Extracting",    "Identifying aspect-opinion pairs."),
    ("Categorising",  "Mapping to hospitality domains."),
    ("Predicting",    "Calculating sentiment scores."),
]

def render_pipeline_log(completed_steps: int):
    """Renders pipeline log with tick icons up to completed_steps."""
    items = ""
    for i, (name, desc) in enumerate(PIPELINE_STEPS):
        if i < completed_steps:
            icon_color = "#2A9D8F"
            icon       = "check_circle"
            fill       = "1"
            name_style = "color:#1A1C1C;"
        else:
            icon_color = "#C7C4D7"
            icon       = "radio_button_unchecked"
            fill       = "0"
            name_style = "color:#767586;"

        items += f"""
        <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:16px;position:relative;">
            <span class="material-symbols-outlined" style="
                font-variation-settings:'FILL' {fill},'wght' 400,'GRAD' 0,'opsz' 24;
                color:{icon_color};font-size:20px;margin-top:1px;flex-shrink:0;">
                {icon}
            </span>
            <div>
                <p style="margin:0;font-size:14px;font-weight:600;{name_style}">{name}</p>
                <p style="margin:0;font-size:12px;color:#767586;">{desc}</p>
            </div>
        </div>
        """

    st.markdown(f"""
        <div style="
            background:#FFFFFF;
            border:1px solid #E5E7EB;
            border-radius:1rem;
            box-shadow:0 1px 3px rgba(0,0,0,0.05);
            padding:1.5rem;
            min-height:100%;
            height:100%;
        ">
            <p class="label-caps" style="margin-bottom:1.25rem;">Pipeline Log</p>
            {items}
        </div>
    """, unsafe_allow_html=True)


# ── Results table renderer ────────────────────────────────────────────────────

def render_results_table(rows: list[dict]):
    COLS = "1fr 1.5fr 1fr 1fr"

    rows_html = ""
    for row in rows:
        conf_pct = f"{row['confidence']*100:.0f}%"

        chip  = f'<span style="display:inline-block;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;background:#EEEEEE;color:#464554;border:1px solid #E5E7EB;font-family:Inter,sans-serif;">{row["category"]}</span>'

        if row["sentiment"] == "positive":
            badge = '<span style="display:inline-block;padding:2px 10px;border-radius:9999px;font-size:11px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;background:#E8F5F3;color:#2A9D8F;font-family:Inter,sans-serif;">Positive</span>'
        else:
            badge = '<span style="display:inline-block;padding:2px 10px;border-radius:9999px;font-size:11px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;background:#FDF0EC;color:#E76F51;font-family:Inter,sans-serif;">Negative</span>'

        if row["low_confidence_flag"]:
            conf_html = f'<span style="color:#F59E0B;font-weight:600;font-family:Inter,sans-serif;">&#9888; {conf_pct}</span>'
        elif row["sentiment"] == "positive":
            conf_html = f'<span style="color:#2A9D8F;font-weight:600;font-family:Inter,sans-serif;">{conf_pct}</span>'
        else:
            conf_html = f'<span style="color:#E76F51;font-weight:600;font-family:Inter,sans-serif;">{conf_pct}</span>'

        rows_html += (
            '<div style="display:grid;grid-template-columns:' + COLS + ';'
            'padding:14px 0;border-bottom:1px solid #F3F3F3;align-items:center;">'
            '<div>' + chip + '</div>'
            '<div style="font-size:14px;color:#1A1C1C;font-family:Inter,sans-serif;">' + row["opinion_word"] + '</div>'
            '<div>' + badge + '</div>'
            '<div>' + conf_html + '</div>'
            '</div>'
        )

    label_style = 'style="font-size:11px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;color:#767586;font-family:Inter,sans-serif;"'

    html = (
        '<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:1rem;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.05);overflow:hidden;margin-top:1.5rem;">'

        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:1.25rem 1.5rem 0.75rem 1.5rem;border-bottom:1px solid #E5E7EB;">'
        '<span style="font-size:16px;font-weight:600;color:#1A1C1C;font-family:Inter,sans-serif;">'
        'Extracted Pairs &amp; Sentiment</span>'
        '<span style="font-size:11px;font-weight:700;letter-spacing:0.05em;background:#EEEEFC;'
        'color:#4648D4;padding:3px 10px;border-radius:9999px;font-family:Inter,sans-serif;">'
        'MODEL: ' + MODEL_LABEL + '</span>'
        '</div>'

        '<div style="padding:0 1.5rem;">'
        '<div style="display:grid;grid-template-columns:' + COLS + ';'
        'padding:10px 0;border-bottom:1px solid #E5E7EB;">'
        '<span ' + label_style + '>Category</span>'
        '<span ' + label_style + '>Opinion Word</span>'
        '<span ' + label_style + '>Sentiment</span>'
        '<span ' + label_style + '>Confidence</span>'
        '</div>'

        + rows_html +

        '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# ── Page ──────────────────────────────────────────────────────────────────────

def main():

    # Equal height columns for input + log cards
    st.markdown("""
        <style>
        [data-testid="stHorizontalBlock"] { align-items: stretch; }
        [data-testid="stHorizontalBlock"] > div {
            display: flex;
            flex-direction: column;
        }
        </style>
    """, unsafe_allow_html=True)

    # Page header
    st.markdown("""
        <style>
        /* Equal height columns */
        [data-testid="stHorizontalBlock"] { align-items: stretch; }
        [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
            display: flex;
            flex-direction: column;
        }
        /* Input card fills its column */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(textarea) {
            flex: 1;
        }
        </style>

        <div style="margin-bottom:1.5rem;">
            <div style="font-size:28px;font-weight:700;letter-spacing:-0.02em;
                        color:#1A1C1C;margin:0 0 6px 0;line-height:1.2;
                        font-family:'Inter',sans-serif;">Sandbox</div>
            <div style="font-size:14px;color:#464554;font-family:'Inter',sans-serif;">
                Enter natural language restaurant reviews to test the
                Aspect-Based Sentiment Analysis pipeline.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Load models ───────────────────────────────────────────────────────────
    with st.spinner("Loading model…"):
        try:
            model, ohe, embedder = load_models()
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.stop()

    # ── Input + pipeline log — 8 / 4 column split ────────────────────────────
    col_input, col_log = st.columns([8, 4], gap="large")

    with col_input:
        st.markdown("""
            <style>
            div[data-testid="stVerticalBlockBorderWrapper"]:has(textarea) {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                padding: 1.25rem 1.5rem 1.5rem 1.5rem;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:0.5rem;">
                <span class="label-caps">Your Review</span>
                <span style="font-size:12px;color:#767586;">Limit: 500 characters</span>
            </div>
        """, unsafe_allow_html=True)

        review_text = st.text_area(
            label="review_input",
            label_visibility="collapsed",
            placeholder="e.g. Makanan sedap gila tapi service lambat...",
            max_chars=MAX_CHARS,
            height=160,
            key="review_input",
        )

        _, btn_col = st.columns([6, 2])
        with btn_col:
            analyse_clicked = st.button(
                "✦ Analyse Review",
                type="primary",
                use_container_width=True,
            )

    # ── Pipeline log — idle state ─────────────────────────────────────────────
    with col_log:
        log_placeholder = st.empty()
        with log_placeholder:
            render_pipeline_log(completed_steps=0)

    # ── Run inference on button click ─────────────────────────────────────────
    if analyse_clicked:
        text = review_text.strip()
        if not text:
            st.markdown("""
                <div style="margin-top:1rem;padding:0.75rem 1rem;background:#FDF0EC;
                            border:1px solid #E76F51;border-radius:0.5rem;
                            font-size:14px;color:#E76F51;font-weight:500;">
                    ⚠ Please enter a review before analysing.
                </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Animate pipeline log step by step
        import time
        for step in range(1, len(PIPELINE_STEPS) + 1):
            with log_placeholder:
                render_pipeline_log(completed_steps=step)
            if step < len(PIPELINE_STEPS):
                time.sleep(0.35)

        # Run the actual pipeline
        try:
            pipeline_result = run_pipeline(text, mode="auto")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

        # Run sentiment inference
        try:
            rows = run_inference(pipeline_result, model, ohe, embedder)
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

        # ── Translation comparison ────────────────────────────────────────────
        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

        original_text    = pipeline_result.get("original", text)
        translated_text  = pipeline_result.get("translated") or pipeline_result.get("normalised") or original_text
        was_translated   = pipeline_result.get("was_translated", False)

        tcol_orig, tcol_trans = st.columns(2, gap="large")

        with tcol_orig:
            st.markdown(f"""
                <div style="
                    background:#FFFFFF;border:1px solid #E5E7EB;
                    border-radius:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);
                    padding:1.25rem 1.5rem;
                ">
                    <p class="label-caps" style="margin-bottom:0.75rem;">Original Text</p>
                    <div style="
                        background:#F3F3F3;border:1px solid #E5E7EB;
                        border-radius:0.5rem;padding:1rem;
                        font-size:14px;color:#1A1C1C;line-height:1.6;
                    ">{original_text}</div>
                </div>
            """, unsafe_allow_html=True)

        with tcol_trans:
            trans_label = "Translated Text" if was_translated else "Normalised Text"
            st.markdown(f"""
                <div style="
                    background:#FFFFFF;border:1px solid #E5E7EB;
                    border-radius:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);
                    padding:1.25rem 1.5rem;
                ">
                    <p class="label-caps" style="margin-bottom:0.75rem;">{trans_label}</p>
                    <div style="
                        background:#F3F3F3;border:1px solid #E5E7EB;
                        border-radius:0.5rem;padding:1rem;
                        font-size:14px;color:#1A1C1C;line-height:1.6;font-style:italic;
                    ">{translated_text}</div>
                </div>
            """, unsafe_allow_html=True)

        # ── Results table ─────────────────────────────────────────────────────
        if not rows:
            st.markdown("""
                <div style="margin-top:2rem;padding:1rem 1.25rem;background:#F3F3F3;
                            border:1px solid #E5E7EB;border-radius:0.5rem;
                            font-size:14px;color:#767586;">
                    No aspect-opinion pairs could be extracted from this review.
                </div>
            """, unsafe_allow_html=True)
        else:
            render_results_table(rows)


main()