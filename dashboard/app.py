"""
app.py — ABSA Dashboard Entry Point

Run with:
    streamlit run app.py
"""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ABSA Dashboard",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Fonts ─────────────────────────────────────────────────────────────────────

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined'
    ':opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet" />'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />',
    unsafe_allow_html=True,
)

# ── Global CSS ────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>

/* ── Base ─────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #FAFAFA;
    color: #1A1C1C;
}

.main .block-container {
    padding-top: 0 !important;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100%;
}

/* Push content up as far as Streamlit allows */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 1rem !important;
}

/* Sandbox heading override — Streamlit resets h tags */
.main h2 {
    font-size: 42px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #1A1C1C !important;
    line-height: 1.1 !important;
}

/* ── Sidebar shell ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FAFAFA !important;
    border-right: 1px solid #E5E7EB;
    min-width: 240px !important;
    max-width: 240px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

/* ── Nav links — target <a> directly (stable across versions) ─ */
[data-testid="stSidebar"] a {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.65rem 1rem;
    border-radius: 0 !important;
    border-left: 4px solid transparent;
    color: #464554 !important;
    text-decoration: none !important;
    transition: all 0.15s ease;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
}

/* Hover */
[data-testid="stSidebar"] a:hover {
    background-color: #F5F5FF;
    border-left: 4px solid transparent;
    color: #4648D4 !important;
}

/* Active page */
[data-testid="stSidebar"] a[aria-current="page"] {
    background-color: #EEEEFC;
    border-left: 4px solid #4648D4;
    color: #4648D4 !important;
    font-weight: 600;
}

/* Icon inside link inherits colour */
[data-testid="stSidebar"] a span {
    color: inherit !important;
}

/* ── Card ─────────────────────────────────────────────────── */
.card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    padding: 1.5rem;
}

/* ── Sentiment badges ─────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'Inter', sans-serif;
}

.badge-positive { background-color: #E8F5F3; color: #2A9D8F; }
.badge-negative { background-color: #FDF0EC; color: #E76F51; }

/* ── Aspect chips ─────────────────────────────────────────── */
.chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 0.25rem;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background-color: #EEEEEE;
    color: #464554;
    border: 1px solid #E5E7EB;
    font-family: 'Inter', sans-serif;
}

/* ── Label caps ───────────────────────────────────────────── */
.label-caps {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #767586;
    font-family: 'Inter', sans-serif;
}

/* ── Material Symbols ─────────────────────────────────────── */
.material-symbols-outlined {
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    vertical-align: middle;
    line-height: 1;
}

.icon-fill {
    font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}

/* ── Confidence colours ───────────────────────────────────── */
.conf-high { color: #2A9D8F; font-weight: 600; }
.conf-low  { color: #F59E0B; font-weight: 600; }
.conf-neg  { color: #E76F51; font-weight: 600; }

/* ── Primary button ───────────────────────────────────────── */
[data-testid="stButton"] > button {
    background-color: #4648D4 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 0.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}

[data-testid="stButton"] > button:hover { opacity: 0.9; }

/* ── Textarea ─────────────────────────────────────────────── */
[data-testid="stTextArea"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 0.5rem !important;
    background-color: #FFFFFF !important;
    color: #1A1C1C !important;
    caret-color: #1A1C1C !important;
    resize: none !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #4648D4 !important;
    box-shadow: 0 0 0 2px rgba(70,72,212,0.15) !important;
}

/* ── Hide Streamlit chrome ────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Page definitions ──────────────────────────────────────────────────────────

sandbox_page = st.Page("pages/sandbox.py",      title="Inference Sandbox",     default=True)
sentiment_page = st.Page("pages/sentiment_report.py", title="Sentiment Report")

pg = st.navigation([sandbox_page, sentiment_page], position="hidden")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Brand block
    st.markdown("""
        <div style="
            padding: 1.25rem 1.5rem 1rem 1.5rem;
            border-bottom: 1px solid #E5E7EB;
            margin-bottom: 0.5rem;
        ">
            <p style="
                font-size: 16px;
                font-weight: 700;
                color: #1A1C1C;
                margin: 0 0 2px 0;
                font-family: Inter, sans-serif;
            ">ABSA Dashboard</p>
            <p style="
                font-size: 12px;
                color: #767586;
                margin: 0;
                font-family: Inter, sans-serif;
            ">Laxman Pillai &bull; 2026</p>
        </div>
    """, unsafe_allow_html=True)

    # Nav links — file path strings, icons on st.page_link directly
    st.page_link("pages/sandbox.py",           label="Inference Sandbox", icon=":material/flare:")
    st.page_link("pages/sentiment_report.py",  label="Sentiment Report",  icon=":material/bar_chart:")

pg.run()