"""
pages/sentiment_report.py — Sentiment Report Page

Offline analytics dashboard loaded from dashboard_data.csv.
Shows per-restaurant KPIs, charts, word clouds, and review feed.

Data file: dashboard/dashboard_data.csv
Charts:    components/charts.py
"""

import sys
import os
import pandas as pd
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DASH_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMP_DIR  = os.path.join(DASH_DIR, "components")

for p in (ROOT, DASH_DIR, COMP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from components.charts import make_donut, make_aspect_bars, make_keyword_bars

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(ROOT, "data", "dashboard_data.csv")
RESTAURANTS = [
    "Jom Corner (J Corner)",
    "Raihana One Bistro",
    "SABA Restaurant Cyberjaya",
]

# ── Health score tier ─────────────────────────────────────────────────────────

def health_tier(pct: float) -> tuple[str, str]:
    """Returns (label, colour) for a given positive percentage."""
    if pct >= 75:
        return "Good", "#2A9D8F"
    elif pct >= 50:
        return "Average", "#F59E0B"
    else:
        return "Poor", "#E76F51"

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["sentiment"]          = df["sentiment"].str.lower().str.strip()
    df["aspect_category"]    = df["aspect_category"].str.lower().str.strip()
    df["opinion_word"]       = df["opinion_word"].str.lower().str.strip()
    return df


def filter_restaurant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["place_name"] == name].copy()

# ── KPI computations ──────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    total_reviews   = df["review_id"].nunique()
    total_pairs     = len(df)
    pos_pairs       = (df["sentiment"] == "positive").sum()
    neg_pairs       = (df["sentiment"] == "negative").sum()
    pos_pct         = round(pos_pairs / total_pairs * 100) if total_pairs else 0

    # Most discussed aspect — by unique review count
    aspect_review_counts = (
        df.groupby("aspect_category")["review_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    top_aspect       = aspect_review_counts.index[0].title() if len(aspect_review_counts) else "—"
    top_aspect_count = int(aspect_review_counts.iloc[0]) if len(aspect_review_counts) else 0

    label, colour = health_tier(pos_pct)

    return {
        "total_reviews":    total_reviews,
        "total_pairs":      total_pairs,
        "pos_pairs":        int(pos_pairs),
        "neg_pairs":        int(neg_pairs),
        "pos_pct":          pos_pct,
        "health_label":     label,
        "health_colour":    colour,
        "top_aspect":       top_aspect,
        "top_aspect_count": top_aspect_count,
    }

# ── Chart data computations ───────────────────────────────────────────────────

def compute_aspect_data(df: pd.DataFrame) -> list[dict]:
    """Returns aspect bar data ordered by descending total mentions."""
    grp = (
        df.groupby(["aspect_category", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    grp = grp[~grp["aspect_category"].isin(["other"])]
    if "positive" not in grp.columns:
        grp["positive"] = 0
    if "negative" not in grp.columns:
        grp["negative"] = 0

    grp["total"] = grp["positive"] + grp["negative"]
    grp = grp.sort_values("total", ascending=False)

    return [
        {
            "category":  row["aspect_category"],
            "pos_count": int(row["positive"]),
            "neg_count": int(row["negative"]),
        }
        for _, row in grp.iterrows()
    ]


def compute_word_cloud(df: pd.DataFrame) -> dict:
    """
    Returns a frequency dict of all opinion words regardless of sentiment,
    top 40, for a single combined word cloud.
    """
    counts = Counter(df["opinion_word"].dropna()).most_common(40)
    return {word: count for word, count in counts}


def compute_keywords(df: pd.DataFrame, top_n: int = 8) -> list[dict]:
    """Top negative opinion words by frequency, filtered for semantic negativity."""
    # Words that shouldn't appear in negative keywords regardless of model label
    POSITIVE_STOPLIST = {
        "good", "great", "nice", "best", "delicious", "excellent", "amazing",
        "wonderful", "fantastic", "perfect", "fresh", "tasty", "cheap",
        "affordable", "friendly", "clean", "comfortable", "fast", "quick",
        "recommend", "love", "like", "enjoy", "happy", "satisf", "pleasing",
        "helpful", "polite", "warm", "cozy", "many", "much", "very",
        "quite", "really", "always", "also", "still", "even", "well",
    }

    neg_words = df[df["sentiment"] == "negative"]["opinion_word"].dropna()
    # Filter out words that are in the stoplist or are too short
    filtered = [
        w for w in neg_words
        if w.lower() not in POSITIVE_STOPLIST
        and len(w) > 2
        and not any(w.lower().startswith(pos) for pos in ["good", "great", "nice"])
    ]
    counts = Counter(filtered).most_common(top_n)
    return [{"word": w, "count": c} for w, c in counts]


def compute_aspect_scores(df: pd.DataFrame) -> list[dict]:
    """Per-aspect health score tiles, filtered aspects only."""
    EXCLUDED = {"other", "overall"}
    result = []
    for aspect in df["aspect_category"].unique():
        if aspect in EXCLUDED:
            continue
        adf       = df[df["aspect_category"] == aspect]
        total     = len(adf)
        pos       = (adf["sentiment"] == "positive").sum()
        pos_pct   = round(pos / total * 100) if total else 0
        label, colour = health_tier(pos_pct)
        result.append({
            "aspect":   aspect.title(),
            "pos_pct":  pos_pct,
            "label":    label,
            "colour":   colour,
            "total":    total,
        })
    # Sort by pos_pct descending
    return sorted(result, key=lambda x: x["pos_pct"], reverse=True)

# ── HTML renderers ────────────────────────────────────────────────────────────

def render_kpi_row(kpis: dict):
    health_colour = kpis["health_colour"]
    pos_pct       = kpis["pos_pct"]

    html = (
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.5rem;margin-bottom:1.5rem;">'

        # Total Reviews
        '<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:1rem;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.05);padding:1.5rem;">'
        '<p style="font-size:12px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'
        'color:#767586;margin:0 0 8px 0;font-family:Inter,sans-serif;">Total Reviews</p>'
        '<p style="font-size:32px;font-weight:700;color:#1A1C1C;margin:0;font-family:Inter,sans-serif;">'
        + str(kpis["total_reviews"]) +
        '</p></div>'

        # Brand Health Score
        '<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:1rem;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.05);padding:1.5rem;">'
        '<p style="font-size:12px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'
        'color:#767586;margin:0 0 8px 0;font-family:Inter,sans-serif;">Brand Health Score</p>'
        '<div style="display:flex;align-items:flex-end;gap:8px;margin-bottom:10px;">'
        '<span style="font-size:32px;font-weight:700;color:#1A1C1C;font-family:Inter,sans-serif;">'
        + str(pos_pct) + '%</span>'
        '<span style="font-size:12px;font-weight:700;color:' + health_colour + ';'
        'margin-bottom:6px;font-family:Inter,sans-serif;">' + kpis["health_label"] + '</span>'
        '</div>'
        '<div style="width:100%;height:6px;background:#E5E7EB;border-radius:9999px;overflow:hidden;">'
        '<div style="height:100%;width:' + str(pos_pct) + '%;background:' + health_colour + ';'
        'border-radius:9999px;"></div>'
        '</div></div>'

        # Most Discussed Aspect
        '<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:1rem;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.05);padding:1.5rem;">'
        '<p style="font-size:12px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'
        'color:#767586;margin:0 0 8px 0;font-family:Inter,sans-serif;">Most Discussed Aspect</p>'
        '<p style="font-size:32px;font-weight:700;color:#4648D4;margin:0 0 12px 0;'
        'font-family:Inter,sans-serif;">' + kpis["top_aspect"] + '</p>'
        '<p style="font-size:12px;color:#767586;margin:0;font-family:Inter,sans-serif;">'
        '&#128172; Mentioned in ' + str(kpis["top_aspect_count"]) + ' reviews'
        '</p></div>'

        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_word_cloud(freq_dict: dict):
    """Renders a word cloud with title label above and image below."""
    from wordcloud import WordCloud, STOPWORDS

    st.markdown(
        '<p style="font-size:12px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'
        'color:#767586;margin:0 0 0.75rem 0;font-family:Inter,sans-serif;">Most Frequent Opinion Words</p>',
        unsafe_allow_html=True
    )

    if not freq_dict:
        st.markdown('<p style="color:#767586;font-family:Inter,sans-serif;">No data.</p>', unsafe_allow_html=True)
        return

    CUSTOM_STOPS = STOPWORDS | {"many", "other"}
    filtered_dict = {w: c for w, c in freq_dict.items() if w.lower() not in CUSTOM_STOPS}

    wc = WordCloud(
        width=1200,
        height=280,
        background_color="#F5F5FF",
        color_func=None,
        colormap="viridis",
        prefer_horizontal=0.85,
        margin=4,
    ).fit_words(filtered_dict)

    st.image(wc.to_array(), use_container_width=True)


def render_review_feed(df: pd.DataFrame):
    """Renders top 6 curated review cards — no other/overall pairs, 2-4 pairs, sorted by mean confidence desc."""
    reviews = (
        df.groupby("review_id")
        .agg(
            review_text=("review_text", "first"),
            pairs=("opinion_word", list),
            categories=("aspect_category", list),
            sentiments=("sentiment", list),
            confidences=("confidence", list),
            low_conf=("low_confidence_flag", list),
        )
        .reset_index()
    )

    EXCLUDED_CATS = {"other", "overall"}
    reviews = reviews[
        reviews["categories"].apply(lambda cats: not any(c in EXCLUDED_CATS for c in cats))
    ].copy()
    reviews = reviews[reviews["pairs"].apply(lambda p: 2 <= len(p) <= 4)].copy()
    reviews["mean_conf"] = reviews["confidences"].apply(
        lambda cs: sum(float(c) for c in cs) / len(cs)
    )
    reviews = reviews.sort_values("mean_conf", ascending=False).head(6)

    # Header
    st.markdown(
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'margin-bottom:0.75rem;">'
        '<p style="font-size:16px;font-weight:600;color:#1A1C1C;margin:0;font-family:Inter,sans-serif;">Review Feed</p>'
        '<span style="background:#EEEEEE;color:#1A1C1C;font-size:12px;font-weight:700;'
        'padding:3px 10px;border-radius:9999px;font-family:Inter,sans-serif;">Top 6</span>'
        '</div>',
        unsafe_allow_html=True
    )

    # 2-column grid of review cards
    cols = st.columns(2, gap="medium")
    for i, (_, review) in enumerate(reviews.iterrows()):
        col = cols[i % 2]
        n_pairs = len(review["pairs"])

        pairs_html = ""
        for word, cat, sent, conf, low in zip(
            review["pairs"], review["categories"],
            review["sentiments"], review["confidences"], review["low_conf"]
        ):
            sent_colour = "#2A9D8F" if sent == "positive" else "#E76F51"
            sent_label  = "Positive" if sent == "positive" else "Negative"
            warn        = "⚠ " if low else ""

            pairs_html += (
                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'padding:5px 0;border-bottom:1px solid #F3F3F3;">'
                '<div style="display:flex;align-items:center;gap:6px;flex:1;">'
                '<span style="font-size:10px;font-weight:700;letter-spacing:0.05em;'
                'text-transform:uppercase;background:#EEEEEE;color:#464554;'
                'padding:1px 6px;border-radius:3px;font-family:Inter,sans-serif;">'
                + cat.upper() + '</span>'
                '<span style="font-size:12px;color:#464554;font-family:Inter,sans-serif;">'
                '&#8594; &#34;' + word + '&#34;</span>'
                '</div>'
                '<span style="font-size:11px;font-weight:700;color:' + sent_colour + ';'
                'font-family:Inter,sans-serif;white-space:nowrap;">'
                + warn + sent_label + '</span>'
                '</div>'
            )

        card_html = (
            '<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:0.75rem;'
            'padding:1rem;margin-bottom:1rem;">'
            '<p style="font-size:11px;color:#767586;margin:0 0 8px 0;font-family:Inter,sans-serif;">'
            + str(n_pairs) + ' pair' + ('s' if n_pairs != 1 else '') + '</p>'
            '<p style="font-size:13px;color:#1A1C1C;font-style:italic;margin:0 0 10px 0;'
            'line-height:1.5;font-family:Inter,sans-serif;">'
            '&#34;' + str(review["review_text"])[:160] + ('…' if len(str(review["review_text"])) > 160 else '') + '&#34;'
            '</p>'
            + pairs_html +
            '</div>'
        )
        col.markdown(card_html, unsafe_allow_html=True)


# ── Page ──────────────────────────────────────────────────────────────────────

def main():

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"dashboard_data.csv not found at: {DATA_PATH}")
        st.stop()

    # ── Session state ─────────────────────────────────────────────────────────
    if "selected_restaurant" not in st.session_state:
        st.session_state["selected_restaurant"] = RESTAURANTS[0]

    # ── Page-level CSS ────────────────────────────────────────────────────────
    st.markdown("""
        <style>
        /* Plotly chart iframe — remove default padding */
        .stPlotlyChart { padding: 0 !important; }
        .stPlotlyChart > div { padding: 0 !important; }

        /* Remove streamlit column gap compensation */
        [data-testid="stHorizontalBlock"] { gap: 1.5rem; }
        </style>
    """, unsafe_allow_html=True)

    # ── Restaurant selector ───────────────────────────────────────────────────
    review_counts = {
        name: df[df["place_name"] == name]["review_id"].nunique()
        for name in RESTAURANTS
    }

    # Short display names for the buttons
    SHORT_NAMES = {
        "Jom Corner (J Corner)":     "Jom Corner",
        "Raihana One Bistro":        "Raihana One Bistro",
        "SABA Restaurant Cyberjaya": "SABA Restaurant",
    }

    # CSS for selector buttons
    st.markdown("""
        <style>
        /* Active restaurant button */
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background-color: #EEEEFC !important;
            color: #4648D4 !important;
            border: 2px solid #4648D4 !important;
            border-radius: 0.75rem !important;
            font-family: Inter, sans-serif !important;
            font-weight: 600 !important;
            padding: 0.6rem 1rem !important;
            width: 100% !important;
        }
        /* Inactive restaurant button */
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background-color: #F3F3F3 !important;
            color: #464554 !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 0.75rem !important;
            font-family: Inter, sans-serif !important;
            font-weight: 500 !important;
            padding: 0.6rem 1rem !important;
            width: 100% !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
            background-color: #EEEEFC !important;
            border-color: #4648D4 !important;
            color: #4648D4 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:12px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'
        'color:#767586;font-family:Inter,sans-serif;margin-bottom:0.5rem;">Restaurant</p>',
        unsafe_allow_html=True
    )

    sel = st.session_state["selected_restaurant"]
    btn_cols = st.columns(3, gap="medium")

    for i, name in enumerate(RESTAURANTS):
        short   = SHORT_NAMES[name]
        count   = review_counts[name]
        active  = name == sel
        label   = f"**{short}**\n{count} reviews" if active else f"{short}\n{count} reviews"
        with btn_cols[i]:
            if st.button(
                label,
                key=f"rest_btn_{i}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                if not active:
                    st.session_state["selected_restaurant"] = name
                    st.rerun()

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    selected = st.session_state["selected_restaurant"]

    # ── Filter data ───────────────────────────────────────────────────────────
    rdf    = filter_restaurant(df, selected)
    kpis   = compute_kpis(rdf)

    # ── Restaurant name heading ───────────────────────────────────────────────
    display_name = selected.split("(")[0].strip() if "(" in selected else selected
    st.markdown(
        '<div style="margin-bottom:1.5rem;">'
        '<div style="font-size:32px;font-weight:700;letter-spacing:-0.02em;'
        'color:#1A1C1C;font-family:Inter,sans-serif;line-height:1.2;">'
        + display_name +
        '</div></div>',
        unsafe_allow_html=True
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    render_kpi_row(kpis)

    # ── Charts row ────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns([5, 7], gap="large")

    # Compute data first so we can calculate heights
    aspect_data   = compute_aspect_data(rdf)
    n_cats        = len(aspect_data)

    CARD_HEIGHT  = 380   # fixed — donut + legend only, no tiles
    ASPECT_FIG_H = 300   # chart height inside the card (card - title - padding)

    with chart_col1:
        pos_pct   = kpis["pos_pct"]
        neg_pct   = 100 - pos_pct
        donut_fig = make_donut(kpis["pos_pairs"], kpis["neg_pairs"])

        legend_html = (
            '<div style="display:flex;flex-direction:column;gap:10px;margin-top:0.5rem;">'
            '<div style="display:flex;align-items:center;gap:8px;">'
            '<div style="width:10px;height:10px;border-radius:2px;background:#2A9D8F;flex-shrink:0;"></div>'
            '<div>'
            '<p style="margin:0;font-size:12px;font-weight:700;font-family:Inter,sans-serif;">'
            + str(pos_pct) + '% Positive</p>'
            '<p style="margin:0;font-size:11px;color:#767586;font-family:Inter,sans-serif;">'
            + str(kpis["pos_pairs"]) + ' pairs</p>'
            '</div></div>'
            '<div style="display:flex;align-items:center;gap:8px;">'
            '<div style="width:10px;height:10px;border-radius:2px;background:#E76F51;flex-shrink:0;"></div>'
            '<div>'
            '<p style="margin:0;font-size:12px;font-weight:700;font-family:Inter,sans-serif;">'
            + str(neg_pct) + '% Negative</p>'
            '<p style="margin:0;font-size:11px;color:#767586;font-family:Inter,sans-serif;">'
            + str(kpis["neg_pairs"]) + ' pairs</p>'
            '</div></div>'
            '</div>'
        )

        donut_inner = donut_fig.to_html(
            full_html=False, include_plotlyjs="cdn",
            config={"displayModeBar": False}
        )
        donut_inner = donut_inner.replace(
            '<div>', '<div style="margin:0 auto;width:fit-content;">', 1
        )

        donut_card = (
            '<html><head>'
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">'
            '</head><body style="margin:0;padding:0;background:transparent;">'
            '<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:1rem;'
            'box-shadow:0 1px 3px rgba(0,0,0,0.05);padding:1.25rem;box-sizing:border-box;font-family:Inter,sans-serif;">'
            '<p style="font-size:16px;font-weight:600;color:#1A1C1C;margin:0 0 0.75rem 0;">Aspect Sentiment Distribution</p>'
            + donut_inner + legend_html +
            '</div></body></html>'
        )
        components.html(donut_card, height=CARD_HEIGHT, scrolling=False)

    with chart_col2:
        aspect_fig   = make_aspect_bars(aspect_data, fig_height=ASPECT_FIG_H)
        aspect_inner = aspect_fig.to_html(
            full_html=False, include_plotlyjs="cdn",
            config={"displayModeBar": False}
        )
        aspect_card = (
            '<html><head>'
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">'
            '</head><body style="margin:0;padding:0;background:transparent;">'
            '<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:1rem;'
            'box-shadow:0 1px 3px rgba(0,0,0,0.05);padding:1.25rem;box-sizing:border-box;font-family:Inter,sans-serif;">'
            '<p style="font-size:16px;font-weight:600;color:#1A1C1C;margin:0 0 0.75rem 0;">Sentiment Per Aspect</p>'
            + aspect_inner +
            '</div></body></html>'
        )
        components.html(aspect_card, height=CARD_HEIGHT, scrolling=False)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Word cloud ────────────────────────────────────────────────────────────
    wc_data = compute_word_cloud(rdf)
    render_word_cloud(wc_data)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Common words per aspect ───────────────────────────────────────────────
    # Fixed order 2x2 grid — food, price, ambiance, service
    # Each card shows top 4 opinion words regardless of sentiment
    ASPECT_ORDER = ["food", "price", "ambiance", "service"]

    aspect_cards_html = ""
    for aspect in ASPECT_ORDER:
        adf       = rdf[rdf["aspect_category"] == aspect]
        top_words = Counter(adf["opinion_word"].dropna()).most_common(4)

        keywords_html = ""
        for word, count in top_words:
            keywords_html += (
                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'padding:5px 0;border-bottom:1px solid #F3F3F3;">'
                '<span style="font-size:13px;color:#1A1C1C;font-family:Inter,sans-serif;">' + word + '</span>'
                '<span style="font-size:11px;font-weight:600;color:#767586;font-family:Inter,sans-serif;">'
                + str(count) + ' mentions</span>'
                '</div>'
            )

        aspect_cards_html += (
            '<div style="background:#FAFAFA;border:1px solid #E5E7EB;border-radius:0.75rem;padding:1rem;">'
            '<p style="font-size:11px;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;'
            'color:#4648D4;margin:0 0 0.75rem 0;font-family:Inter,sans-serif;">' + aspect.upper() + '</p>'
            + keywords_html +
            '</div>'
        )

    drilldown_html = (
        '<div style="background:#FFFFFF;border:1px solid #E5E7EB;border-radius:1rem;'
        'box-shadow:0 1px 3px rgba(0,0,0,0.05);padding:1.5rem;">'
        '<p style="font-size:16px;font-weight:600;color:#1A1C1C;margin:0 0 0.25rem 0;font-family:Inter,sans-serif;">'
        'Common Words Per Aspect</p>'
        '<p style="font-size:12px;color:#767586;margin:0 0 1.25rem 0;font-family:Inter,sans-serif;">'
        'Most frequently mentioned opinion words per aspect category</p>'
        '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;">'
        + aspect_cards_html +
        '</div></div>'
    )
    st.markdown(drilldown_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Review feed ───────────────────────────────────────────────────────────
    render_review_feed(rdf)

main()