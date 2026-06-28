"""
components/charts.py
====================
Plotly figure builders for the Sentiment Report page.

Colour tokens (locked):
    Positive : #2A9D8F  (teal)
    Negative : #E76F51  (coral)
    Primary  : #4648D4  (indigo)
"""

import plotly.graph_objects as go

# ── Colour constants ──────────────────────────────────────────────────────────

C_POS    = "#2A9D8F"
C_NEG    = "#E76F51"
C_INDIGO = "#4648D4"
C_TEXT   = "#1A1C1C"
C_MUTED  = "#767586"
C_BORDER = "#E5E7EB"
FONT     = "Inter, sans-serif"


def _base_layout(**kwargs) -> dict:
    """Shared layout defaults — transparent bg, Inter font.
    Pass margin= here to override the default zero margin."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family=FONT, color=C_TEXT),
        showlegend=False,
    )
    # Allow callers to override margin by passing it in kwargs
    base.update(kwargs)
    return base


# ── 1. Donut chart ────────────────────────────────────────────────────────────

def make_donut(pos_count: int, neg_count: int) -> go.Figure:
    total   = pos_count + neg_count

    fig = go.Figure(go.Pie(
        values=[pos_count, neg_count],
        labels=["Positive", "Negative"],
        hole=0.72,
        marker=dict(
            colors=[C_POS, C_NEG],
            line=dict(color="#FFFFFF", width=2),
        ),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "%{value} pairs<br>"
            "%{percent}<extra></extra>"
        ),
        textinfo="none",
        direction="clockwise",
        sort=False,
    ))

    fig.add_annotation(
        text=f"<b>{total}</b>",
        x=0.5, y=0.55,
        font=dict(size=22, color=C_TEXT, family=FONT),
        showarrow=False,
        xanchor="center",
    )
    fig.add_annotation(
        text="PAIRS",
        x=0.5, y=0.38,
        font=dict(size=10, color=C_MUTED, family=FONT),
        showarrow=False,
        xanchor="center",
    )

    fig.update_layout(
        **_base_layout(),
        height=200,
        width=200,
    )

    return fig


# ── 2. Sentiment per aspect — horizontal stacked bars ────────────────────────

def make_aspect_bars(aspect_data: list[dict], fig_height: int = None) -> go.Figure:
    categories = [d["category"].title() for d in aspect_data]
    pos_counts = [d["pos_count"] for d in aspect_data]
    neg_counts = [d["neg_count"] for d in aspect_data]
    totals     = [p + n for p, n in zip(pos_counts, neg_counts)]
    pos_pcts   = [
        round(p / t * 100) if t else 0
        for p, t in zip(pos_counts, totals)
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Positive",
        y=categories,
        x=pos_counts,
        orientation="h",
        marker=dict(color=C_POS, line=dict(width=0)),
        hovertemplate="<b>%{y} — Positive</b><br>%{x} pairs<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="Negative",
        y=categories,
        x=neg_counts,
        orientation="h",
        marker=dict(color=C_NEG, line=dict(width=0)),
        hovertemplate="<b>%{y} — Negative</b><br>%{x} pairs<extra></extra>",
    ))

    max_total = max(totals) if totals else 1
    for cat, pct, total in zip(categories, pos_pcts, totals):
        fig.add_annotation(
            x=total + max_total * 0.02,
            y=cat,
            text=f"{pct}% Positive",
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            font=dict(size=11, color=C_MUTED, family=FONT),
        )

    n_cats = len(categories)
    computed_h = fig_height if fig_height else max(380, n_cats * 80)
    fig.update_layout(
        **_base_layout(
            showlegend=False,
            margin=dict(l=110, r=100, t=20, b=20),
        ),
        barmode="stack",
        height=computed_h,
        autosize=True,
        xaxis=dict(
            visible=False,
            range=[0, max_total * 1.1] if totals else [0, 1],
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=13, color=C_TEXT, family=FONT),
            autorange="reversed",
            showticklabels=True,
            ticksuffix="  ",
        ),
        bargap=0.2,
    )

    return fig


# ── 3. Top negative keywords — horizontal bar chart ──────────────────────────

def make_keyword_bars(keyword_data: list[dict]) -> go.Figure:
    words  = [d["word"].upper() for d in keyword_data]
    counts = [d["count"] for d in keyword_data]

    fig = go.Figure(go.Bar(
        y=words,
        x=counts,
        orientation="h",
        marker=dict(color=C_NEG, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>%{x} mentions<extra></extra>",
        text=[f"{c} mentions" for c in counts],
        textposition="inside",
        textfont=dict(color="#FFFFFF", size=11, family=FONT),
        insidetextanchor="start",
    ))

    fig.update_layout(
        **_base_layout(
            margin=dict(l=100, r=40, t=10, b=10),
        ),
        height=max(200, len(words) * 52),
        autosize=True,
        xaxis=dict(visible=False),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color=C_TEXT, family=FONT),
            autorange="reversed",
            showticklabels=True,
        ),
        bargap=0.3,
    )

    return fig