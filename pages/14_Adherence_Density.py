import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from textblob import TextBlob

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "conversations_GPT-GPT.jsonl")

st.set_page_config(page_title="Adherence Density", layout="wide")
st.title("How Is Persona Adherence Distributed Across Conversations?")

st.markdown("**What**")
st.markdown(
    "Kernel density estimate (KDE) curves showing the full distribution of persona adherence scores "
    "across conversations, overlaid by domain or emotion group. "
    "Each curve is a smoothed probability density: tall and narrow means scores cluster tightly; "
    "flat and wide means they spread across the range."
)

st.markdown("""
**Why**
- To reveal whether most conversations achieve high fidelity or whether drift is the norm
- To detect bimodal distributions where some conversations score very high and others very low — a sign of inconsistency within a group
- To compare the shape of distributions across domains or emotions, not just their averages

**How**
- Per-conversation adherence scores were computed by averaging the five drift signals across all client turns
- A Gaussian KDE was fitted to each group using Scott's bandwidth rule, then normalised so every ridge reaches the same peak height
- Groups are stacked vertically (ridge plot), sorted by median adherence — the highest-scoring group sits at the top
- A dotted vertical tick marks each group's median; individual scores are shown as a rug on each baseline
""")

# ── Signal constants ───────────────────────────────────────────────────────────
EMOTION_POLARITY = {
    "anxious": -0.6, "sad": -0.7, "angry": -0.8, "frustrated": -0.6,
    "depressed": -0.8, "fearful": -0.6, "stressed": -0.5, "worried": -0.5,
    "upset": -0.6, "lonely": -0.6, "confused": -0.3, "nervous": -0.4,
    "neutral": 0.0, "calm": 0.1, "curious": 0.2, "content": 0.3,
    "happy": 0.7, "excited": 0.7, "hopeful": 0.5, "confident": 0.5,
    "grateful": 0.6, "optimistic": 0.5, "joyful": 0.8,
    "overwhelmed": -0.6,
}
ASSERTIVENESS_SUBJECTIVITY = {
    "low": (0.0, 0.35), "medium": (0.3, 0.65), "high": (0.55, 1.0),
}
DISCLOSURE_FIRST_PERSON = {
    "low": (0.0, 0.06), "medium": (0.04, 0.12), "high": (0.10, 1.0),
}
FILLER_PHRASES = [
    "i understand", "i see", "that's great", "absolutely", "certainly",
    "of course", "i'm here to help", "feel free", "let me know",
    "happy to help", "great question", "no problem", "sounds good",
    "sure thing", "definitely", "you're welcome",
    "i appreciate", "thank you for sharing",
]
PERSONAL_WORDS = {
    "i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll",
    "we", "our", "ours", "ourselves",
}

def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

def score_emotion(text, emotion):
    target = EMOTION_POLARITY.get(emotion.lower(), 0.0)
    return max(0.0, 1.0 - abs(TextBlob(text).sentiment.polarity - target) / 2.0)

def score_assertiveness(text, level):
    lo, hi = ASSERTIVENESS_SUBJECTIVITY.get(level.lower(), (0.3, 0.65))
    subj = TextBlob(text).sentiment.subjectivity
    if lo <= subj <= hi:
        return 1.0
    return max(0.0, 1.0 - (lo - subj) / lo) if subj < lo and lo > 0 else \
           max(0.0, 1.0 - (subj - hi) / (1.0 - hi)) if hi < 1 else 0.5

def score_disclosure(text, level):
    words = tokenize(text)
    if not words:
        return 0.5
    fp = sum(1 for w in words if w in PERSONAL_WORDS) / len(words)
    lo, hi = DISCLOSURE_FIRST_PERSON.get(level.lower(), (0.04, 0.12))
    if lo <= fp <= hi:
        return 1.0
    return max(0.0, fp / lo) if fp < lo and lo > 0 else \
           max(0.0, 1.0 - (fp - hi) / hi) if hi > 0 else 0.5

def score_no_filler(text):
    hits = sum(1 for p in FILLER_PHRASES if p in text.lower())
    return max(0.0, 1.0 - hits / max(len(text.split()) / 10, 1))

def score_vocab(text):
    words = tokenize(text)
    return min(len(set(words)) / len(words), 1.0) if len(words) >= 3 else 0.5

# ── Load & score ───────────────────────────────────────────────────────────────
@st.cache_data
def build_scores():
    rows = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pf = r["persona_fields"]
            emotion  = pf.get("current_emotion", "neutral")
            assertiv = pf.get("assertiveness", "medium")
            disclos  = pf.get("self_disclosure_level", "medium")
            domain   = pf.get("domain", "unknown")

            client_turns = [t for t in r.get("turns", []) if t["speaker"] == "client"]
            if not client_turns:
                continue

            turn_scores = []
            for ct in client_turns:
                txt = ct["text"]
                turn_scores.append([
                    score_emotion(txt, emotion),
                    score_assertiveness(txt, assertiv),
                    score_disclosure(txt, disclos),
                    score_no_filler(txt),
                    score_vocab(txt),
                ])

            means = np.mean(turn_scores, axis=0)
            rows.append({
                "conversation_id": r["conversation_id"],
                "domain":   domain,
                "emotion":  emotion,
                "adherence": float(np.mean(means)),
                "num_turns": len(r.get("turns", [])),
            })
    return pd.DataFrame(rows)

df = build_scores()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")

group_by = st.sidebar.radio("Group by", ["Emotion", "Domain"], horizontal=True)
group_col = "emotion" if group_by == "Emotion" else "domain"

all_groups = sorted(df[group_col].unique())
group_labels = {g: g.replace("_", " ").title() for g in all_groups}

if group_by == "Emotion":
    default = all_groups  # emotions are few enough to show all
else:
    default = all_groups[:6]

selected_groups = st.sidebar.multiselect(
    f"Select {group_by.lower()}s",
    options=all_groups,
    default=default,
    format_func=lambda g: group_labels[g],
)

fill_alpha = 0.40
overlap    = 0.8

if not selected_groups:
    st.warning("Select at least one group.")
    st.stop()

# ── Colour map ─────────────────────────────────────────────────────────────────
PALETTE = [
    "#2E86AB", "#E07B39", "#6BAE6B", "#C25656", "#A97CC4",
    "#C9A227", "#56A0A0", "#D4678A", "#7A9E56", "#8C7057",
    "#4A90D9", "#E86B3A", "#5BA65B", "#B84040", "#9B6AC0",
]

def hex_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

color_map = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(all_groups)}

# ── Sort groups by mean (descending → top of chart) ───────────────────────────
valid_groups = [g for g in selected_groups
                if len(df[df[group_col] == g]["adherence"].values) >= 3]

means = {g: float(np.mean(df[df[group_col] == g]["adherence"].values))
         for g in valid_groups}
# bottom-to-top order: lowest mean at index 0, highest at end
sorted_groups = sorted(valid_groups, key=lambda g: means[g])

# ── Compute all KDEs first ─────────────────────────────────────────────────────
x_grid = np.linspace(0, 1, 400)
kdes = {}
for g in sorted_groups:
    vals = df[df[group_col] == g]["adherence"].values
    kde = gaussian_kde(vals)
    y = kde(x_grid)
    kdes[g] = y / y.max()          # normalise peak to 1

row_height = 1.0                    # each row occupies 1 unit; overlap compresses spacing
spacing    = row_height * (1.8 - overlap)
spacing    = max(spacing, 0.3)

# ── Build ridge plot ───────────────────────────────────────────────────────────
fig = go.Figure()

label_x_pos = -0.04   # annotation x in data coords (slightly left of 0)

for i, group in enumerate(sorted_groups):
    baseline  = i * spacing
    vals      = df[df[group_col] == group]["adherence"].values
    color     = color_map[group]
    label     = group_labels[group]
    y_kde     = kdes[group] * row_height * 0.9   # scale ridge height
    med       = means[group]

    # ── Invisible baseline for fill anchor ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=np.full_like(x_grid, baseline),
        mode="lines",
        line=dict(color=hex_rgba(color, 0.0), width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Filled KDE ridge ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=y_kde + baseline,
        mode="lines",
        fill="tonexty",
        fillcolor=hex_rgba(color, fill_alpha),
        line=dict(color=color, width=1.8),
        name=label,
        hovertemplate=f"<b>{label}</b><br>Score: %{{x:.3f}}<extra></extra>",
    ))

    # ── Median tick (vertical segment at median) ───────────────────────────────
    med_idx = int(np.argmin(np.abs(x_grid - med)))
    med_kde_height = float(kdes[group][med_idx]) * row_height * 0.9
    fig.add_trace(go.Scatter(
        x=[med, med],
        y=[baseline, baseline + med_kde_height],
        mode="lines",
        line=dict(color=color, width=1.5, dash="dot"),
        showlegend=False,
        hovertemplate=f"<b>Mean</b>: {med:.3f}<extra></extra>",
    ))


    # ── Group label annotation (left of chart) ────────────────────────────────
    fig.add_annotation(
        x=-0.015, y=baseline,
        text=f"<b>{label}</b>  {med:.2f}",
        xref="x", yref="y",
        showarrow=False,
        xanchor="right",
        font=dict(size=10, color=color),
    )

n = len(sorted_groups)
y_max = (n - 1) * spacing + row_height * 1.1

fig.update_layout(
    xaxis=dict(
        title="Persona adherence score",
        range=[-0.02, 1.02],
        showgrid=True,
        gridcolor="#ECECEC",
        zeroline=False,
        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    ),
    yaxis=dict(
        visible=False,
        range=[-0.1, y_max],
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=max(380, n * 38 + 80),
    margin=dict(l=200, r=30, t=40, b=50),
    showlegend=False,
    hovermode="closest",
)

st.plotly_chart(fig, use_container_width=True)

# ── Summary stats ──────────────────────────────────────────────────────────────
with st.expander("Summary statistics"):
    rows = []
    for group in selected_groups:
        vals = df[df[group_col] == group]["adherence"].values
        rows.append({
            group_by: group_labels[group],
            "N": len(vals),
            "Mean":   round(float(np.mean(vals)), 3),
            "Median": round(float(np.median(vals)), 3),
            "Std":    round(float(np.std(vals)), 3),
            "Min":    round(float(np.min(vals)), 3),
            "Max":    round(float(np.max(vals)), 3),
        })
    st.dataframe(pd.DataFrame(rows).sort_values("Mean", ascending=False),
                 use_container_width=True, hide_index=True)
