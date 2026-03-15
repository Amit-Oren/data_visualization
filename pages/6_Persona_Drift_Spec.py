import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "conversations_GPT-GPT.jsonl")

st.set_page_config(page_title="Persona Drift", layout="wide")

st.title("Persona Drift")
st.markdown(
    "Does the **client LLM agent stay true to its assigned persona** throughout the conversation? "
    "Each turn is scored against the persona specification — not against the first message. "
    "A falling score means the agent is drifting away from who it was told to be."
)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_conversations():
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pf = r.get("persona_fields", {})
            records.append({
                "conversation_id":       r["conversation_id"],
                "domain":                pf.get("domain", "unknown"),
                "current_emotion":       pf.get("current_emotion", "unknown"),
                "assertiveness":         pf.get("assertiveness", "unknown"),
                "self_disclosure_level": pf.get("self_disclosure_level", "unknown"),
                "turns":                 r.get("turns", []),
            })
    return records

conversations = load_conversations()
conv_by_id    = {c["conversation_id"]: c for c in conversations}
all_domains   = sorted({c["domain"] for c in conversations})
domain_labels = {d: d.replace("_", " ").title() for d in all_domains}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")

selected_domain_label = st.sidebar.selectbox(
    "Filter by domain",
    ["All"] + [domain_labels[d] for d in all_domains],
)
selected_domain = None if selected_domain_label == "All" else next(
    d for d in all_domains if domain_labels[d] == selected_domain_label
)

filtered = conversations if selected_domain is None else [
    c for c in conversations if c["domain"] == selected_domain
]

conv_options = {
    f"Conv {c['conversation_id']} — {domain_labels[c['domain']]} ({c['current_emotion'].title()})": c["conversation_id"]
    for c in filtered
}

if not conv_options:
    st.warning("No conversations found for the selected domain.")
    st.stop()

selected_label = st.sidebar.selectbox("Conversation", list(conv_options.keys()))
conv_id        = conv_options[selected_label]
conv           = conv_by_id[conv_id]

# ── Persona spec ───────────────────────────────────────────────────────────────
EMOTION_POLARITY = {
    "anxious": -0.6, "sad": -0.7, "angry": -0.8, "frustrated": -0.6,
    "depressed": -0.8, "fearful": -0.6, "stressed": -0.5, "worried": -0.5,
    "upset": -0.6, "lonely": -0.6, "confused": -0.3, "nervous": -0.4,
    "neutral": 0.0, "calm": 0.1, "curious": 0.2, "content": 0.3,
    "happy": 0.7, "excited": 0.7, "hopeful": 0.5, "confident": 0.5,
    "grateful": 0.6, "optimistic": 0.5, "joyful": 0.8,
}

ASSERTIVENESS_SUBJECTIVITY = {
    "low":    (0.0, 0.35),
    "medium": (0.3, 0.65),
    "high":   (0.55, 1.0),
}

DISCLOSURE_FIRST_PERSON = {
    "low":    (0.0, 0.06),
    "medium": (0.04, 0.12),
    "high":   (0.10, 1.0),
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

# ── Signal computation ─────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

def wrap_text(text, width=80):
    words = text.split()
    lines, line = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) >= width:
            lines.append(" ".join(line))
            line = []
    if line:
        lines.append(" ".join(line))
    return "<br>".join(lines)

def score_emotion(text, target_emotion):
    target_pol = EMOTION_POLARITY.get(target_emotion.lower(), 0.0)
    actual_pol = TextBlob(text).sentiment.polarity
    distance = abs(actual_pol - target_pol)
    return max(0.0, 1.0 - distance / 2.0)

def score_assertiveness(text, level):
    lo, hi = ASSERTIVENESS_SUBJECTIVITY.get(level.lower(), (0.3, 0.65))
    subj = TextBlob(text).sentiment.subjectivity
    if lo <= subj <= hi:
        return 1.0
    elif subj < lo:
        return max(0.0, 1.0 - (lo - subj) / lo) if lo > 0 else 0.5
    else:
        return max(0.0, 1.0 - (subj - hi) / (1.0 - hi)) if hi < 1 else 0.5

def score_disclosure(text, level):
    words = tokenize(text)
    if not words:
        return 0.5
    fp_ratio = sum(1 for w in words if w in PERSONAL_WORDS) / len(words)
    lo, hi = DISCLOSURE_FIRST_PERSON.get(level.lower(), (0.04, 0.12))
    if lo <= fp_ratio <= hi:
        return 1.0
    elif fp_ratio < lo:
        return max(0.0, fp_ratio / lo) if lo > 0 else 0.5
    else:
        return max(0.0, 1.0 - (fp_ratio - hi) / hi) if hi > 0 else 0.5

def score_no_filler(text):
    tl = text.lower()
    hits = sum(1 for p in FILLER_PHRASES if p in tl)
    words = len(text.split())
    density = hits / max(words / 10, 1)
    return max(0.0, 1.0 - density)

def score_personal_vocab(text):
    words = tokenize(text)
    if len(words) < 3:
        return 0.5
    ttr = len(set(words)) / len(words)
    return min(ttr, 1.0)

def compute_adherence(turns, conv_meta):
    client_turns = [t for t in turns if t["speaker"] == "client"]
    if len(client_turns) < 2:
        return pd.DataFrame(), []

    emotion  = conv_meta["current_emotion"]
    assertiv = conv_meta["assertiveness"]
    disclos  = conv_meta["self_disclosure_level"]

    rows = []
    for i, ct in enumerate(client_turns):
        text = ct["text"]
        rows.append({
            "turn_seq":              i + 1,
            "Emotion Consistency":   score_emotion(text, emotion),
            "Assertiveness Match":   score_assertiveness(text, assertiv),
            "Self-Disclosure Match": score_disclosure(text, disclos),
            "No Filler Language":    score_no_filler(text),
            "Personal Vocabulary":   score_personal_vocab(text),
            "text":                  wrap_text(text),
        })

    df = pd.DataFrame(rows)

    SIGNALS = [
        "Emotion Consistency", "Assertiveness Match", "Self-Disclosure Match",
        "No Filler Language", "Personal Vocabulary",
    ]

    for s in SIGNALS:
        df[s + "_share"] = df[s] * 0.20

    df["adherence_score"] = df[[s + "_share" for s in SIGNALS]].sum(axis=1)
    df["trend"] = df["adherence_score"].rolling(window=3, min_periods=2, center=True).mean()
    return df, SIGNALS

df, SIGNALS = compute_adherence(conv["turns"], conv)

if df.empty or len(df) < 2:
    st.warning("Not enough client turns to analyze persona drift.")
    st.stop()

# ── Metrics ────────────────────────────────────────────────────────────────────
first_val = df["adherence_score"].iloc[0]
last_val  = df["adherence_score"].iloc[-1]
delta     = last_val - first_val

col1, col2, col3, col4 = st.columns(4)
col1.metric("Starting score", f"{first_val:.2f}")
col2.metric("Final score",    f"{last_val:.2f}", delta=f"{delta:+.2f}")
col3.metric("Total change",   f"{abs(delta):.2f}", delta=f"{delta:+.2f}", delta_color="inverse")
col4.metric("Client turns",   str(len(df)))

if delta < -0.10:
    verdict, vcolor = "Persona drift detected — the client agent drifted away from its assigned persona", "#e74c3c"
elif delta < -0.03:
    verdict, vcolor = "Mild drift — subtle divergence from the assigned character", "#e67e22"
else:
    verdict, vcolor = "Persona maintained — the client agent stayed true to its specification", "#27ae60"

st.markdown(
    f"<p style='font-size:1.05rem;font-weight:600;color:{vcolor};'>{verdict}</p>",
    unsafe_allow_html=True,
)

# ── Stacked area chart ─────────────────────────────────────────────────────────
SIGNAL_COLORS = {
    "Personal Vocabulary":   "#7DC0D8",  # sky blue
    "No Filler Language":    "#8ECBBE",  # mint
    "Self-Disclosure Match": "#A3BAD0",  # steel blue
    "Assertiveness Match":   "#74ABC6",  # ocean blue
    "Emotion Consistency":   "#A8D2EB",  # ice blue
}

SIGNAL_FILL = {
    "Personal Vocabulary":   "rgba(125, 192, 216, 0.55)",
    "No Filler Language":    "rgba(142, 203, 190, 0.55)",
    "Self-Disclosure Match": "rgba(163, 186, 208, 0.55)",
    "Assertiveness Match":   "rgba(116, 171, 198, 0.55)",
    "Emotion Consistency":   "rgba(168, 210, 235, 0.55)",
}

TREND_COLOR = "#2E86AB"

fig = go.Figure()

for signal in SIGNAL_COLORS:
    fig.add_trace(go.Scatter(
        x=df["turn_seq"],
        y=df[signal + "_share"],
        name=signal,
        stackgroup="one",
        mode="lines",
        line=dict(color=SIGNAL_COLORS[signal], width=1),
        fillcolor=SIGNAL_FILL[signal],
        customdata=df[signal].round(3),
        hovertemplate=f"<b>{signal}</b>: %{{customdata:.2f}}<extra></extra>",
    ))

trend_color = TREND_COLOR
fig.add_trace(go.Scatter(
    x=df["turn_seq"],
    y=df["trend"],
    mode="lines",
    name="Trend",
    line=dict(color=trend_color, width=2.5, dash="dot"),
    hovertemplate="<b>Trend — Turn %{x}</b>: %{y:.2f}<extra></extra>",
))

fig.add_annotation(x=df["turn_seq"].iloc[0],  y=first_val,
    text=f"<b>{first_val:.2f}</b>", showarrow=False, yshift=14,
    font=dict(size=12, color="#2c3e50"))
fig.add_annotation(x=df["turn_seq"].iloc[-1], y=last_val,
    text=f"<b>{last_val:.2f}</b>", showarrow=False, yshift=14,
    font=dict(size=12, color=trend_color))

fig.update_layout(
    xaxis_title="Client turn (sequential)",
    yaxis_title="Persona drift score  (each layer = one signal, max 0.20 each)",
    yaxis=dict(range=[0, 1.12], gridcolor="#E8E8E8", zeroline=False),
    xaxis=dict(showgrid=False, dtick=1),
    height=480,
    plot_bgcolor="white",
    legend=dict(font=dict(size=11), orientation="h",
                yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=70, b=40),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ── Explanation ────────────────────────────────────────────────────────────────
with st.expander("How to read this chart"):
    st.markdown(f"""
**Persona spec:** Emotion = *{conv['current_emotion'].title()}* · Assertiveness = *{conv['assertiveness'].title()}* · Self-disclosure = *{conv['self_disclosure_level'].title()}*

Each coloured band measures one dimension of adherence to the spec. When a band **shrinks**, the agent is breaking character on that dimension. Every score is computed against the spec directly — not relative to previous turns. Each signal contributes up to **0.20**, stacking to a maximum of **1.0**.

| Signal | What drifting looks like |
|--------|--------------------------|
| 🔷 **Emotion Consistency** | Tone no longer matches *"{conv['current_emotion']}"* (TextBlob polarity vs. expected) |
| 🔹 **Assertiveness Match** | Language too confident or too hesitant for *"{conv['assertiveness']}"* assertiveness (subjectivity score) |
| 🩵 **Self-Disclosure Match** | Sharing too little or too much for *"{conv['self_disclosure_level']}"* disclosure (first-person pronoun ratio) |
| 🟦 **No Filler Language** | Slipping into bot-speak — "absolutely", "great question", "happy to help" |
| 🔵 **Personal Vocabulary** | Language becoming repetitive and generic (low type-token ratio) |
""")

with st.expander("Per-turn data"):
    display = df[["turn_seq", "adherence_score"] + SIGNALS + ["text"]].copy()
    display.columns = ["Turn", "Adherence"] + SIGNALS + ["Client Text"]
    st.dataframe(display.round(3), use_container_width=True, hide_index=True)
