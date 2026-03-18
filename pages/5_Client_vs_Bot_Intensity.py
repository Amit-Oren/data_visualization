import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.graph_objects as go

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "conversations_GPT-GPT.jsonl")

st.set_page_config(page_title="Client vs Bot Intensity", layout="wide")
st.title("Does the Bot Mirror the Client's Emotional Intensity?")

st.markdown("**What**")
st.markdown(
    "A comparison of the emotional intensity expressed by the client and the therapist bot "
    "across conversations with different assigned severity levels. "
    "Each violin shows the distribution of message-level intensity scores for one speaker. "
    "The split shape allows direct comparison between client and bot behavior within **Mild**, **Moderate**, and **Severe** persona groups."
)

st.markdown("""
**Why**
- To examine whether the therapist bot responds with a similar, lower, or stronger emotional intensity than the client
- To explore whether the bot mirrors the client's emotional severity appropriately across different emotional contexts
- To assess whether response behavior remains sensitive to the intended emotional profile of the conversation

**How**
- Each conversation was assigned to a severity group based on the client's predefined emotion label, mapped into Mild, Moderate, or Severe categories
- For every client and bot message, an intensity score was computed using TextBlob sentiment features, combining polarity strength and subjectivity into a single emotional intensity measure
- These scores were aggregated by speaker and severity group, then visualized as split violin plots to compare the full distribution and average intensity of client and bot messages
""")

# ── Emotion → severity ─────────────────────────────────────────────────────────
EMOTION_SEVERITY = {
    "calm": "Mild", "content": "Mild", "curious": "Mild", "hopeful": "Mild",
    "confident": "Mild", "grateful": "Mild", "optimistic": "Mild",
    "joyful": "Mild", "happy": "Mild",
    "anxious": "Moderate", "stressed": "Moderate", "worried": "Moderate",
    "nervous": "Moderate", "lonely": "Moderate", "confused": "Moderate",
    "frustrated": "Moderate", "excited": "Moderate",
    "sad": "Severe", "depressed": "Severe", "angry": "Severe",
    "fearful": "Severe", "upset": "Severe",
}

SEVERITY_ORDER = ["Mild", "Moderate", "Severe"]

CLIENT_COLOR = "#A8D8A8"   # pastel green
CLIENT_DARK  = "#4A9E5A"
BOT_COLOR    = "#A8C8E8"   # pastel blue
BOT_DARK     = "#3A7AB0"

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r  = json.loads(line)
            pf = r.get("persona_fields", {})
            records.append({
                "conversation_id": r["conversation_id"],
                "domain":          pf.get("domain", "unknown"),
                "current_emotion": pf.get("current_emotion", "unknown"),
                "turns":           r.get("turns", []),
            })
    return records

conversations = load_data()

# ── Build data ─────────────────────────────────────────────────────────────────
def intensity(text):
    blob = TextBlob(text)
    return round((abs(blob.sentiment.polarity) * 0.6 + blob.sentiment.subjectivity * 0.4) * 10, 3)

@st.cache_data
def build_data(conv_list):
    rows = []
    for conv in conv_list:
        severity = EMOTION_SEVERITY.get(conv["current_emotion"].lower(), "Moderate")
        for t in conv["turns"]:
            if t["speaker"] not in ("client", "tested"):
                continue
            rows.append({
                "severity": severity,
                "speaker":  "Client" if t["speaker"] == "client" else "Bot",
                "intensity": intensity(t["text"]),
            })
    return pd.DataFrame(rows)

df = build_data(tuple(conversations))

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ── Split violin chart ─────────────────────────────────────────────────────────
fig = go.Figure()

for severity in SEVERITY_ORDER:
    sub_c = df[(df["severity"] == severity) & (df["speaker"] == "Client")]["intensity"].tolist()
    sub_b = df[(df["severity"] == severity) & (df["speaker"] == "Bot")]["intensity"].tolist()

    fig.add_trace(go.Violin(
        x=[severity] * len(sub_c),
        y=sub_c,
        name="Client",
        side="negative",
        fillcolor=CLIENT_COLOR,
        line_color=CLIENT_DARK,
        line_width=1.2,
        opacity=0.85,
        box_visible=False,
        meanline_visible=True,
        meanline=dict(color=CLIENT_DARK, width=2),
        points=False,
        spanmode="hard",
        legendgroup="Client",
        showlegend=(severity == "Mild"),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Violin(
        x=[severity] * len(sub_b),
        y=sub_b,
        name="Bot",
        side="positive",
        fillcolor=BOT_COLOR,
        line_color=BOT_DARK,
        line_width=1.2,
        opacity=0.85,
        box_visible=False,
        meanline_visible=True,
        meanline=dict(color=BOT_DARK, width=2),
        points=False,
        spanmode="hard",
        legendgroup="Bot",
        showlegend=(severity == "Mild"),
        hoverinfo="skip",
    ))

fig.update_layout(
    violinmode="overlay",
    violingap=0.3,
    xaxis=dict(title="Assigned Severity Level", showgrid=False,
               categoryorder="array", categoryarray=SEVERITY_ORDER,
               range=[-0.6, 2.6]),
    yaxis=dict(title="Intensity", gridcolor="#ECECEC", rangemode="nonnegative", range=[0, None]),
    plot_bgcolor="#FAFAFA",
    paper_bgcolor="#FAFAFA",
    height=500,
    margin=dict(l=40, r=40, t=30, b=40),
    font=dict(size=12),
    legend=dict(font=dict(size=12), orientation="h", yanchor="bottom", y=1.01, x=0),
)

st.plotly_chart(fig, use_container_width=True)
