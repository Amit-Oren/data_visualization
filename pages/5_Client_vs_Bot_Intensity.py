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
st.title("Client vs Bot Intensity")
st.markdown(
    "Does the **therapist bot mirror, dampen, or amplify** the client's emotional intensity? "
    "Each violin shows the distribution of TextBlob intensity for each speaker "
    "across Mild / Moderate / Severe persona groups."
)

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


st.caption("Left half = Client · Right half = Bot · Centre line = mean")

st.divider()

with st.expander("What is this visualization and how do I read it?"):
    st.markdown("""
**The core question:** Does the therapist bot mirror, dampen, or amplify the emotional intensity of the client — and does that pattern change depending on how severe the client's emotional state is?

---

#### What
Each shape is a **violin plot** — a smoothed view of how intensity scores are distributed across all turns for that speaker and severity group.
The wider the violin, the more turns cluster at that intensity level.
The centre line marks the **mean** intensity.

| Column | Meaning |
|--------|---------|
| **Mild** | Personas with calm, content, or hopeful emotions |
| **Moderate** | Personas with anxious, stressed, or frustrated emotions |
| **Severe** | Personas with sad, depressed, or angry emotions |

---

#### How to read it
- **Left (green) half** = Client turns · **Right (blue) half** = Bot turns
- A **higher mean line** = that speaker uses more emotionally intense language on average
- **Wider belly** = intensity is spread across a broad range; **narrow shape** = most turns cluster tightly around one value
- If the bot's mean is consistently *lower* than the client's, the bot is dampening intensity (a calming effect)
- If they track closely, the bot is mirroring the client's emotional level

---

#### What to look for
- Does the gap between Client and Bot means grow as severity increases?
- Is the bot's distribution narrower than the client's — suggesting more controlled, consistent language?
- Which severity group shows the biggest mismatch between the two speakers?
""")
