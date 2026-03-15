import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "conversations_GPT-GPT.jsonl")

st.title("Emotional Journey")
st.markdown("How does the client's sentiment shift turn by turn — from the first message to the last?")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_conversations():
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pf = r.get("persona_fields", {})
            records.append({
                "conversation_id": r["conversation_id"],
                "domain":          pf.get("domain", "unknown"),
                "current_emotion": pf.get("current_emotion", "unknown"),
                "turns":           r.get("turns", []),
            })
    return records

conversations = load_conversations()
conv_by_id    = {c["conversation_id"]: c for c in conversations}
all_domains   = sorted({c["domain"] for c in conversations})
domain_labels = {d: d.replace("_", " ").title() for d in all_domains}

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

# ── Score client turns only ────────────────────────────────────────────────────
def wrap_text(text, width=80):
    words, lines, line = text.split(), [], []
    for word in words:
        if sum(len(w) + 1 for w in line) + len(word) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "<br>".join(lines)

def score_turns(turns):
    rows = []
    for t in turns:
        if t.get("speaker") != "client":
            continue
        polarity = TextBlob(t["text"]).sentiment.polarity
        rows.append({
            "turn":     t["turn_index"],
            "polarity": polarity,
            "text":     t["text"],
            "wrapped":  wrap_text(t["text"]),
        })
    return pd.DataFrame(rows)

df_turns = score_turns(conv["turns"])

if df_turns.empty:
    st.warning("No client turns found.")
    st.stop()

# ── Verdict ───────────────────────────────────────────────────────────────────
first_score = df_turns["polarity"].iloc[0]
last_score  = df_turns["polarity"].iloc[-1]
delta       = last_score - first_score

if delta > 0.05:
    verdict, verdict_color, arrow = "Sentiment improved", "#2ecc71", "↗"
elif delta < -0.05:
    verdict, verdict_color, arrow = "Sentiment declined", "#e74c3c", "↘"
else:
    verdict, verdict_color, arrow = "Sentiment stayed neutral", "#95a5a6", "→"

col1, col2, col3 = st.columns(3)
col1.metric("First turn polarity", f"{first_score:+.3f}")
col2.metric("Last turn polarity",  f"{last_score:+.3f}")
col3.metric("Net change", f"{delta:+.3f}", delta_color="normal")

st.markdown(
    f"<p style='font-size:1.1rem; font-weight:600; color:{verdict_color};'>{verdict} {arrow}</p>",
    unsafe_allow_html=True,
)

# ── Line chart ────────────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_hline(y=0, line_dash="dot", line_color="#cccccc", line_width=1)
fig.add_hrect(y0=0,  y1=1,  fillcolor="#2ecc71", opacity=0.12, line_width=0)
fig.add_hrect(y0=-1, y1=0,  fillcolor="#e74c3c", opacity=0.12, line_width=0)

fig.add_trace(go.Scatter(
    x=df_turns["turn"],
    y=df_turns["polarity"],
    mode="lines+markers",
    name="Client",
    line=dict(color="#3498db", width=2.5),
    marker=dict(size=8, color="#3498db", line=dict(color="white", width=1.5)),
    customdata=df_turns["wrapped"],
    hovertemplate="<b>Turn %{x}</b> — Polarity: %{y:.3f}<br><br>%{customdata}<extra></extra>",
))

fig.update_layout(
    xaxis_title="Client turn",
    yaxis_title="Sentiment polarity",
    yaxis=dict(range=[-1.05, 1.05], zeroline=False, gridcolor="#ECECEC"),
    xaxis=dict(showgrid=False),
    height=450,
    plot_bgcolor="#FAFAFA",
    margin=dict(l=10, r=10, t=30, b=40),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

with st.expander("What is this visualization and how do I read it?"):
    st.markdown(f"""
**The core question:** Does the conversation actually help — does the client end up in a more positive emotional state than when they started?

---

#### What
Each point on the line is one of the client's messages, scored for **sentiment polarity** using TextBlob — a value between **−1** (very negative) and **+1** (very positive). The line connects those scores across the conversation, tracing the client's emotional arc from start to finish.

---

#### Why
Raw conversation logs don't tell you whether a bot session was effective. Sentiment polarity gives a proxy signal: if the line trends upward, the client's language became more positive. If it drops or stays flat, the bot may not have moved the needle emotionally.

---

#### How to read it
- **Above the dotted line (0)** → positive, constructive language
- **Below the dotted line** → negative, distressed language
- **Rising arc** → emotional improvement across the session
- **Flat or falling arc** → little or no emotional shift
- **Hover any point** to read the first 120 characters of that client message
- The **net change** metric above the chart (+/−) summarises the full arc in one number
""")

