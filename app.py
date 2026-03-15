import streamlit as st

st.set_page_config(
    page_title="Conversation Analytics",
    page_icon="💬",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; font-size:3rem; margin-bottom:0;'>
        💬 Conversation Analytics
    </h1>
    <p style='text-align:center; color:gray; font-size:1.1rem; margin-top:0.3rem;'>
        Data Visualization Course — Final Project
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── About ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("About the Dataset")
    st.markdown(
        """
        This project explores a synthetic dataset of **GPT-to-GPT conversations**,
        where each conversation is driven by a unique persona profile.
        The dataset captures how language models with different personalities,
        emotions, and backgrounds communicate with one another.

        Each persona is characterised by attributes such as:

        | Attribute | Examples |
        |---|---|
        | **Domain** | health, finance, education, technology… |
        | **Emotion** | happy, anxious, frustrated, curious… |
        | **Occupation** | engineer, teacher, student, nurse… |
        | **Demographics** | age, gender |
        | **Style** | assertiveness, expressiveness, intensity |

        Conversations were embedded using a language model and then analysed
        to discover hidden structure in how different personas communicate.
        """
    )

with col_right:
    st.subheader("Course Context")
    st.markdown(
        """
        This dashboard was built as the final project for a **Data Visualization course**.
        It presents **7 interactive visualizations**, each exploring a different
        dimension of the conversation dataset — from clustering and word analysis
        to emotional dynamics and persona drift.

        Use each visualization to answer questions like:
        - How do conversations cluster by persona?
        - Which words define each domain?
        - How do emotions shift across a conversation?
        - Do client and bot differ in emotional intensity?
        """
    )

    st.info(
        "Use the **sidebar** on each page to navigate and adjust settings.",
        icon="👈",
    )

st.divider()

# ── Quick stats ───────────────────────────────────────────────────────────────
import os, json

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "conversations_GPT-GPT.jsonl")

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    num_conversations = len(records)
    domains = {r.get("persona_fields", {}).get("domain") for r in records if r.get("persona_fields", {}).get("domain")}
    emotions = {r.get("persona_fields", {}).get("current_emotion") for r in records if r.get("persona_fields", {}).get("current_emotion")}
    avg_turns = sum(len(r.get("turns", [])) for r in records) / max(num_conversations, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Conversations", f"{num_conversations:,}")
    m2.metric("Unique Domains", len(domains))
    m3.metric("Unique Emotions", len(emotions))
    m4.metric("Avg. Turns / Conv.", f"{avg_turns:.1f}")
else:
    st.warning("Dataset file not found at `data/conversations_GPT-GPT.jsonl`.")

st.divider()

# ── Visualizations overview ───────────────────────────────────────────────────
st.subheader("Visualizations")
st.markdown("This app contains **7 interactive visualizations** for the course:")

v1, v2, v3 = st.columns(3)
v4, v5, v6 = st.columns(3)
v7, _, _   = st.columns(3)

with v1:
    st.markdown("**1. Demographics Overview**")
    st.caption(
        "Breaks down the dataset by age, gender, and occupation, "
        "giving a demographic profile of the personas in the dataset."
    )

with v2:
    st.markdown("**2. Top Words per Domain**")
    st.caption(
        "Identifies the most distinctive vocabulary used in each domain (health, finance, education, etc.) "
        "using TF-IDF-style scoring."
    )

with v3:
    st.markdown("**3. t-SNE Conversation Map**")
    st.caption(
        "Reduces high-dimensional conversation embeddings to 2-D with t-SNE. "
        "Points close together had similar conversations — coloured by domain or cluster."
    )

with v4:
    st.markdown("**4. Domain Semantic Similarity**")
    st.caption(
        "Uses conversation embeddings to compute how semantically similar "
        "different domains are to each other, displayed as a heatmap."
    )

with v5:
    st.markdown("**5. Client vs Bot Intensity**")
    st.caption(
        "Compares the emotional intensity of the client (user) messages vs. "
        "the bot (assistant) responses throughout each conversation."
    )

with v6:
    st.markdown("**6. Persona Drift**")
    st.caption(
        "Tracks how a persona's style and tone shift over the course of a conversation, "
        "revealing drift in assertiveness, expressiveness, and other style attributes."
    )

with v7:
    st.markdown("**7. Client vs Bot — Vocabulary Gap**")
    st.caption(
        "Diverging bar chart: left side shows words distinctive to the client, "
        "right side shows words distinctive to the bot — immediately revealing the vocabulary gap between them."
    )

st.divider()
st.caption("Data Visualization Course — Final Project · GPT-GPT Conversation Dataset")
