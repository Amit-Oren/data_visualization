import streamlit as st

st.set_page_config(
    page_title="Introduction",
    page_icon="💬",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; font-size:2.8rem; margin-bottom:0;'>
        Visualizing Persona-Driven Multi-Agent Conversations
    </h1>
    <p style='text-align:center; color:gray; font-size:1.1rem; margin-top:0.5rem; margin-bottom:0;'>
        Data Visualization Course — Final Project
    </p>
    <p style='text-align:center; color:#555; font-size:1rem; margin-top:0.2rem;'>
        Eden Cohen &nbsp;&nbsp;·&nbsp;&nbsp; Amit Oren
    </p>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── About ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.subheader("Data Source")
    st.markdown(
        """
        Generated using the **SPASM** multi-agent simulation framework —
        **S**table **P**ersona driven **A**gent **S**imulation for **M**ulti-turn dialogue generation.

        Persona-driven, multi-turn conversations between LLM agents, where one agent assumes
        the role of a **client** seeking advice and the other acts as a **responder**.
        The conversations span diverse domains and simulated human personas,
        capturing both topical and emotional variation.

        Each persona is characterised by attributes such as:

        | Attribute | Examples |
        |---|---|
        | **Domain** | health, finance, education, technology… |
        | **Emotion** | happy, anxious, frustrated, curious… |
        | **Occupation** | engineer, teacher, student, nurse… |
        | **Demographics** | age, gender |
        | **Style** | assertiveness, expressiveness, intensity |
        """
    )

with col_right:
    st.subheader("Data Description")
    st.markdown(
        """
        **Data Acquisition**

        The dataset was provided in raw JSON format, consisting of multi-turn
        conversations between agents.
        """
    )

    st.subheader("Course Context")
    st.markdown(
        """
        This dashboard was built as the final project for a **Data Visualization course**.
        It presents **6 interactive visualizations**, each exploring a different
        dimension of the conversation dataset — from clustering and demographic profiles
        to emotional dynamics and persona drift.
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
st.markdown("This app contains **6 interactive visualizations** for the course:")

v1, v2, v3 = st.columns(3)
v4, v5, v6 = st.columns(3)

with v1:
    st.markdown("**1. Demographics Overview**")
    st.caption(
        "Breaks down the dataset by age, gender, and occupation, "
        "giving a demographic profile of the personas in the dataset."
    )

with v2:
    st.markdown("**2. t-SNE Conversation Map**")
    st.caption(
        "Reduces high-dimensional conversation embeddings to 2-D with t-SNE. "
        "Points close together had similar conversations — coloured by domain or cluster."
    )

with v3:
    st.markdown("**3. Client vs Bot Intensity**")
    st.caption(
        "Compares the emotional intensity of the client messages vs. "
        "the bot responses throughout each conversation."
    )

with v4:
    st.markdown("**4. Emotional Journey**")
    st.caption(
        "Visualizes how the emotional tone of a conversation evolves turn by turn, "
        "capturing shifts in sentiment across the dialogue."
    )

with v5:
    st.markdown("**5. Persona Drift**")
    st.caption(
        "Tracks how consistently the client agent maintains its assigned persona "
        "across a multi-turn conversation, scored across five behavioral signals."
    )

with v6:
    st.markdown("**6. Persona Drift Density**")
    st.caption(
        "Ridge plot showing the distribution of persona drift scores across conversations, "
        "grouped by emotion — revealing whether drift is the norm or the exception."
    )

st.divider()
st.caption("Data Visualization Course — Final Project · GPT-GPT Conversation Dataset")
