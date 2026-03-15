import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "output_visualizations", "conversation_embeddings_detailed.csv"
)

st.set_page_config(page_title="Domain Semantic Similarity", layout="wide")
st.title("Domain Semantic Similarity")
st.markdown(
    "Pairwise **cosine similarity between domain centroids** computed from 50-dim PCA embeddings. "
    "Which support domains share the same conversational DNA?"
)

if not os.path.exists(EMBEDDINGS_CSV):
    st.error("Embeddings CSV not found. Run `generate_embeddings.py` first.")
    st.stop()

# ── Load & compute ────────────────────────────────────────────────────────────
@st.cache_data
def compute_similarity():
    df = pd.read_csv(EMBEDDINGS_CSV)
    pca_cols = [c for c in df.columns if c.startswith("pca50_")]
    centroids = df.groupby("domain")[pca_cols].mean()

    sim = cosine_similarity(centroids.values)
    sim_df = pd.DataFrame(sim, index=centroids.index, columns=centroids.index)
    return sim_df, centroids.index.tolist()

sim_df, raw_domain_order = compute_similarity()

# ── Curated subset ────────────────────────────────────────────────────────────
CURATED_DOMAINS = [
    "anxiety_disorder_support",
    "depression_support",
    "social_anxiety_support",
    "impostor_syndrome_support",
    "emotional_dependency_discussion",
    "medical_consultation",
    "mental_vs_physical_symptom_clarification",
    "salary_negotiation",
    "promotion_negotiation",
    "second_opinion_guidance",
    "workplace_burnout_support",
    "identity_crisis_support",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
show_values  = st.sidebar.checkbox("Show similarity values in cells", value=False)
curated_only = st.sidebar.checkbox("Show curated subset only", value=True)

# Always alphabetical
all_domains = sorted(sim_df.index.tolist())
if curated_only:
    subset = sorted([d for d in CURATED_DOMAINS if d in sim_df.index])
else:
    subset = all_domains
sim_display = sim_df.loc[subset, subset]

domains = sim_display.index.tolist()
labels  = [d.replace("_", " ").title() for d in domains]
z       = sim_display.values.copy()

# ── Heatmap ───────────────────────────────────────────────────────────────────
cell_text = None
if show_values:
    cell_text = [[f"{v:.2f}" for v in row] for row in z]

fig = go.Figure(go.Heatmap(
    z=z,
    x=labels,
    y=labels,
    text=cell_text,
    texttemplate="%{text}" if show_values else None,
    textfont=dict(size=7),
    colorscale=[
        [0.0,  "#2166ac"],
        [0.35, "#d1e5f0"],
        [0.5,  "#f7f7f7"],
        [0.65, "#fddbc7"],
        [1.0,  "#b2182b"],
    ],
    zmid=0,
    colorbar=dict(
        title="Cosine<br>similarity",
        tickformat=".2f",
        len=0.7,
    ),
    xgap=1, ygap=1,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "<b>%{x}</b><br>"
        "Similarity: %{z:.4f}<extra></extra>"
    ),
))

n = len(domains)
fig.update_layout(
    xaxis=dict(showgrid=False, tickfont=dict(size=11), tickangle=-60, side="bottom"),
    yaxis=dict(showgrid=False, tickfont=dict(size=11), autorange="reversed"),
    height=max(700, n * 17 + 150),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=30, b=160),
    font=dict(family="sans-serif", size=11),
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Diagonal is 1.0 (self-similarity). Domains sorted alphabetically.")

# ── Domain spotlight ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Domain spotlight")
st.caption("Pick a domain to see its top 10 nearest neighbours.")

selected = st.selectbox(
    "Domain",
    options=sorted([d.replace("_", " ").title() for d in raw_domain_order]),
)
sel_key = selected.lower().replace(" ", "_")

if sel_key in sim_df.index:
    neighbours = (
        sim_df[sel_key]
        .drop(sel_key)
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    neighbours.columns = ["Domain", "Similarity"]
    neighbours["Domain"] = neighbours["Domain"].str.replace("_", " ").str.title()

    fig2 = go.Figure(go.Bar(
        x=neighbours["Similarity"],
        y=neighbours["Domain"],
        orientation="h",
        marker_color=[
            f"rgba(178,24,43,{0.4 + 0.55 * (s - neighbours['Similarity'].min()) / max(neighbours['Similarity'].max() - neighbours['Similarity'].min(), 1e-9)})"
            for s in neighbours["Similarity"]
        ],
        marker_line_width=0,
        text=neighbours["Similarity"].round(4),
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig2.update_layout(
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FAFAFA",
        xaxis=dict(title="Cosine similarity", showgrid=True, gridcolor="#ECECEC", range=[0, 1.05]),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=360,
        margin=dict(l=10, r=80, t=30, b=40),
        font=dict(family="sans-serif", size=12),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)
