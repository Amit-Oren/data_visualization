import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


from utils.metrics import run_kmeans_on_embeddings

EMBEDDINGS_CSV = os.path.join(os.path.dirname(__file__), "..", "output_visualizations", "conversation_embeddings_detailed.csv")
OPTIMAL_K = 40

st.set_page_config(page_title="t-SNE Clustering", layout="wide")
st.title("How Conversations Naturally Group in Semantic Space")

st.markdown("**What**")
st.markdown(
    "An interactive 2D visualization of multi-turn LLM conversations. "
    "Each point represents a single conversation. "
    "Colors indicate clusters of semantically similar conversations."
)

st.markdown("""
**Why**
- To explore patterns and structure in large-scale conversational data
- To examine how domains, emotions, and personas naturally group together
- To assess whether conversations align with their intended contexts

**How**
- Conversations were encoded into semantic embeddings using a pre-trained language model, then reduced to 50 dimensions via PCA
- K-Means clustering (k=40) was applied to group semantically similar conversations in the 50-dimensional embedding space
- t-SNE was used to project these high-dimensional embeddings into a 2D space for visualization — points that are close together represent conversations with similar meaning and style
- Cluster quality was evaluated using the Silhouette Score, which measures how well each conversation fits its assigned cluster
""")

# ── Load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(EMBEDDINGS_CSV):
    st.error("Embeddings CSV not found. Please run `generate_embeddings.py` first.")
    st.code("python generate_embeddings.py")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(EMBEDDINGS_CSV)

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")


# ── Run K-Means ───────────────────────────────────────────────────────────────
@st.cache_data
def get_results():
    return run_kmeans_on_embeddings(load_data(), n_clusters=OPTIMAL_K)

with st.spinner("Running K-Means clustering..."):
    results = get_results()

df["cluster"] = results["labels"].astype(str)

# ── Cluster quality metrics ────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Clusters (k)", OPTIMAL_K)
c2.metric("Silhouette Score", f"{results['silhouette']:.3f}")
c3.metric("Conversations", len(df))

# ── t-SNE Scatter ─────────────────────────────────────────────────────────────
df["hover_text"] = df.apply(
    lambda row: (
        f"<b>Conversation {row['conversation_id']}</b><br>"
        f"Cluster: {row['cluster']}<br>"
        f"Domain: {str(row['domain']).replace('_', ' ').title()}<br>"
        f"Emotion: {str(row['current_emotion']).title()}<br>"
        f"Occupation: {str(row['occupation']).replace('_', ' ').title()}<br>"
        f"Age: {row['age']}<br>"
        f"Intensity: {row['intensity']}<br>"
        f"Expressiveness: {row['expressiveness']}"
    ), axis=1
)



# ── Shared domain color map (used by both Cluster and Domain views) ────────────
PLOTLY_COLORS = [
    "#FF6B7A", "#FF8C42", "#F5D000", "#3EC98A", "#7B8FE8",
    "#E8607A", "#5AAAD8", "#8EC840", "#D8A040", "#8060D8",
    "#FF8040", "#6EC858", "#40A8D8", "#D860A8", "#40D898",
    "#FFB020", "#3AAED8", "#FE9040", "#50C8A0", "#A060D0",
    "#FF70B0", "#40C8F8", "#FFB060", "#90E040", "#F060C8",
    "#40D8FF", "#FFB860", "#60E880", "#D060FF", "#FF6060",
    "#50B8FE", "#E8F020", "#70E870", "#FF80FF", "#FF6848",
    "#00D8FF", "#80C870", "#E06030", "#C89870", "#60A8C8",
]
cluster_ids = sorted(df["cluster"].unique(), key=int)
cluster_color = {cid: PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i, cid in enumerate(cluster_ids)}

fig_scatter = go.Figure()
for cluster_id in cluster_ids:
    cdata = df[df["cluster"] == cluster_id]
    fig_scatter.add_trace(go.Scatter(
        x=cdata["tsne_x"], y=cdata["tsne_y"],
        mode="markers",
        name=f"Cluster {cluster_id}",
        marker=dict(
            color=cluster_color[cluster_id],
            size=9, opacity=0.75, line=dict(color="white", width=0.6),
        ),
        text=cdata["hover_text"],
        hovertemplate="%{text}<extra></extra>",
    ))


fig_scatter.update_layout(
    height=600,
    plot_bgcolor="white",
    xaxis=dict(
        title="t-SNE Dimension 1",
        showticklabels=True,
        showgrid=True,
        gridcolor="#ECECEC",
        zeroline=False,
    ),
    yaxis=dict(
        title="t-SNE Dimension 2",
        showticklabels=True,
        showgrid=True,
        gridcolor="#ECECEC",
        zeroline=False,
    ),
    legend=dict(font=dict(size=9), itemsizing="constant"),
)

st.plotly_chart(fig_scatter, width='stretch')


