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
st.title("t-SNE Conversation Map")

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
color_by_label = st.sidebar.selectbox(
    "Color scatter by",
    ["Cluster", "Domain"],
)
color_by = {"Cluster": "cluster", "Domain": "domain"}[color_by_label]


# ── Run K-Means ───────────────────────────────────────────────────────────────
@st.cache_data
def get_results():
    return run_kmeans_on_embeddings(load_data(), n_clusters=OPTIMAL_K)

with st.spinner("Running K-Means clustering..."):
    results = get_results()

df["cluster"] = results["labels"].astype(str)


# ── t-SNE Scatter ─────────────────────────────────────────────────────────────
st.subheader("t-SNE Scatter Plot — Conversations in 2D Space")

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
    # "#A9AEF3", "#E37865", "#63D7B8", "#B68BE3", "#F7A160",
    # "#4464D7", "#FD97B4", "#B8EA7C", "#FF97CB", "#FECB52",
    # "#76B7F0", "#4C854F", "#75EE8F", "#ED8DF2", "#F77745"

"#9DA3EE", "#DC6E58", "#52C9A8", "#A87ADA", "#F4964E",
"#3A5AC8", "#FC88AA", "#AADF6C", "#FC88BE", "#FEC038",
"#62A8EC", "#4A7A4C", "#6AE488", "#E88AEE", "#F46838"
]
all_domains_sorted = sorted(df["domain"].dropna().unique())
domain_color = {d: PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i, d in enumerate(all_domains_sorted)}

if color_by == "cluster":
    fig_scatter = go.Figure()
    cluster_ids = sorted(df["cluster"].unique(), key=int)
    cluster_ids = sorted(cluster_ids, key=lambda cid: df[df["cluster"] == cid]["domain"].value_counts().index[0])
    seen_domains = set()
    for cluster_id in cluster_ids:
        cdata = df[df["cluster"] == cluster_id]
        top_domain = cdata["domain"].value_counts().index[0]
        first_occurrence = top_domain not in seen_domains
        seen_domains.add(top_domain)
        fig_scatter.add_trace(go.Scatter(
            x=cdata["tsne_x"], y=cdata["tsne_y"],
            mode="markers",
            name=top_domain.replace("_", " ").title(),
            legendgroup=top_domain,
            showlegend=first_occurrence,
            marker=dict(
                color=domain_color.get(top_domain, "#aaaaaa"),
                size=9, opacity=0.75, line=dict(color="white", width=0.6),
            ),
            text=cdata["hover_text"],
            hovertemplate="%{text}<extra></extra>",
        ))

elif color_by == "domain":
    fig_scatter = go.Figure()
    for domain in all_domains_sorted:
        cdata = df[df["domain"] == domain]
        fig_scatter.add_trace(go.Scatter(
            x=cdata["tsne_x"], y=cdata["tsne_y"],
            mode="markers",
            name=domain.replace("_", " ").title(),
            marker=dict(
                color=domain_color.get(domain, "#aaaaaa"),
                size=9, opacity=0.75, line=dict(color="white", width=0.6),
            ),
            text=cdata["hover_text"],
            hovertemplate="%{text}<extra></extra>",
        ))


fig_scatter.update_layout(
    height=600,
    plot_bgcolor="#FAFAFA",
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

st.divider()



with st.expander("What is this visualization and how do I read it?", expanded=True):
    st.markdown("""
**The core question:** Do conversations with similar topics and styles naturally group together — even when we don't tell the model what the topic is?

---

#### What was done
Each conversation was converted into a single numeric vector using a language embedding model — a representation that captures the *meaning* of the whole conversation, not just keywords. K-Means then grouped these vectors into **40 clusters** based on similarity in that meaning space.

To make 40-dimensional clusters visible to the human eye, **t-SNE** was applied — a technique that compresses high-dimensional distances into a 2D map while preserving neighborhood structure. **Points that are close together had similar conversations.**

---

#### How to read it

| Mode | What the colors show |
|------|---------------------|
| **Cluster** | Each color = one K-Means group. Points of the same color were clustered together by the algorithm. The legend label shows the most common domain in that cluster. |
| **Domain** | Each color = one topic area (e.g. anxiety, relationships). Use this to see whether domain boundaries match cluster boundaries. |

- **Tight, well-separated blobs** → the model found clean, meaningful groups
- **Mixed or overlapping colors** → those conversation types are semantically similar — hard even for the algorithm to separate
- **Hover over any point** to see the conversation's domain, cluster, occupation, age, and other persona fields

---

#### What to look for
Switch between **Cluster** and **Domain** views. If the color patterns look similar, it means the algorithm recovered domain structure *purely from conversation content* — without being told what domain each conversation belonged to. That's the signal this chart is trying to surface.
""")