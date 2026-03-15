import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

EXTRA_STOPWORDS = {
    "yeah", "ok", "okay", "actually", "really", "maybe", "perhaps", "quite",
    "still", "already", "though", "since", "sounds", "bit", "lot", "things",
    "thing", "people", "sure", "right", "try", "come", "back", "look",
    "want", "need", "feel", "feels", "feeling", "great", "going",
    "think", "know", "get", "go", "one", "would", "could", "make", "time",
    "even", "well", "see", "say", "said", "got", "way", "something", "much",
    "many", "might", "may", "always", "every", "also", "just",  
}
ALL_STOPWORDS = tuple(sorted(ENGLISH_STOP_WORDS | EXTRA_STOPWORDS))

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "conversations_GPT-GPT.jsonl")

st.title("Most Distinctive Words per Domain")

# ── Load & build one text blob per domain ─────────────────────────────────────
@st.cache_data
def load_tfidf(stopwords: tuple):
    domain_texts: dict[str, str] = {}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            domain = record.get("persona_fields", {}).get("domain", "unknown")
            text   = " ".join(t.get("text", "") for t in record.get("turns", []))
            domain_texts[domain] = domain_texts.get(domain, "") + " " + text

    domains  = sorted(domain_texts.keys())
    corpus   = [domain_texts[d] for d in domains]

    vectorizer = TfidfVectorizer(stop_words=list(stopwords), token_pattern=r"[a-z]{3,}")
    matrix     = vectorizer.fit_transform(corpus)
    vocab      = vectorizer.get_feature_names_out()

    # top-15 words per domain by TF-IDF score
    top_words: dict[str, list[tuple[str, float]]] = {}
    for i, domain in enumerate(domains):
        row    = matrix[i].toarray()[0]
        top15  = sorted(zip(vocab, row), key=lambda x: x[1], reverse=True)[:15]
        top_words[domain] = top15

    return domains, top_words

all_domains, domain_top_words = load_tfidf(ALL_STOPWORDS)
domain_labels = {d: d.replace("_", " ").title() for d in all_domains}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
selected_label  = st.sidebar.selectbox("Domain", [domain_labels[d] for d in all_domains])
selected_domain = next(d for d in all_domains if domain_labels[d] == selected_label)

top15  = domain_top_words[selected_domain]
words  = [w for w, _ in reversed(top15)]
scores = [s for _, s in reversed(top15)]

# ── Bar chart ──────────────────────────────────────────────────────────────────
fig = go.Figure(go.Bar(
    x=scores,
    y=words,
    orientation="h",
    marker=dict(color=scores, colorscale="Teal", showscale=False),
    hovertemplate="<b>%{y}</b>: TF-IDF %{x:.4f}<extra></extra>",
))

fig.update_layout(
    title=f"Top 15 distinctive words — {selected_label}",
    xaxis_title="TF-IDF score",
    yaxis_title="",
    height=550,
    plot_bgcolor="#FAFAFA",
    xaxis=dict(showgrid=True, gridcolor="#ECECEC"),
    yaxis=dict(showgrid=False),
    margin=dict(l=10, r=20, t=50, b=40),
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

with st.expander("What is this visualization and how do I read it?"):
    st.markdown(f"""
**The core question:** What words make *{selected_label}* conversations unique — words you wouldn't find at the same rate in any other domain?

---

#### What
Each bar represents a word found in **{selected_label}** conversations. The length of the bar is its **TF-IDF score** — not how often the word appears, but how *exclusively* it appears in this domain compared to all others.

---

#### Why TF-IDF instead of word frequency?
Raw word counts show the most *common* words — but common words like *feel*, *help*, or *know* appear in every domain equally and tell you nothing distinctive.

TF-IDF solves this by penalizing words that appear across many domains. A word scores high only when it is:
- **frequent** in this domain (TF — term frequency), AND
- **rare** in other domains (IDF — inverse document frequency)

The result: every domain gets a unique fingerprint of words that actually characterize it.

---

#### How to read it
- **Longer bar** = more exclusive to this domain
- **Switch domains** in the sidebar — notice how the words change completely
- Words at the top are the clearest signals of what makes this domain's conversations distinct from the other {len(all_domains) - 1} domains in the dataset
""")


