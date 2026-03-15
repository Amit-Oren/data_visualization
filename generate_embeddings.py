"""
Generate sentence embeddings, reduce with PCA→t-SNE, save CSV.
Pipeline: 384D embeddings → PCA 50D → t-SNE 2D
"""

import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

INPUT_JSONL = "data/conversations_GPT-GPT.jsonl"
OUTPUT_DIR  = "output_visualizations"
OUTPUT_CSV  = f"{OUTPUT_DIR}/conversation_embeddings_detailed.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading conversations...")
records = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
print(f"  {len(records)} conversations loaded")

texts = [" ".join(t["text"] for t in r.get("turns", [])) for r in records]

# ── Embed ─────────────────────────────────────────────────────────────────────
print("Encoding with all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
print(f"  Embeddings shape: {embeddings.shape}")

# ── PCA 50D ───────────────────────────────────────────────────────────────────
print("PCA -> 50D...")
n_components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
pca_50 = PCA(n_components=n_components, random_state=42)
embeddings_pca50 = pca_50.fit_transform(embeddings)
print(f"  Explained variance: {pca_50.explained_variance_ratio_.sum():.2%}")

# ── t-SNE 2D ─────────────────────────────────────────────────────────────────
print("t-SNE -> 2D (on PCA-50 for better quality)...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
coords = tsne.fit_transform(embeddings_pca50)
print("  t-SNE done")

# ── Build DataFrame ───────────────────────────────────────────────────────────
rows = []
for i, r in enumerate(records):
    pf = r.get("persona_fields", {})
    row = {
        "conversation_id":       r.get("conversation_id"),
        "persona_id":            r.get("persona_id"),
        "domain":                pf.get("domain", ""),
        "current_emotion":       pf.get("current_emotion", ""),
        "intensity":             pf.get("intensity", ""),
        "expressiveness":        pf.get("expressiveness", ""),
        "self_disclosure_level": pf.get("self_disclosure_level", ""),
        "politeness_style":      pf.get("politeness_style", ""),
        "assertiveness":         pf.get("assertiveness", ""),
        "gender":                pf.get("gender", ""),
        "occupation":            pf.get("occupation", ""),
        "age":                   pf.get("age", 0),
        "location":              pf.get("location", ""),
        "num_turns":             len(r.get("turns", [])),
        "termination_reason":    r.get("termination_reason", ""),
        "tsne_x":                float(coords[i, 0]),
        "tsne_y":                float(coords[i, 1]),
    }
    for j, v in enumerate(embeddings_pca50[i]):
        row[f"pca50_{j}"] = float(v)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} rows -> {OUTPUT_CSV}")
print(f"Columns: metadata + tsne_x/y + {embeddings_pca50.shape[1]} pca50_* dimensions")
