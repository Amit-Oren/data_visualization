# Conversation Analytics Dashboard

A Streamlit dashboard built as the final project for a **Data Visualization course**. It explores a synthetic dataset of GPT-to-GPT conversations, where each conversation is driven by a unique persona profile.

## Dataset

The dataset (`data/conversations_GPT-GPT.jsonl`) contains synthetic conversations between two language model agents. Each conversation is associated with a persona profile defined by:

| Attribute | Examples |
|---|---|
| **Domain** | health, finance, education, technology |
| **Emotion** | happy, anxious, frustrated, curious |
| **Assertiveness** | low, medium, high |
| **Self-disclosure level** | low, medium, high |
| **Occupation** | engineer, teacher, student, nurse |

## Visualizations

| # | Page | What it shows |
|---|------|---------------|
| 1 | **Demographics Overview** | Age, gender, and occupation breakdown of the personas |
| 2 | **Top Words per Domain** | Most distinctive vocabulary per domain using TF-IDF scoring |
| 3 | **t-SNE Conversation Map** | Conversation embeddings reduced to 2D — similar conversations cluster together |
| 4 | **Domain Semantic Similarity** | Heatmap of how semantically similar domains are to each other |
| 5 | **Client vs Bot Intensity** | Split violin comparing emotional intensity of client vs. bot across severity groups |
| 6 | **Persona Drift** | Stacked area chart tracking how well the client agent adheres to its assigned persona over time |
| 7 | **Emotional Journey** | Turn-by-turn sentiment polarity across a conversation |

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd data_visualization

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
data_visualization/
├── app.py                        # Home page
├── pages/
│   ├── 1_Demographics_Overview.py
│   ├── 2_Top_Words_per_Domain.py
│   ├── 3_tSNE_Clustering.py
│   ├── 4_Embedding_Similarity.py
│   ├── 5_Client_vs_Bot_Intensity.py
│   ├── 6_Persona_Drift_Spec.py
│   └── 7_Emotional_Journey.py
├── data/
│   └── conversations_GPT-GPT.jsonl
├── utils/
└── requirements.txt
```
