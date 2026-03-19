# Visualizing Persona-Driven Multi-Agent Conversations

A Streamlit dashboard built as the final project for a **Data Visualization course**.

**Eden Cohen · Amit Oren**

## Data Source

Generated using the **SPASM** multi-agent simulation framework —
**S**table **P**ersona driven **A**gent **S**imulation for **M**ulti-turn dialogue generation.

Persona-driven, multi-turn conversations between LLM agents, where one agent assumes the role of a **client** seeking advice and the other acts as a **responder**. The conversations span diverse domains and simulated human personas, capturing both topical and emotional variation.

The dataset was provided in raw JSON format, consisting of multi-turn conversations between agents.

## Persona Attributes

| Attribute | Examples |
|---|---|
| **Domain** | health, finance, education, technology |
| **Emotion** | happy, anxious, frustrated, curious |
| **Assertiveness** | low, medium, high |
| **Self-disclosure level** | low, medium, high |
| **Occupation** | engineer, teacher, student, nurse |
| **Demographics** | age, gender |
| **Style** | expressiveness, intensity |

## Visualizations

| # | Page | What it shows |
|---|------|---------------|
| 1 | **Demographics Overview** | Age, gender, and occupation breakdown of the personas |
| 2 | **t-SNE Conversation Map** | Conversation embeddings reduced to 2D — similar conversations cluster together |
| 3 | **Client vs Bot Intensity** | Comparing emotional intensity of client vs. bot messages |
| 4 | **Emotional Journey** | Turn-by-turn sentiment across a conversation |
| 5 | **Persona Drift** | Stacked area chart tracking how well the client agent maintains its assigned persona over time |
| 6 | **Persona Drift Density** | Ridge plot of persona drift score distributions grouped by emotion |

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
├── app.py                        # Home / introduction page
├── pages/
│   ├── 1_Demographics_Overview.py
│   ├── 2_tSNE_Clustering.py
│   ├── 3_Client_vs_Bot_Intensity.py
│   ├── 4_Emotional_Journey.py
│   ├── 5_Persona_Drift_Spec.py
│   └── 6_Persona_Drift_Density.py
├── data/
│   └── conversations_GPT-GPT.jsonl
├── utils/
└── requirements.txt
```
