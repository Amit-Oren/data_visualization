import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "conversations_GPT-GPT.jsonl")

st.set_page_config(page_title="Demographics Overview", layout="wide")
st.title("Demographics Overview")
st.markdown("**What**")
st.markdown(
    "An interactive overview of the **demographic and contextual composition** of the synthetic persona dataset. "
    "The visualizations summarize how key attributes — gender, age, geographic region, occupation category, and conversation domain — "
    "are distributed across all generated personas. Together, these charts provide a high-level profile of **who is represented in the dataset** "
    "before analyzing the conversations themselves."
)

st.markdown("""
**Why**
- To understand the overall structure and balance of the dataset before interpreting conversational patterns
- To identify whether certain demographic groups, occupations, emotional profiles, or domains are overrepresented or underrepresented
- To provide context for downstream analyses by showing the population of personas from which the conversations were generated

**How**
- Persona metadata was extracted from the JSONL conversation file, including demographic attributes and assigned conversation context fields
- These attributes were grouped and aggregated into interpretable categories — occupation families, geographic regions, and domain families — and their frequencies were counted across all personas
- The resulting distributions were visualized using bar charts and an age histogram with density smoothing to reveal both categorical balance and continuous age patterns in the dataset
""")

# ── Pastel Palette ────────────────────────────────────────────────────────────
BLUE   = "#A8C8E8"   # pastel blue (age histogram)
MUTED  = "#B8D4E0"
POS    = "#96D4A8"   # pastel green (positive)
NEG    = "#F4A7A7"   # pastel rose  (negative)
DOMAIN = "#A8C8E8"

POSITIVE_EMOTIONS = {"calm", "content", "happy", "hopeful"}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_demographics():
    rows = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r  = json.loads(line)
            pf = r.get("persona_fields", {})
            rows.append({
                "gender":     pf.get("gender", "unknown"),
                "age":        pf.get("age"),
                "location":   pf.get("location", "unknown"),
                "occupation": pf.get("occupation", "unknown"),
                "emotion":    pf.get("current_emotion", "unknown"),
                "domain":     pf.get("domain", "unknown"),
            })
    return pd.DataFrame(rows)

df = load_demographics()

# ── Helper: bar label annotation list ─────────────────────────────────────────
def bar_annotations(x_vals, y_vals, horizontal=False):
    """Return a list of Plotly annotation dicts with count labels."""
    anns = []
    for x, y in zip(x_vals, y_vals):
        if horizontal:
            anns.append(dict(
                x=y + max(y_vals) * 0.01,
                y=x,
                text=str(int(y)),
                showarrow=False,
                xanchor="left",
                font=dict(size=11),
            ))
        else:
            anns.append(dict(
                x=x,
                y=y + max(y_vals) * 0.02,
                text=str(int(y)),
                showarrow=False,
                yanchor="bottom",
                font=dict(size=11),
            ))
    return anns

LAYOUT_BASE = dict(
    plot_bgcolor="#FAFAFA",
    paper_bgcolor="#FAFAFA",
    margin=dict(l=10, r=10, t=40, b=40),
    font=dict(family="sans-serif", size=12),
)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Gender | Age
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

# ── Gender bar ────────────────────────────────────────────────────────────────
with col1:
    st.subheader("Gender Distribution")
    g_order  = ["male", "female", "non-binary", "prefer_not_to_say"]
    g_labels = ["Male", "Female", "Non-binary", "Prefer not to say"]
    g_counts = [int(df["gender"].value_counts().get(g, 0)) for g in g_order]

    fig_g = go.Figure(go.Bar(
        x=g_labels,
        y=g_counts,
        marker_color="#A8D8B8",
        marker_line_width=0,
    ))
    fig_g.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Count"),
        annotations=bar_annotations(g_labels, g_counts),
        height=380,
    )
    st.plotly_chart(fig_g, use_container_width=True)

# ── Age histogram + KDE ───────────────────────────────────────────────────────
with col2:
    st.subheader("Age Distribution")
    ages = df["age"].dropna().astype(float).values

    bin_size  = 10
    bin_edges = np.arange(0, 101, bin_size)
    counts, edges = np.histogram(ages, bins=bin_edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    kde_x = np.linspace(0, 100, 400)
    kde_y = gaussian_kde(ages, bw_method=0.4)(kde_x)   # raw density

    fig_a = go.Figure()
    bin_labels = [f"{int(edges[i])}–{int(edges[i+1])}" for i in range(len(counts))]
    fig_a.add_trace(go.Bar(
        x=centres,
        y=counts,
        width=bin_size * 0.9,
        marker_color=BLUE,
        marker_line_color="white",
        marker_line_width=1,
        opacity=0.85,
        name="Count",
        yaxis="y1",
        customdata=bin_labels,
        hovertemplate="Age %{customdata}: %{y}<extra></extra>",
    ))
    fig_a.add_trace(go.Scatter(
        x=kde_x,
        y=kde_y,
        mode="lines",
        line=dict(color="#C06080", width=2.5),
        name="Density",
        yaxis="y2",
    ))
    fig_a.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(showgrid=False, title="Age (years)", range=[0, 100], dtick=10),
        yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Number of Participants", range=[0, 120]),
        yaxis2=dict(
            title=dict(text="Density", font=dict(color="#C06080")),
            tickfont=dict(color="#C06080"),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        showlegend=False,
        height=420,
    )
    st.plotly_chart(fig_a, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Continent / Ethnicity proxy
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

CITY_CONTINENT = {
    "Brisbane": "Oceania",        "Sydney": "Oceania",
    "Melbourne": "Oceania",       "Auckland": "Oceania",
    "Boston": "North America",    "San Francisco": "North America",
    "Washington D.C.": "North America", "Toronto": "North America",
    "Nairobi": "Africa",          "Lagos": "Africa",
    "Johannesburg": "Africa",     "Cairo": "Africa",
    "Taipei": "East Asia",        "Seoul": "East Asia",
    "Busan": "East Asia",         "Beijing": "East Asia",
    "Shanghai": "East Asia",      "Hong Kong": "East Asia",
    "Osaka": "East Asia",
    "Singapore": "Southeast Asia","Jakarta": "Southeast Asia",
    "Manila": "Southeast Asia",   "Bangkok": "Southeast Asia",
    "Delhi": "South Asia",
    "Stockholm": "Europe",        "Rome": "Europe",
    "Barcelona": "Europe",        "Zurich": "Europe",
    "Amsterdam": "Europe",        "Madrid": "Europe",
    "Istanbul": "Europe",
    "Riyadh": "Middle East",      "Dubai": "Middle East",
    "Abu Dhabi": "Middle East",
}
CONTINENT_COLORS = {
    "Oceania":        "#96D4C8",
    "North America":  "#F7D4A0",
    "Africa":         "#C8B8E0",
    "East Asia":      "#A8C8E8",
    "Southeast Asia": "#B8D8EC",
    "South Asia":     "#FBA8A8",
    "Europe":         "#F4C2CE",
    "Middle East":    "#F4D4A8",
}

df["continent"] = df["location"].map(CITY_CONTINENT).fillna("Other")
cont        = df["continent"].value_counts().sort_values(ascending=False)
cont_labels = cont.index.tolist()
cont_counts = cont.values.tolist()

fig_cont = go.Figure(go.Bar(
    x=cont_labels,
    y=cont_counts,
    marker_color="#C4B8E8",
    marker_line_width=0,
))
fig_cont.update_layout(
    **LAYOUT_BASE,
    xaxis=dict(showgrid=False, title=""),
    yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Number of Participants"),
    annotations=bar_annotations(cont_labels, cont_counts),
    height=420,
)
st.subheader("Geographic Distribution")
st.plotly_chart(fig_cont, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Occupation (stacked by category)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

OCCUPATION_CATEGORY = {
    # Technology
    "software_engineer":        "Technology",
    "ux_designer":              "Technology",
    "it_support_specialist":    "Technology",
    "network_administrator":    "Technology",
    # Engineering
    "civil_engineer":           "Engineering",
    "electrical_engineer":      "Engineering",
    "mechanical_engineer":      "Engineering",
    "construction_worker":      "Engineering",
    # Healthcare
    "doctor":                   "Healthcare",
    "nurse":                    "Healthcare",
    "dentist":                  "Healthcare",
    "physiotherapist":          "Healthcare",
    "social_worker":            "Healthcare",
    # Education
    "university_professor":     "Education",
    "education_administrator":  "Education",
    "librarian":                "Education",
    # Science & Research
    "biologist":                "Science & Research",
    "environmental_analyst":    "Science & Research",
    # Business & Consulting
    "business_consultant":      "Business",
    "sales_representative":     "Business",
    # Legal
    "judge":                    "Legal",
    # Creative & Media
    "graphic_designer":         "Creative & Media",
    "journalist":               "Creative & Media",
    "writer":                   "Creative & Media",
    "photographer":             "Creative & Media",
    "filmmaker":                "Creative & Media",
    # Trades & Services
    "electrician":              "Trades & Services",
    "chef":                     "Trades & Services",
    "barista":                  "Trades & Services",
    "bartender":                "Trades & Services",
    "retail_worker":            "Trades & Services",
    "driver":                   "Trades & Services",
    "caregiver":                "Trades & Services",
    "tour_guide":               "Trades & Services",
    "fisherman":                "Trades & Services",
    "machine_operator":         "Trades & Services",
    "warehouse_worker":         "Trades & Services",
    "quality_assurance_inspector": "Trades & Services",
}

OCCUPATION_CATEGORY_COLORS = {
    "Technology":        "#A8C8E8",
    "Engineering":       "#B8D4EC",
    "Healthcare":        "#96D4C8",
    "Education":         "#C8B8E0",
    "Science & Research":"#D4C8EC",
    "Business":          "#F7D4A0",
    "Legal":             "#B8E0B8",
    "Creative & Media":  "#F4C2CE",
    "Trades & Services": "#C8CBD4",
    "Other":             "#DCDFE4",
}

OCCUPATION_CATEGORY_ORDER = [
    "Technology", "Engineering", "Healthcare", "Education",
    "Science & Research", "Business", "Legal", "Creative & Media", "Trades & Services", "Other",
]

st.subheader("Occupation Distribution by Category")
df["occ_category"] = df["occupation"].map(OCCUPATION_CATEGORY).fillna("Other")
occ_cat_totals = df["occ_category"].value_counts()
occ_cats_ordered = [c for c in OCCUPATION_CATEGORY_ORDER if c in occ_cat_totals.index]
occ_cat_counts   = [int(occ_cat_totals.get(c, 0)) for c in occ_cats_ordered]
occ_max = max(occ_cat_counts) if occ_cat_counts else 1

fig_o = go.Figure(go.Bar(
    x=occ_cats_ordered,
    y=occ_cat_counts,
    marker_color="#F8DC94",
    marker_line_color="white",
    marker_line_width=1,
    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
))
fig_o.update_layout(
    **LAYOUT_BASE,
    xaxis=dict(showgrid=False, categoryorder="array", categoryarray=OCCUPATION_CATEGORY_ORDER),
    yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Count", range=[0, occ_max * 1.15]),
    annotations=bar_annotations(occ_cats_ordered, occ_cat_counts),
    showlegend=False,
    height=420,
)
st.plotly_chart(fig_o, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Domain
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

DOMAIN_CATEGORY_COLORS = {
    # Mental Health — pastel purple shades
    "anxiety_disorder_support":                 "#C8B8E0",
    "depression_support":                       "#BCA8D8",
    "social_anxiety_support":                   "#D8CCE8",
    "motivation_loss_support":                  "#C4B4DC",
    "self_esteem_guidance":                     "#E0D4EC",
    "impostor_syndrome_support":                "#D3C9EA",
    "mental_vs_physical_symptom_clarification": "#CCB8E4",
    "emotional_dependency_discussion":          "#C0ACDC",
    "identity_crisis_support":                  "#D7C2ED",
    "grief_and_loss_support":                   "#B0A0CC",
    # Relationships — pastel rose shades
    "boundary_setting_guidance":                "#F4C2CE",
    "breakup_recovery_support":                 "#EEB4C0",
    "dating_support":                           "#F8D0D8",
    "friendship_dynamics":                      "#FADCE0",
    "long_distance_relationship_support":       "#E8ACBC",
    # Career & Work — pastel blue shades
    "career_counseling":                        "#A8C8E8",
    "job_search_support":                       "#8AB3E2",
    "salary_negotiation":                       "#B8D4EC",
    "promotion_negotiation":                    "#A0C4E8",
    "workplace_conflict_resolution":            "#C8DCF0",
    "workplace_burnout_support":                "#90B8E0",
    "workplace_rights":                         "#B0CCE8",
    # Legal & Finance — pastel green shades
    "consumer_complaint_resolution":            "#B8E0B8",
    "contract_interpretation":                  "#A8D8A8",
    "financial_planning":                       "#9EE09E",
    "insurance_policy_explanation":             "#BEE4BE",
    "small_claims_guidance":                    "#A0D0A0",
    "identity_theft_response":                  "#D0ECD0",
    # Health — pastel teal shades
    "fitness_and_physical_health":              "#96D4C8",
    "medical_consultation":                     "#88CCC0",
    "reproductive_health":                      "#86CBC1",
    # Life & Society — pastel peach shades
    "cultural_adjustment":                      "#FAC16B",
    "data_sharing_decision":                    "#FFE8B7",
    "home_maintenance_advice":                  "#FAE0B4",
    "immigration_consultation":                 "#F0C888",
    "life_goal_setting":                        "#FCE8C4",
    "online_harassment_support":                "#ECC07C",
    "travel_planning_support":                  "#F8D8A8",
    "discrimination_experience_discussion":     "#E8B870",
    # Practical Help — warm beige shades
    "lost_and_found_guidance":                  "#E8D4B8",
    "second_opinion_guidance":                  "#DEC8A8",
    "technical_support":                        "#F0DEC8",
}

# ── Domain stacked bar (grouped by category) ──────────────────────────────────
DOMAIN_TO_CATEGORY = {
    "anxiety_disorder_support":                "Mental Health",
    "depression_support":                      "Mental Health",
    "social_anxiety_support":                  "Mental Health",
    "motivation_loss_support":                 "Mental Health",
    "self_esteem_guidance":                    "Mental Health",
    "impostor_syndrome_support":               "Mental Health",
    "mental_vs_physical_symptom_clarification":"Mental Health",
    "emotional_dependency_discussion":         "Mental Health",
    "identity_crisis_support":                 "Mental Health",
    "grief_and_loss_support":                  "Mental Health",
    "boundary_setting_guidance":               "Relationships",
    "breakup_recovery_support":                "Relationships",
    "dating_support":                          "Relationships",
    "friendship_dynamics":                     "Relationships",
    "long_distance_relationship_support":      "Relationships",
    "career_counseling":                       "Career & Work",
    "job_search_support":                      "Career & Work",
    "salary_negotiation":                      "Career & Work",
    "promotion_negotiation":                   "Career & Work",
    "workplace_conflict_resolution":           "Career & Work",
    "workplace_burnout_support":               "Career & Work",
    "workplace_rights":                        "Career & Work",
    "consumer_complaint_resolution":           "Legal & Finance",
    "contract_interpretation":                 "Legal & Finance",
    "financial_planning":                      "Legal & Finance",
    "insurance_policy_explanation":            "Legal & Finance",
    "small_claims_guidance":                   "Legal & Finance",
    "identity_theft_response":                 "Legal & Finance",
    "fitness_and_physical_health":             "Health",
    "medical_consultation":                    "Health",
    "reproductive_health":                     "Health",
    "cultural_adjustment":                     "Life & Society",
    "data_sharing_decision":                   "Life & Society",
    "home_maintenance_advice":                 "Life & Society",
    "immigration_consultation":                "Life & Society",
    "life_goal_setting":                       "Life & Society",
    "online_harassment_support":               "Life & Society",
    "travel_planning_support":                 "Life & Society",
    "discrimination_experience_discussion":    "Life & Society",
    "lost_and_found_guidance":                 "Practical Help",
    "second_opinion_guidance":                 "Practical Help",
    "technical_support":                       "Practical Help",
}

CATEGORY_ORDER = [
    "Mental Health", "Relationships", "Career & Work",
    "Legal & Finance", "Health", "Life & Society", "Practical Help",
]

with st.container():
    st.subheader("Domain Distribution by Category")
    df["dom_category"] = df["domain"].map(DOMAIN_TO_CATEGORY).fillna("Other")
    dom_cat_totals    = df["dom_category"].value_counts()
    dom_cats_ordered  = [c for c in CATEGORY_ORDER if c in dom_cat_totals.index]
    dom_cat_counts    = [int(dom_cat_totals.get(c, 0)) for c in dom_cats_ordered]
    dom_max = max(dom_cat_counts) if dom_cat_counts else 1

    CATEGORY_COLORS = {
        "Mental Health":  "#C8B8E0",
        "Relationships":  "#F4C2CE",
        "Career & Work":  "#A8C8E8",
        "Legal & Finance":"#B8E0B8",
        "Health":         "#96D4C8",
        "Life & Society": "#F7D4A0",
        "Practical Help": "#E8D4B8",
    }

    fig_d = go.Figure(go.Bar(
        x=dom_cats_ordered,
        y=dom_cat_counts,
        marker_color="#A8DDD4",
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))
    fig_d.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(showgrid=False, categoryorder="array", categoryarray=CATEGORY_ORDER),
        yaxis=dict(showgrid=True, gridcolor="#ECECEC", title="Count", range=[0, dom_max * 1.15]),
        annotations=bar_annotations(dom_cats_ordered, dom_cat_counts),
        showlegend=False,
        height=460,
    )
    st.plotly_chart(fig_d, use_container_width=True)
