# This was written with the help of Claude AI, but substantial written text is mine.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fiction Classification Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data once and cache
@st.cache_data
def load_data():
    """
    Loads and preprocesses the novel dataset.
    This function ensures data is loaded only once and shared across all pages.
    """
    df = pd.read_csv('data/final_dataset.csv')
    df['decade'] = (df['publication_year'] // 10) * 10
    
    # Consolidate country names
    uk_variants = ['United Kingdom', 'England', 'United Kingdom of Great Britain and Ireland',
                   'Kingdom of Great Britain', 'Great Britain']
    df['country_consolidated'] = df['country_grouped'].replace(uk_variants, 'United Kingdom')
    
    us_variants = ['United States', 'United States of America']
    df['country_consolidated'] = df['country_consolidated'].replace(us_variants, 'United States')
    
    # Set chronological ordering for literary periods
    period_order = ['classical', 'romantic', 'victorian', 'modernist', 'postwar', 'contemporary', 'modern', 'unknown']
    df['literary_period'] = pd.Categorical(df['literary_period'], categories=period_order, ordered=True)
    
    return df

# Initialize Session State for Data (available to all pages)
if 'data' not in st.session_state:
    st.session_state['data'] = load_data()

df = st.session_state['data']
df_classified = df[df['fiction_type'].notna()].copy()

# Color scheme
colors = {'speculative': '#e74c3c', 'realistic': '#3498db', 'other': '#95a5a6'}

# ============================================
# HOME PAGE CONTENT
# ============================================

st.title("Wikipedia Novels: Speculative vs. Realistic Fiction")
st.markdown("""
### Overview

The dataset I worked with contains **709 novels from Wikiproject Novels**, and this project 
explores the relationship between literary period and a novel's fiction type (i.e. speculative vs. 
realistic fiction). After first getting the data, I had 944 articles. However, I filtered through
using "instance of" to determine which articles were truly about novels as opposed to actors or
films, which gave me my 709 articles.

---

### Research Question

**Main Question:** Is there a relationship between literary period and fiction genre?

**Sub-Question:** Does literary period improve genre classification?

---
            
### Dataset Summary
""")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total", len(df))
with col2:
    st.metric("Speculative", len(df[df['fiction_type']=='speculative']))
with col3:
    st.metric("Realistic", len(df[df['fiction_type']=='realistic']))
with col4:
    st.metric("Other", len(df[df['fiction_type']=='other']))
with col5:
    st.metric("Unclassified", len(df[df['fiction_type'].isna()]))

st.markdown("---")

st.markdown("### Fiction Type Definitions")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Speculative Fiction**
    - Fantasy
    - Science Fiction
    - Horror
    - Gothic
    - Dystopian/Utopian
    - Fairy Tales
    - Magic Realism
    - etc.
    """)
with col2:
    st.markdown("""
    **Realistic Fiction**
    - Historical Fiction
    - Romance
    - Mystery/Crime
    - Autobiography/Memoir
    - Literary Fiction
    - etc.
    """)

st.markdown("""
---

### Findings
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Text-Only Model", "69.4%")
with col2:
    st.metric("Text + Period Model", "75.8%", delta="+6.4%")
with col3:
    st.metric("Statistical Significance", "p = 0.50", delta="Not Significant", delta_color="inverse")

st.markdown("""
**Main Finding:** Although adding literary period did increase text classificatoin accuracy 
by 6.4% points, conducting a hypothesis test showed that this was not statistically significant
(p = 0.50). Therefore, the increase in accuracy may be due to chance.
""")

st.markdown("---")

st.markdown("### Change Over Time")

decade_counts = pd.crosstab(df_classified['decade'], df_classified['fiction_type'])

fig = go.Figure()
for ftype in ['realistic', 'speculative', 'other']:
    if ftype in decade_counts.columns:
        fig.add_trace(go.Scatter(
            x=decade_counts.index, y=decade_counts[ftype],
            name=ftype.title(), mode='lines+markers',
            fill='tonexty' if ftype != 'realistic' else 'tozeroy',
            line=dict(color=colors[ftype], width=2), stackgroup='one'
        ))

fig.update_layout(
    title='Novels Published by Decade',
    xaxis_title='Decade',
    yaxis_title='Number of Novels',
    hovermode='x unified',
    height=500
)
st.plotly_chart(fig, use_container_width=True, key="home_timeline")

st.markdown("**Observation:** Speculative fiction seems to become more dominant from 1950.")

st.markdown("---")

st.markdown("### Fiction Types by Literary Period")

period_pct = pd.crosstab(df_classified['literary_period'], df_classified['fiction_type'], 
                         normalize='index') * 100

fig = go.Figure()
for ftype in ['realistic', 'speculative', 'other']:
    if ftype in period_pct.columns:
        fig.add_trace(go.Bar(
            name=ftype.title(), x=period_pct.index, y=period_pct[ftype],
            marker_color=colors[ftype],
            text=period_pct[ftype].round(1),
            texttemplate='%{text}%',
            textposition='inside'
        ))

fig.update_layout(
    title='Fiction Type Distribution by Literary Period (%)',
    xaxis_title='Literary Period',
    yaxis_title='Percentage',
    barmode='stack',
    height=500
)
st.plotly_chart(fig, use_container_width=True, key="home_period")

st.markdown("**Observation:** Genre distribution does seem (visually) to change across period.")

st.markdown("---")

st.markdown("### Takeaways")

st.markdown("""
    - 6.4% accuracy increase observed but was not statistically significant
    - 75.8% accuracy remains above a 50% baseline of random guessing
    - Speculative fiction seems to have grown in 20th century
    - Relationship exists visually, but period doesn't improve prediction
    """)