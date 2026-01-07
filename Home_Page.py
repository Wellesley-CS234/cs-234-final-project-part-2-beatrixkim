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

st.title("Distinguishing Speculative from Realistic Fiction")
st.markdown("""
### Overview

This project analyzes **709 novels from Wikidata** to investigate the relationship between 
literary period and fiction type (speculative vs. realistic).

---

### Research Question

**Main Question:** Is there a significant relationship between literary period and fiction type?

**Sub-Question:** Can this temporal relationship improve automated genre classification?

---

### Key Findings
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Text-Only Model", "69.4%")
with col2:
    st.metric("Text + Period Model", "75.8%", delta="+6.4%")
with col3:
    st.metric("Statistical Significance", "p = 0.50", delta="Not Significant")

st.info("""
**Main Finding:** While adding literary period improves accuracy by 6.4 percentage points (69.4% → 75.8%), 
this improvement is **not statistically significant** (McNemar's test, p = 0.50). The observed difference 
may be due to chance rather than a true effect of temporal context.
""")

st.markdown("---")

st.markdown("### Fiction Type Definitions")

col1, col2 = st.columns(2)
with col1:
    st.info("""
    **Speculative Fiction**
    - Fantasy, Science Fiction
    - Horror, Gothic
    - Dystopian/Utopian
    - Fairy Tales, Magic Realism
    """)
with col2:
    st.info("""
    **Realistic Fiction**
    - Historical Fiction, Romance
    - Mystery, Crime, Thriller
    - Autobiography, Memoir
    - Literary Fiction
    """)

st.markdown("---")

st.markdown("### Evolution Over Time")

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

st.markdown("**Observation:** Speculative fiction shows significant growth starting in the mid-20th century.")

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

st.markdown("**Observation:** Distinct temporal patterns explain why literary period improves classification.")

st.markdown("---")

st.markdown("### Implications")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **For Computational Analysis:**
    - 6.4% accuracy improvement observed but not statistically significant
    - Small test set (62 novels) limits statistical power
    - Model exhibits class imbalance bias (56% speculative training data)
    - 75.8% accuracy still above 50% baseline
    """)
with col2:
    st.markdown("""
    **For Literary Understanding:**
    - Descriptive patterns show genre evolution over time
    - Speculative fiction grew significantly in 20th century
    - Relationship exists visually but doesn't improve prediction
    - Future work needs larger datasets for significance testing
    """)

# Sidebar
st.sidebar.markdown(f"""
**Dataset Summary**
- Total novels: **{len(df)}**
- Speculative: {len(df[df['fiction_type']=='speculative'])}
- Realistic: {len(df[df['fiction_type']=='realistic'])}
- Other/Ambiguous: {len(df[df['fiction_type']=='other'])}
- Unclassified: {len(df[df['fiction_type'].isna()])}
""")