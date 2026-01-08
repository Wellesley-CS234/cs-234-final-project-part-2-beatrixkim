# This was written with the help of Claude AI, but substantial written text is mine.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Interactive Visualizations", page_icon="", layout="wide")

# Load data from session state
if 'data' not in st.session_state:
    st.error("Data not loaded. Please return to Home page first.")
    st.stop()

df = st.session_state['data']
df_classified = df[df['fiction_type'].notna()].copy()

colors = {'speculative': '#e74c3c', 'realistic': '#3498db', 'other': '#95a5a6'}

st.title("Interactive Visualizations")
st.markdown("---")

# VIZ 1: Timeline with slider
st.header("1. Timeline Explorer")

min_year = int(df_classified['publication_year'].min())
max_year = int(df_classified['publication_year'].max())
year_range = st.slider("Year Range:", min_year, max_year, (min_year, max_year), 10)

df_filtered = df_classified[
    (df_classified['publication_year'] >= year_range[0]) &
    (df_classified['publication_year'] <= year_range[1])
]

col1, col2 = st.columns([3, 1])
with col1:
    fig = px.histogram(df_filtered, x='publication_year', color='fiction_type', nbins=50,
                      title=f'Novels ({year_range[0]}-{year_range[1]})',
                      color_discrete_map=colors, barmode='stack')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, key="viz1")

with col2:
    st.metric("Total", len(df_filtered))
    st.metric("Speculative", len(df_filtered[df_filtered['fiction_type']=='speculative']))
    st.metric("Realistic", len(df_filtered[df_filtered['fiction_type']=='realistic']))
    st.metric("Other", len(df_filtered[df_filtered['fiction_type']=='other']))

st.markdown("---")

# VIZ 2: Geographic
st.header("2. Geographic Distribution")

st.markdown("""
**Note:** I consolidated the following country groups:
- **United Kingdom**: Includes "United Kingdom", "England", "United Kingdom of Great Britain and Ireland", 
  "Kingdom of Great Britain", "Great Britain", etc.
- **United States**: Includes "United States" and "United States of America"
""")

top_countries = df_classified['country_consolidated'].value_counts().head(15).index.tolist()
selected = st.multiselect("Select Countries to Display:", top_countries, default=top_countries)

if selected:
    df_geo = df_classified[df_classified['country_consolidated'].isin(selected)]
    
    country_counts = pd.crosstab(df_geo['country_consolidated'], df_geo['fiction_type'])
    country_counts = country_counts.loc[country_counts.sum(axis=1).sort_values(ascending=False).index]
    
    fig = px.bar(country_counts, barmode='group', title='Fiction Types by Country',
                color_discrete_map=colors, height=500)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key="viz2a")

st.markdown("---")

# VIZ 3: Heatmap
st.header("3. Period × Country Heatmap")

col1, col2, col3 = st.columns(3)
with col1:
    show_spec = st.checkbox("Speculative", True)
with col2:
    show_real = st.checkbox("Realistic", True)
with col3:
    show_other = st.checkbox("Other", True)

types = []
if show_spec: types.append('speculative')
if show_real: types.append('realistic')
if show_other: types.append('other')

if types:
    df_hm = df_classified[df_classified['fiction_type'].isin(types)]
    top10 = df_hm['country_consolidated'].value_counts().head(10).index
    df_hm = df_hm[df_hm['country_consolidated'].isin(top10)]
    
    heatmap = pd.crosstab(df_hm['literary_period'], df_hm['country_consolidated'])
    fig = px.imshow(heatmap.values, x=heatmap.columns, y=heatmap.index,
                   color_continuous_scale='YlOrRd', text_auto=True,
                   title='Period × Country Heatmap')
    fig.update_layout(height=500)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key="viz3")

st.markdown("---")

# VIZ 4: Explorer
st.header("4. Novel Explorer")

col1, col2, col3, col4 = st.columns(4)
with col1:
    fiction_filter = st.multiselect("Fiction:", ['speculative', 'realistic', 'other'],
                                    default=['speculative', 'realistic', 'other'])
with col2:
    period_filter = st.multiselect("Period:", sorted(df_classified['literary_period'].unique()),
                                   default=sorted(df_classified['literary_period'].unique()))
with col3:
    country_filter = st.multiselect("Country:",
                                    sorted(df_classified['country_consolidated'].value_counts().head(20).index),
                                    default=[])
with col4:
    source = st.radio("Source:", ['All', 'Ground Truth', 'Predicted'])

df_exp = df_classified[
    (df_classified['fiction_type'].isin(fiction_filter)) &
    (df_classified['literary_period'].isin(period_filter))
]

if country_filter:
    df_exp = df_exp[df_exp['country_consolidated'].isin(country_filter)]

if source == 'Ground Truth':
    df_exp = df_exp[df_exp['is_predicted'] == False]
elif source == 'Predicted':
    df_exp = df_exp[df_exp['is_predicted'] == True]

st.markdown(f"**Showing {len(df_exp)} novels**")

fig = px.scatter(df_exp, x='publication_year', y='fiction_type', color='fiction_type',
                hover_data=['label', 'author'], color_discrete_map=colors, opacity=0.6)
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True, key="viz5")

st.dataframe(df_exp[['label', 'author', 'fiction_type', 'publication_year', 
                     'literary_period', 'country_consolidated']].sort_values('publication_year', ascending=False).head(50),
            use_container_width=True)