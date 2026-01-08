# This was written with the help of Claude AI, but substantial written text is mine.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Methodology", page_icon="", layout="wide")

# Load data from session state
if 'data' not in st.session_state:
    st.error("Data not loaded. Please return to Home page first.")
    st.stop()

df = st.session_state['data']
df_classified = df[df['fiction_type'].notna()].copy()

colors = {'speculative': '#e74c3c', 'realistic': '#3498db', 'other': '#95a5a6'}

st.title("Methodology & Classification")
st.markdown("---")

# Data Collection
st.header("1. Data Collection")
st.markdown("**Source:** Wikidata queries")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total", len(df))
with col2:
    st.metric("With Genres", len(df[df['genre'].notna()]))
with col3:
    st.metric("With Descriptions", len(df[df['description'].notna()]))
with col4:
    st.metric("Countries", df['country_grouped'].nunique())

st.markdown("---")

# Feature Engineering
st.header("2. Feature Engineering")

st.subheader("2.1 Fiction Type")
st.markdown("""
Mapped Wikidata novel genres into three categories:
- **Speculative** (405): fantasy, sci-fi, horror, dystopian, etc.
- **Realistic** (227): historical, romance, mystery, etc.
- **Other** (31): ambiguous or multi-genre
""")

fiction_dist = df_classified['fiction_type'].value_counts()
fig = px.bar(x=fiction_dist.index, y=fiction_dist.values, color=fiction_dist.index,
            color_discrete_map=colors, title='Fiction Type Distribution')
st.plotly_chart(fig, use_container_width=True, key="method1")

st.subheader("2.2 Literary Period")
period_table = pd.DataFrame({
    'Period': ['Classical', 'Romantic', 'Victorian', 'Modernist', 'Postwar', 'Contemporary', 'Modern'],
    'Years': ['< 1800', '1800-1836', '1837-1900', '1901-1944', '1945-1969', '1970-1999', '2000+']
})
st.table(period_table)

st.subheader("2.3 Text Features")
st.markdown("In terms of text to analyze, I used a combinations of a novel's title, description, " \
"and where available, first line, last line, and epigraph. " \
"During text cleaning, I removed proper nouns and numbers.")

st.markdown("---")

# Classification
st.header("3. Text Classification")

st.markdown("""
For genre classification, I split novels into one of three categories: speculative, realistic, or other, based
on their given genre. I trained two models: one with text only, and one with text + period. 

I filtered the dataset for novels where the cleaned text contained at least 15 characters. This removed novels whose text
consisted mostly of metadata like author names and publication years. For feature extraction, however, I used the original text 
to keep any words wrongly classified as proper nouns due to capitalization (in the case of novel titles). 


**Breakdown:** Speculative vs. Realistic (248 novels, 80/20 split)

**Model Used:** Random Forest (n=200, depth=20)

**Features:** CountVectorizer (1000 features) + Literary Period
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Model 1: Text Only** - Accuracy: **69.4%**")
with col2:
    st.markdown("**Model 2: Text + Period** - Accuracy: **75.8%** (+6.4%)")

results = pd.DataFrame({'Model': ['Text Only', 'Text + Period'], 'Accuracy': [69.4, 75.8]})

fig = go.Figure(data=[
    go.Bar(x=results['Model'], y=results['Accuracy'], marker_color=['#3498db', '#2ecc71'],
           text=results['Accuracy'].apply(lambda x: f'{x}%'), textposition='outside')
])
fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Baseline")
fig.update_layout(title='Model Performance', yaxis_range=[0, 100], height=400)
st.plotly_chart(fig, use_container_width=True, key="method2")

# Confusion Matrix
st.subheader("Confusion Matrix for Model 2")

confusion_data = {
    'Predicted Realistic': [13, 0],
    'Predicted Speculative': [15, 34]
}
confusion_df = pd.DataFrame(confusion_data, index=['Actually Realistic', 'Actually Speculative'])

confusion_pct = confusion_df.div(confusion_df.sum(axis=1), axis=0) * 100

fig = px.imshow(confusion_pct.values, x=['Realistic', 'Speculative'], y=['Realistic', 'Speculative'],
               color_continuous_scale='Blues', text_auto='.1f',
               labels=dict(color="Percentage (%)"), title='Confusion Matrix (% of actual class)')
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True, key="method3")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Interpretation:**
    - We see a  100% recall on speculative fiction (34/34)
    - We see a 46.4% recall on realistic fiction (13/28)
    - The model predicts speculative 79% of time (49/62)
    - In the training data, 56% speculative, 44% realistic
    - Therefore, the model favors speculative
    """)
with col2:
    st.dataframe(confusion_df)

st.markdown("---")

# Hypothesis Testing
st.header("4. Hypothesis Testing")

st.markdown("""
**Test:** McNemar's test (for paired predictions)

**H₀:** Period does NOT improve classification

**H₁:** Period DOES improve classification

**Test Results:**
- Test statistic: 0.0000
- p-value: 0.5000
- Significance level (α): 0.05
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Observed Difference", "6.4%")
with col2:
    st.metric("p-value", "0.5000")
with col3:
    st.metric("α", "0.05")

st.markdown("""
**Result:** Fail to reject null hypothesis (p = 0.5000 > 0.05)

Although Model 2 had a higher accuracy than Model 1 (75.8% vs 69.4%), the improvement is not statistically significant.
Again, this suggests that the increase in accuracy may be due to chance.
""")