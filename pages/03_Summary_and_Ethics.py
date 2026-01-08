# This was written with the help of Claude AI, but substantial written text is mine.

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Summary and Ethics", page_icon="", layout="wide")

# Load data from session state
if 'data' not in st.session_state:
    st.error("Data not loaded. Please return to Home page first.")
    st.stop()

df = st.session_state['data']

st.title("Summary and Ethics")
st.markdown("---")

st.markdown("### Summary")

st.markdown("""
    - 6.4% accuracy increase observed but was not statistically significant
    - 75.8% model accuracy remains above a 50% baseline of random guessing
    - Speculative fiction seems to have grown in 20th century
    - Relationship exists visually, but period doesn't improve prediction
    - Geographic/cultural biases in dataset
    """)

st.markdown("---")

# Limitations
st.header("Limitations")

st.markdown("""
**1. Class Imbalance**
            
The training data consisted of 56% speculative novels and 44% realistic novels, so 
the model was likely biased toward speculative fiction. This is reflected by the model's
high recall for speculative (100%) and low recall for realistic (46%). If I had more time, I might
try to use class weights to see if this could improve the model.

**2. Sample Size**

Overall, I had a relatively smaller sample size. Again, if I had more time, I would have liked
to see if I could've expanded my dataset somehow. 

**3. Geographic Bias**
            
I acknowledge that my data is very US/UK-centric, and even the literary periods I chose may not
correlate well to non-Western literature. Also, since I focused on English Wikipedia, there
may be a bias towards English novels.

**4. Genre Simplification**

Some novels were unclassified because they did not have sufficient text Wikidata. Furthermore,
some novels weren't categorized by genre due to ambiguity. "Genre" may also be culturally significant,
and some countries or regions may have genres that don't fall neatly into speculative or realistic. 

**5. Text Features**

Based on the Wikidata I could get, I only used descriptions and first/last lines (where available), 
not the full text of a novel. Not many novels had first/last lines. It would be more interesting to perform
this text analysis with excerpts from the novels. 
""")

st.markdown("---")

# Ethics
st.header("Ethical Considerations")

st.markdown("""
- Cultural biases may not generalize to non-Western novels
- Binary genre classification (speculative vs. realistic) might ignore non-Western novels
- Binary classification oversimplifies more complex novels
- Wikidata itself reflects bias towards English language/Western novels
""")