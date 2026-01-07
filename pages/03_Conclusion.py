import streamlit as st
import pandas as pd

st.set_page_config(page_title="About & Ethics", page_icon="", layout="wide")

# Load data from session state
if 'data' not in st.session_state:
    st.error("Data not loaded. Please return to Home page first.")
    st.stop()

df = st.session_state['data']

st.title("About & Ethics")
st.markdown("---")

# Dataset Details
st.header("Dataset Details")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Source:** Wikidata
    - Size: 709 novels
    - Time: 1605-2024
    - Countries: 30+
    - Languages: Primarily English
    """)
with col2:
    st.markdown("""
    **Features:**
    - Genre labels (Wikidata)
    - Text (title, description, first line)
    - Publication metadata
    - Geographic data
    """)

sample = df[['label', 'author', 'fiction_type', 'publication_year', 'country_grouped']].sample(10)
st.dataframe(sample, use_container_width=True)

st.markdown("---")

# Limitations
st.header("Limitations")

st.warning("""
**1. Class Imbalance**
- Training: 56% speculative, 44% realistic
- Model biased toward speculative (79% of predictions)
- High recall for speculative (100%), low for realistic (46%)
- Future: use class weights or balanced sampling

**2. Sample Size**
- Only 248 novels for binary classification
- Test set: 62 novels
- Some periods underrepresented

**3. Geographic Bias**
- Heavily US/UK-centric (440/709 novels)
- May not generalize to non-Western traditions
- English-language dominant

**4. Binary Simplification**
- Real novels span multiple genres
- 31 novels in "other" category
- Genre boundaries are cultural constructs

**5. Text Features**
- Used descriptions, not full text
- Descriptions may be marketing language
- Only ~18% had first lines

**6. Temporal Confounding**
- Period correlates with writing style
- Modern descriptions more standardized
- Genre conventions evolved over time
""")

st.markdown("---")

# Ethics
st.header("Ethical Considerations")

st.error("""
**Cultural Bias**
- May not generalize to non-Western traditions

**Genre Stereotyping**
- Binary classification reinforces boundaries

**Algorithmic Reductionism**
- Oversimplifies complex literary works

**Crowd-Sourced Labels**
- Wikidata may reflect systemic biases

**Prediction Errors**
- 137 novels have potentially inaccurate predicted labels
""")

st.markdown("---")

# Conclusion
st.header("Conclusion")

st.success("""
**Achievements:**
- 75.8% accuracy (25.8% above baseline)
- Demonstrated temporal patterns in genre evolution
- Perfect recall on speculative fiction (100%)
- Identified class imbalance as key limitation

**Caveats:**
- **Improvement not statistically significant** (p = 0.50)
- Small test set (62 novels) limits statistical power
- Class imbalance bias (favors speculative predictions)
- Realistic recall only 46.4%
- Geographic/cultural bias in dataset
- Computational methods complement human scholarship
""")

st.info("""
**Future Work:**
- Class-balanced sampling or weighted loss
- Expand dataset
- Include non-Western traditions
- Full-text analysis
- Multi-class classification
""")