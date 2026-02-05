import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ------------------------------------------------
# LOAD FILES (SAFE + CLOUD FRIENDLY)
# ------------------------------------------------
@st.cache_resource
def load_data():
    df = joblib.load("movies_df.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    indices = joblib.load("indices.pkl")
    return df, tfidf_matrix, indices

df, tfidf_matrix, indices = load_data()

# ------------------------------------------------
# UTILITY
# ------------------------------------------------
def get_single_index(idx):
    if isinstance(idx, (list, tuple, np.ndarray, pd.Series)):
        return int(idx[0])
    return int(idx)

# ------------------------------------------------
# RECOMMEND FUNCTION
# ------------------------------------------------
def recommend(title, n=10):
    if title not in indices:
        return []

    idx = get_single_index(indices[title])

    # compute similarity for one movie only (RAM safe)
    sim_scores = linear_kernel(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    sorted_idx = np.argsort(sim_scores)[::-1]
    sorted_idx = sorted_idx[sorted_idx != idx]
    top_idx = sorted_idx[:n]

    return df["title"].iloc[top_idx].tolist()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
with st.sidebar:
    st.markdown("## 🎥 Movie Recommender")
    st.write("Built using NLP & Machine Learning 🚀")

    num_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    st.markdown("---")
    st.markdown("👨‍💻 **Tech Stack**")
    st.markdown("- Python")
    st.markdown("- Scikit-learn")
    st.markdown("- Streamlit")

# ------------------------------------------------
# MAIN UI
# ------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🎬 Movie Recommendation System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Select a movie and get similar movie recommendations instantly"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Movie selection
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    selected_movie = st.selectbox(
        "🎞 Choose a Movie",
        sorted(df["title"].unique())
    )

    recommend_btn = st.button(
        "🚀 Get Recommendations",
        use_container_width=True
    )

# ------------------------------------------------
# RESULTS
# ------------------------------------------------
if recommend_btn:
    with st.spinner("Finding the best movies for you..."):
        results = recommend(selected_movie, num_recommendations)

    st.markdown("## 📌 Recommended Movies")

    if not results:
        st.warning("Movie not found in the dataset.")
    else:
        cols = st.columns(5)
        for i, movie in enumerate(results):
            with cols[i % 5]:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#1f2933;
                        padding:15px;
                        border-radius:12px;
                        margin-bottom:15px;
                        text-align:center;
                        box-shadow:0 4px 8px rgba(0,0,0,0.2);
                    ">
                        <h4 style="color:#f9fafb;">🎬</h4>
                        <p style="color:#e5e7eb; font-size:14px;">
                            {movie}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "Creator: Yagnik Patel 😎"
    "</p>",
    unsafe_allow_html=True
)
