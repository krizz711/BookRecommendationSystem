import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from urllib.request import urlopen
from urllib.error import HTTPError

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("book_with_semantic_categories.csv")
    embeddings = np.load("book_embeddings.npy")
    index = faiss.read_index("book_index.faiss")
    return df, embeddings, index

df, embeddings, index = load_data()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to get book cover image from ISBN
def get_cover_url(isbn):
    if pd.isna(isbn):
        return "https://via.placeholder.com/120x180?text=No+Cover"
    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    try:
        urlopen(url)  # check if image exists
        return url
    except HTTPError:
        return "https://via.placeholder.com/120x180?text=No+Cover"

# Page title
st.title("📚 Book Recommender")

# Layout with tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📂 Category", "😊 Emotion"])

# --- Tab 1: Query Search ---
with tab1:
    st.subheader("Search by Description")
    query = st.text_input("Describe what you're looking for", "a thrilling science fiction adventure")

    category_filter = st.selectbox("Filter by Category (optional)", ["All"] + sorted(df["semantic_category"].dropna().unique()))
    emotion_filter = st.selectbox("Filter by Emotion (optional)", ["All"] + sorted(df["emotion_tone"].dropna().unique()))

    if st.button("🔍 Search"):
        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k=10)
        results = df.iloc[I[0]]

        # Apply filters
        if category_filter != "All":
            results = results[results["semantic_category"] == category_filter]
        if emotion_filter != "All":
            results = results[results["emotion_tone"] == emotion_filter]

        if not results.empty:
            for _, row in results.iterrows():
                st.markdown("---")
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(get_cover_url(str(row.get("isbn13", ""))), width=120)
                with col2:
                    st.markdown(f"### **{row['title']}**")
                    st.markdown(f"**Category:** {row['semantic_category']} | **Emotion:** {row['emotion_tone']}")
                    st.markdown(f"**Description:** {row['description'][:300]}...")
                    with st.expander("Full description"):
                        st.write(row['description'])
        else:
            st.warning("No results found.")

# --- Tab 2: Category Filter ---
with tab2:
    st.subheader("Browse by Category")
    selected_category = st.selectbox("Select Category", ["All"] + sorted(df["semantic_category"].unique()))
    cat_df = df if selected_category == "All" else df[df["semantic_category"] == selected_category]
    st.write(f"Found {len(cat_df)} books")
    for _, row in cat_df.head(10).iterrows():
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(get_cover_url(str(row.get("isbn13", ""))), width=120)
        with col2:
            st.markdown(f"### {row['title']}")
            st.markdown(f"**Category:** {row['semantic_category']}")
            st.markdown(f"**Description:** {row['description'][:300]}...")

# --- Tab 3: Emotion Filter ---
with tab3:
    st.subheader("Browse by Emotion")
    selected_emotion = st.selectbox("Select Emotion", ["All"] + sorted(df["emotion_tone"].unique()))
    emo_df = df if selected_emotion == "All" else df[df["emotion_tone"] == selected_emotion]
    st.write(f"Found {len(emo_df)} books")
    for _, row in emo_df.head(10).iterrows():
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(get_cover_url(str(row.get("isbn13", ""))), width=120)
        with col2:
            st.markdown(f"### {row['title']}")
            st.markdown(f"**Emotion:** {row['emotion_tone']}")
            st.markdown(f"**Description:** {row['description'][:300]}...")
