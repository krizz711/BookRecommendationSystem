# 📚 Semantic Book Recommendation System

A content-based book recommendation system that uses semantic embeddings to suggest similar books based on descriptions and categories. Built with **Sentence Transformers**, **FAISS**, and a **Streamlit** web interface.

---

## 🚀 How It Works

- **Embeddings** are generated from book descriptions using `all-MiniLM-L6-v2`.
- **FAISS index** enables fast similarity search over the vector space.
- **Web Interface** (Streamlit) allows users to input book descriptions and get recommendations.
- **Optional** category filtering enhances results with genre-aware suggestions.

---

## 🗂️ Project Structure

| File / Folder                   | Description |
|--------------------------------|-------------|
| `webpage.py`                   | Streamlit web interface for book recommendations |
| `bookEmbeddings.py`            | Generates semantic embeddings and FAISS index |
| `testBookSearch.py`            | Command-line test for similarity search |
| `semantic_book_categories.py`  | Classifies books into genres/categories using semantic analysis |
| `book_cleaned.csv`             | Cleaned dataset with book descriptions |
| `book_with_semantic_categories.csv` | Dataset enriched with semantic genre/category labels |
| `book_metadata.csv`            | Processed book metadata for output |
| `book_embeddings.npy`          | Numpy file containing sentence embeddings |
| `book_index.faiss`             | FAISS index for vector search |
---

## 📊 Data Source

The book metadata was sourced from the Kaggle dataset [**7k+ Books with Metadata**](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata). It was downloaded using the [`kagglehub`](https://github.com/kagglehub/kagglehub) library:



## 🔧 Cleaning Process

The original dataset was cleaned and saved as book_cleaned.csv. Cleaning involved:

Removing unused or inconsistent columns

Handling missing values (e.g., filling missing descriptions with empty strings)

Renaming and standardizing columns

Ensuring the dataset is ready for semantic embedding and indexing

This cleaned CSV is the primary input used for generating embeddings and building the recommendation system.


## ⚙️ Installation

Install the required packages:

pip install sentence-transformers faiss-cpu streamlit pandas numpy



## 🧪 Usage
Step 1: Generate Embeddings & Index

python bookEmbeddings.py

This creates:

book_embeddings.npy

book_metadata.csv

book_index.faiss

Step 2: Run the Web App

streamlit run webpage.py

Step 3: Test via Command Line

python testBookSearch.py

(Optional) Run Category Classifier

python semantic_book_categories.py

 
## 🌟 Features
Semantic search for book descriptions

Genre/category-aware recommendations

Fast and scalable with FAISS

Minimal UI for interaction (Streamlit)

## 🔮 Future Ideas
Add personalized user profiles

Emotion-aware recommendations

Connect with Goodreads / Google Books APIs

## 📜 License
MIT License. Feel free to use, share, and modify.
