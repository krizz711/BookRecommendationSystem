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

| File / Folder              | Description |
|---------------------------|-------------|
| `webpage.py`              | Streamlit web interface for book recommendations |
| `bookEmbeddings.py`       | Generates semantic embeddings and FAISS index |
| `testBookSearch.py`       | Command-line test for similarity search |
| `semantic_book_categories.py` | Classifies books into genres/categories |
| `book_cleaned.csv`        | Cleaned dataset with book descriptions |
| `book_metadata.csv`       | Processed book metadata for output |
| `book_embeddings.npy`     | Numpy file containing sentence embeddings |
| `book_index.faiss`        | FAISS index for vector search |

---

## ⚙️ Installation

Install the required packages:

```bash
pip install sentence-transformers faiss-cpu streamlit pandas numpy
🧪 Usage
Step 1: Generate Embeddings & Index
bash
Copy
Edit
python bookEmbeddings.py
This creates:

book_embeddings.npy

book_metadata.csv

book_index.faiss

Step 2: Run the Web App
bash
Copy
Edit
streamlit run webpage.py
Step 3: Test via Command Line
bash
Copy
Edit
python testBookSearch.py
(Optional) Run Category Classifier
bash
Copy
Edit
python semantic_book_categories.py
🌟 Features
Semantic search for book descriptions

Genre/category-aware recommendations

Fast and scalable with FAISS

Minimal UI for interaction (Streamlit)

🔮 Future Ideas
Add personalized user profiles

Emotion-aware recommendations

Connect with Goodreads / Google Books APIs

📜 License
MIT License. Feel free to use, share, and modify.
