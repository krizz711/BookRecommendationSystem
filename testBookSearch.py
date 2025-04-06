import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load saved embeddings and FAISS index
df = pd.read_csv("book_metadata.csv")
embeddings = np.load("book_embeddings.npy")
index = faiss.read_index("book_index.faiss")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to get book recommendations
def recommend_books(query, k=10):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)  # Search top k books
    recommended_books = df.iloc[I[0]][["isbn13", "title", "description"]]
    return recommended_books

# Example usage
query = "a book to teach children about nature"
print(recommend_books(query))

 