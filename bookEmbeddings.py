import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the cleaned book data
df = pd.read_csv("book_cleaned.csv")
df["description"].fillna("", inplace=True)

# Load the SentenceTransformer model (semantic vectorizer)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert descriptions into embeddings (vectors)
embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)

# Save the embeddings and metadata
np.save("book_embeddings.npy", embeddings)
df[["isbn13", "title", "description"]].to_csv("book_metadata.csv", index=False)

# Build and save the FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "book_index.faiss")

print("✅ FAISS index and embeddings saved successfully.")
 