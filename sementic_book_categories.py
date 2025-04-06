# Semantic classification of book categories and emotions using zero-shot learning

import pandas as pd
from transformers import pipeline

# Load data
df = pd.read_csv("book_cleaned.csv")
df["description"] = df["description"].fillna("").astype(str)

# Load zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define updated top 10 target categories
categories = [
    "Fiction",
    "Children / Juvenile",
    "Biography",
    "History",
    "Philosophy & Religion",
    "Comics / Graphic Novels",
    "Drama / Poetry / Literature",
    "Science",
    "Business & Self-help",
    "Lifestyle (Travel / Cooking / Health)"
]

# Define emotion labels
emotions = [
    "Joyful", "Sad", "Dark", "Inspiring", "Wholesome", "Suspenseful", "Neutral"
]

# Apply classification to each book
def classify_row(desc):
    cat = classifier(desc, categories, multi_label=False)
    emo = classifier(desc, emotions, multi_label=False)
    return pd.Series([cat["labels"][0], emo["labels"][0]])

# Run classification (may take time)
df[["semantic_category", "emotion_tone"]] = df["description"].apply(classify_row)

# Save result
df.to_csv("book_with_semantic_categories.csv", index=False)
print("✅ Semantic category and emotion labels saved to 'book_with_semantic_categories.csv'")