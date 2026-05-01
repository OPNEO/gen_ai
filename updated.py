"""
    Full pipeline:
    embeddings → FAISS → query search
"""

import os
import numpy as np
import faiss
import google.generativeai as genai

genai.configure(api_key="AIzaSyBKmdG0lR9TfwrwpZmkXdXniJv3Z-YyjM4")

texts = [
    "kafka handles real time data",
    "kafka is a streaming platform",
    "spark processes big data",
    "databricks is analytics platform",
    "pizza is tasty"
]

"""
    Generate embedding for given text.
"""
def get_embedding(text):
    response = genai.embed_content(
        model="gemini-embedding-2",
        content=text
    )
    return response["embedding"]


# Step 1: Generate embeddings
embeddings = [get_embedding(text) for text in texts]

embedding_matrix = np.array(embeddings).astype("float32")

print("Shape:", embedding_matrix.shape)


# Step 2: FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

print("Total vectors:", index.ntotal)


# Step 3: Query
query = "real time streaming systems"

query_embedding = np.array([get_embedding(query)]).astype("float32")

k = 2

distances, indices = index.search(query_embedding, k)


# Step 4: Results
print("\n--- Results ---")

for i, dist in zip(indices[0], distances[0]):
    print(f"{texts[i]} → distance: {dist}")