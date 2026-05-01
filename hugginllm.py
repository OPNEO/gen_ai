"""
    FULL LOCAL RAG PIPELINE (Stable Version)

    - Embeddings: SentenceTransformers
    - Vector DB: FAISS
    - LLM: FLAN-T5 (manual load, no pipeline issues)

    Works reliably on Windows + PyCharm
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


"""
    Step 1: Load embedding model
"""
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


"""
    Step 2: Documents
"""
texts = [
    "Kafka is a distributed streaming platform used for real-time data pipelines",
    "Kafka is scalable and fault-tolerant",
    "Spark is used for batch processing",
    "Databricks is built on Spark",
    "Pizza is tasty"
]


"""
    Step 3: Generate embeddings
"""
embeddings = embedding_model.encode(texts)

print("Embedding shape:", embeddings.shape)
print("First vector (first 5 values):", embeddings[0][:5])


"""
    Step 4: Cosine similarity (learning purpose)
"""
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


print("\n--- Similarity ---")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        score = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{texts[i]} <--> {texts[j]} = {score:.4f}")


"""
    Step 5: FAISS index
"""
embedding_matrix = embeddings.astype("float32")

dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

print("\nVectors stored:", index.ntotal)


"""
    Step 6: Query
"""
query = "real time streaming"

query_embedding = embedding_model.encode([query]).astype("float32")

k = 2

distances, indices = index.search(query_embedding, k)

print("\n--- FAISS Results ---")
for i, dist in zip(indices[0], distances[0]):
    print(f"{texts[i]} → distance: {dist}")


"""
    Step 7: Build context
"""
retrieved_docs = [texts[i] for i in indices[0]]
context = "\n".join(retrieved_docs)

print("\n--- Context ---")
print(context)


"""
    
"""
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


"""
    Step 9: RAG Prompt
"""
prompt = f"""
You are a senior data engineer.

Answer the question using ONLY the context below.

Write a complete and clear explanation in 1-2 sentences.

Context:
{context}

Question:
What is Kafka?

Answer:
"""


"""
    Step 10: Generate answer
"""
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


print("\n--- Final Answer ---")
print(answer)